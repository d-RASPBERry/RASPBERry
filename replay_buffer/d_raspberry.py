import logging
import numpy as np
import time
from gymnasium.spaces import Space
from replay_buffer.raspberry import PrioritizedBlockReplayBuffer, decompress_sample_batch
from typing import Dict, Optional, Any
from ray.rllib.utils.typing import PolicyID, SampleBatchType
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.replay_buffers import StorageUnit
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import ReplayMode, merge_dicts_with_warning
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import MultiAgentPrioritizedReplayBuffer
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.util.debug import log_once

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@DeveloperAPI
class MultiAgentPrioritizedBlockReplayBuffer(MultiAgentPrioritizedReplayBuffer):
    """Multi-agent prioritized block replay buffer."""

    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            sub_buffer_size: int = 1,
            rollout_fragment_length: int = 4,
            capacity: int = 10000,
            storage_unit: str = "timesteps",
            num_shards: int = 1,
            replay_mode: str = "independent",
            replay_sequence_override: bool = True,
            replay_sequence_length: int = 1,
            replay_burn_in: int = 0,
            replay_zero_init_states: bool = True,
            prioritized_replay_alpha: float = 0.6,
            prioritized_replay_beta: float = 0.4,
            prioritized_replay_eps: float = 1e-6,
            compress_base: int = -1,
            compress_pool_size: int = 0,
            compression_algorithm: str = 'zstd',
            compression_level: int = 5,
            compression_nthreads: int = 1,
            **kwargs
    ):
        if "replay_mode" in kwargs and (
                kwargs["replay_mode"] == "lockstep"
                or kwargs["replay_mode"] == ReplayMode.LOCKSTEP
        ):
            if log_once("lockstep_mode_not_supported"):
                logger.error(
                    "Replay mode `lockstep` is not supported for "
                    "MultiAgentPrioritizedReplayBuffer. "
                    "This buffer will run in `independent` mode."
                )
            kwargs["replay_mode"] = "independent"

        # Store block-level parameters for later use
        self.sub_buffer_size = sub_buffer_size
        self.compress_base = compress_base
        # Ensure prioritized_replay_eps is numeric
        self.prioritized_replay_eps = float(prioritized_replay_eps)

        pber_config = {
            "type": PrioritizedBlockReplayBuffer,
            "action_space": action_space,
            "obs_space": obs_space,
            "storage_unit": StorageUnit.FRAGMENTS,
            "sub_buffer_size": sub_buffer_size,
            "prioritized_replay_alpha": prioritized_replay_alpha,
            "prioritized_replay_beta": prioritized_replay_beta,
            "compress_base": compress_base,
            "compress_pool_size": compress_pool_size,
            "compression_algorithm": compression_algorithm,
            "compression_level": compression_level,
            "compression_nthreads": compression_nthreads,
            "prioritized_replay_eps": prioritized_replay_eps,
        }

        MultiAgentPrioritizedReplayBuffer.__init__(
            self,
            capacity=capacity,
            storage_unit=storage_unit,
            num_shards=num_shards,
            replay_mode=replay_mode,
            replay_sequence_override=replay_sequence_override,
            replay_sequence_length=replay_sequence_length,
            replay_burn_in=replay_burn_in,
            replay_zero_init_states=replay_zero_init_states,
            underlying_buffer_config=pber_config,
            prioritized_replay_alpha=prioritized_replay_alpha,
            prioritized_replay_beta=prioritized_replay_beta,
            prioritized_replay_eps=prioritized_replay_eps,
            **kwargs,
        )
        self.rollout_fragment_length = rollout_fragment_length

    @DeveloperAPI
    @override(MultiAgentPrioritizedReplayBuffer)
    def update_priorities(self, prio_dict: Dict) -> None:
        """Update priorities on underlying buffers using block-level logic."""
        with self.update_priorities_timer:
            for policy_id, (batch_indexes, td_errors) in prio_dict.items():

                # Convert to block-level indices/priorities
                block_indices, block_priorities = self._convert_to_block_priorities(
                    batch_indexes, td_errors
                )
                logger.debug(f"Block priority update: "
                             f"{len(batch_indexes)} samples -> {len(block_indices)} blocks")

                # Update underlying per-policy buffer
                if hasattr(self.replay_buffers[policy_id], 'update_priorities'):
                    self.replay_buffers[policy_id].update_priorities(
                        block_indices, block_priorities
                    )
                else:
                    logger.warning(f"Policy {policy_id} replay buffer does not support priority updates")

    def _convert_to_block_priorities(self, batch_indexes: np.ndarray, td_errors: np.ndarray) -> tuple:
        """Convert per-sample (indexes, td_errors) to per-block values.

        Args:
            batch_indexes: original batch indexes [batch_size]
            td_errors: original TD-errors [batch_size]

        Returns:
            tuple: (block_indices, block_priorities)
        """
        try:
            # Reshape into [num_blocks, sub_buffer_size]
            block_indices = batch_indexes.reshape(-1, self.sub_buffer_size)[:, 0]
            block_td_errors = td_errors.reshape(-1, self.sub_buffer_size).mean(axis=1)

            # Compute block-level priorities
            block_priorities = np.abs(block_td_errors) + self.prioritized_replay_eps

            logger.debug(f"Block conversion: {batch_indexes.shape} -> {block_indices.shape}, "
                         f"TD errors: {td_errors.shape} -> {block_td_errors.shape}")

            return block_indices, block_priorities

        except Exception as e:
            logger.warning(f"Block conversion failed: {e}, falling back to direct processing")
            # Fallback to original per-sample processing
            return batch_indexes, np.abs(td_errors) + self.prioritized_replay_eps

    def _maybe_split_into_policy_batches(self, batch: SampleBatchType) -> Dict:
        """Split into policy batches based on replay mode."""
        if isinstance(batch, MultiAgentBatch):
            return batch.policy_batches
        else:
            # If a single SampleBatch, convert to multi-agent format
            return {"__default_policy__": batch}

    @DeveloperAPI
    @override(MultiAgentPrioritizedReplayBuffer)
    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Add a batch to the corresponding policy's underlying replay buffer."""
        if batch is None:
            if log_once("empty_batch_added_to_buffer"):
                logger.info(
                    "A batch that is `None` was added to {}. This can be "
                    "normal at the beginning of execution but might "
                    "indicate an issue.".format(type(self).__name__)
                )
            return

        batch = batch.copy()
        batch = batch.as_multi_agent()

        with self.add_batch_timer:
            pids_and_batches = self._maybe_split_into_policy_batches(batch)
            for policy_id, sample_batch in pids_and_batches.items():
                if len(sample_batch) == 1:
                    self._add_to_underlying_buffer(policy_id, sample_batch)
                else:
                    time_slices = sample_batch.timeslices(size=self.rollout_fragment_length)
                    for s_batch in time_slices:
                        self._add_to_underlying_buffer(policy_id, s_batch)

        self._num_added += batch.count

    @DeveloperAPI
    @override(MultiAgentPrioritizedReplayBuffer)
    def sample(
            self, num_items: int, policy_id: Optional[PolicyID] = None, **kwargs
    ) -> Optional[SampleBatchType]:
        """Sample data and conditionally decompress."""
        t_start = time.time()
        logger.debug(f"Starting sample for policy '{policy_id}' with {num_items} items.")

        kwargs = merge_dicts_with_warning(self.underlying_buffer_call_args, kwargs)

        # Retrieve beta from kwargs or parent/underlying buffer (RLlib may pass annealed beta via kwargs)
        beta = kwargs.pop("beta", None)
        if beta is None:
            beta = getattr(self, "prioritized_replay_beta", None)
        if beta is None:
            try:
                any_buf = next(iter(self.replay_buffers.values()))
                beta = getattr(any_buf, "prioritized_replay_beta", getattr(any_buf, "beta", None))
            except Exception:
                beta = None
        if beta is None:
            beta = 0.4

        with self.replay_timer:
            if self.replay_mode == ReplayMode.LOCKSTEP:
                assert policy_id is None, "`policy_id` specifier not allowed in `lockstep` mode!"

                raw_sample = self.replay_buffers["__all__"].sample(
                    num_items, beta=beta, **kwargs
                )
                import pdb
                pdb.set_trace()
                if raw_sample is None:
                    return None

                # If already decompressed sample, return directly
                if not self._is_compressed(raw_sample):
                    return raw_sample

                return decompress_sample_batch(raw_sample, self.compress_base)

            elif policy_id is not None:
                sample = self.replay_buffers[policy_id].sample(
                    num_items, beta=beta, **kwargs
                )
                if sample is None:
                    return None

                # Conditionally decompress
                if self._is_compressed(sample):
                    sample = decompress_sample_batch(sample, self.compress_base)

                ma_batch = MultiAgentBatch({policy_id: sample}, sample.count)
                logger.info(ma_batch.count)
                return ma_batch

            else:
                # Sample independently from all policies
                samples = {}
                for pid, replay_buffer in self.replay_buffers.items():
                    sample = replay_buffer.sample(num_items, beta=beta, **kwargs)

                    if sample is not None:
                        # Conditionally decompress
                        if self._is_compressed(sample):
                            sample = decompress_sample_batch(sample, self.compress_base)
                        samples[pid] = sample
                if samples:
                    total_count = sum(s.count for s in samples.values())
                    ma_batch = MultiAgentBatch(samples, total_count)
                    return ma_batch
                else:
                    return None

    def _is_compressed(self, sample: SampleBatch) -> bool:
        """Whether the sample is in compressed state (object-typed obs)."""
        return ('obs' in sample and
                isinstance(sample['obs'], np.ndarray) and
                sample['obs'].dtype == object)

    def stats(self, debug: bool = False) -> Dict[str, Any]:
        """Return replay buffer statistics."""
        stat = {
            "add_batch_time_ms": round(1000 * self.add_batch_timer.mean, 3),
            "replay_time_ms": round(1000 * self.replay_timer.mean, 3),
            "update_priorities_time_ms": round(
                1000 * self.update_priorities_timer.mean, 3
            ),
            "est_size_bytes": 0
        }

        total_estimated_bytes = 0
        agg_metrics = {
            "compress_time_ms": 0.0,
            "backpressure_wait_ms": 0.0,
            "decompress_time_ms": 0.0,
            "compress_prep_ms": 0.0,
            "compress_pack_obs_ms": 0.0,
            "compress_pack_new_obs_ms": 0.0,
        }
        for policy_id, replay_buffer in self.replay_buffers.items():
            policy_stats = replay_buffer.stats(debug=debug)
            total_estimated_bytes += policy_stats.get("est_size_bytes", 0)
            # Aggregate per-policy metrics if present
            m = policy_stats.get("metrics") or {}
            for k in agg_metrics.keys():
                agg_metrics[k] += float(m.get(k, 0.0))
            stat.update(
                {"policy_{}".format(policy_id): policy_stats}
            )
        stat["est_size_bytes"] = total_estimated_bytes
        stat["metrics"] = agg_metrics
        return stat
