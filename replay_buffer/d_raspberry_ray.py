"""Multi-Agent Prioritized Block Replay Buffer with Ray-based Compression.

Multi-agent wrapper for RASPBERry, providing per-policy block replay buffers.

Key operations:
- add(): Route batches to per-policy buffers
- sample(): Decompress sampled batches (weights already expanded by underlying buffer)
- update_priorities(): Aggregate transition-level TD-errors to block-level priorities

See docs/raspberr_design.md for architecture details.
"""

import logging
import numpy as np
from gymnasium.spaces import Space
from replay_buffer.raspberry_ray import PrioritizedBlockReplayBuffer, decompress_sample_batch
from typing import Dict, Optional, Any
from ray.rllib.utils.typing import PolicyID, SampleBatchType
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.replay_buffers import StorageUnit
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import ReplayMode, merge_dicts_with_warning
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import MultiAgentPrioritizedReplayBuffer
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.util.debug import log_once

logger = logging.getLogger(__name__)


@DeveloperAPI
class MultiAgentPrioritizedBlockReplayBuffer(MultiAgentPrioritizedReplayBuffer):
    """Multi-agent prioritized block replay buffer with Ray-based compression."""

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
            compress_pool_size: int = 5,
            compression_algorithm: str = 'zstd',
            compression_level: int = 5,
            compression_nthreads: int = 1,
            compression_mode: str = "D",  # "A": PBER (no compression), "B": sync, "C": batch_ray, "D": async_ray
            chunk_size: int = 10,
            **kwargs
    ):
        if "replay_mode" in kwargs and (
                kwargs["replay_mode"] == "lockstep"
                or kwargs["replay_mode"] == ReplayMode.LOCKSTEP
        ):
            if log_once("lockstep_mode_not_supported"):
                logger.error("Lockstep mode not supported, falling back to independent mode")
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
            "compression_mode": compression_mode,
            "chunk_size": chunk_size,
            "prioritized_replay_eps": prioritized_replay_eps,
        }

        # Track configured capacity (in transitions) and effective capacity (in blocks)
        self._configured_capacity_transitions = int(capacity)
        # Protect against divide-by-zero
        self._effective_block_capacity = max(1, int(capacity // max(1, sub_buffer_size)))
        # Record top-level storage unit for debug/metrics
        self._top_storage_unit = storage_unit

        # Enforce capacity on underlying (block-based) buffer
        pber_config["capacity"] = self._effective_block_capacity

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
        """Aggregate transition-level TD-errors to block-level priorities."""
        with self.update_priorities_timer:
            for policy_id, (batch_indexes, td_errors) in prio_dict.items():
                block_indices, block_priorities = self._convert_to_block_priorities(
                    batch_indexes, td_errors
                )
                
                if hasattr(self.replay_buffers[policy_id], 'update_priorities'):
                    self.replay_buffers[policy_id].update_priorities(
                        block_indices, block_priorities
                    )

    def _convert_to_block_priorities(
        self, batch_indexes: np.ndarray, td_errors: np.ndarray
    ) -> tuple:
        """Aggregate transition-level TD-errors to block-level priorities.
        
        Args:
            batch_indexes: Transition indices from training batch
            td_errors: TD-errors for each transition
            
        Returns:
            (block_indices, block_priorities): Block indices and aggregated priorities
            
        Note:
            Falls back to transition-level if aggregation fails (e.g., shape mismatch)
        """
        try:
            block_indices = batch_indexes.reshape(-1, self.sub_buffer_size)[:, 0]
            block_td_errors = td_errors.reshape(-1, self.sub_buffer_size)
            block_priorities = np.abs(block_td_errors).mean(axis=1) + self.prioritized_replay_eps
            return block_indices, block_priorities
        except Exception as e:
            logger.warning("Block aggregation failed (%s), using transition-level fallback", e)
            return batch_indexes, np.abs(td_errors) + self.prioritized_replay_eps

    def _maybe_split_into_policy_batches(self, batch: SampleBatchType) -> Dict:
        """Split batch into per-policy sub-batches.
        
        Args:
            batch: Input batch (MultiAgentBatch or SampleBatch)
            
        Returns:
            Dict mapping policy_id to SampleBatch
        """
        if isinstance(batch, MultiAgentBatch):
            return batch.policy_batches
        else:
            return {"__default_policy__": batch}

    @DeveloperAPI
    @override(MultiAgentPrioritizedReplayBuffer)
    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Add a batch to the corresponding policy's underlying replay buffer."""
        if batch is None:
            if log_once("empty_batch_added_to_buffer"):
                logger.info("Empty batch added to %s (normal at start, check if persistent)", 
                           type(self).__name__)
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
        """Sample from buffer and decompress (Mode A returns raw data, B/C/D decompress)."""
        kwargs = merge_dicts_with_warning(self.underlying_buffer_call_args, kwargs)

        beta = kwargs.pop("beta", None) or getattr(self, "prioritized_replay_beta", None) or 0.4

        with self.replay_timer:
            if self.replay_mode == ReplayMode.LOCKSTEP:
                assert policy_id is None, "`policy_id` specifier not allowed in `lockstep` mode!"
                raw_sample = self.replay_buffers["__all__"].sample(num_items, beta=beta, **kwargs)
                # Mode A: raw data, no decompression needed
                if self._is_mode_a():
                    return raw_sample
                return decompress_sample_batch(raw_sample, self.compress_base) if raw_sample else None

            elif policy_id is not None:
                sample = self.replay_buffers[policy_id].sample(num_items, beta=beta, **kwargs)
                if sample is None:
                    return None
                # Mode A: raw data, no decompression needed
                if not self._is_mode_a():
                    sample = decompress_sample_batch(sample, self.compress_base)
                return MultiAgentBatch({policy_id: sample}, sample.count)

            else:
                samples = {}
                for pid, replay_buffer in self.replay_buffers.items():
                    sample = replay_buffer.sample(num_items, beta=beta, **kwargs)
                    if sample is not None:
                        # Mode A: raw data, no decompression needed
                        if not self._is_mode_a():
                            sample = decompress_sample_batch(sample, self.compress_base)
                        samples[pid] = sample
                
                if samples:
                    return MultiAgentBatch(samples, sum(s.count for s in samples.values()))
                return None

    def _is_mode_a(self) -> bool:
        """Check if underlying buffer is in Mode A (PBER, no compression)."""
        # Check first buffer's compression mode
        if self.replay_buffers:
            first_buffer = next(iter(self.replay_buffers.values()))
            return getattr(first_buffer, '_compression_mode', 'D') == 'A'
        return False

    def stats(self, debug: bool = False) -> Dict[str, Any]:
        """Return replay buffer statistics."""
        stat = {
            "add_batch_time_ms": round(1000 * self.add_batch_timer.mean, 3),
            "replay_time_ms": round(1000 * self.replay_timer.mean, 3),
            "update_priorities_time_ms": round(
                1000 * self.update_priorities_timer.mean, 3
            ),
            "est_size_bytes": 0,
            "est_compressed_bytes": 0,
            "est_raw_bytes": 0,
            "compression_ratio": 0.0,
            # Added for parity with RLlib PER buffer stats
            "added_count": int(getattr(self, "_num_added", 0)),
        }

        total_estimated_bytes = 0
        total_compressed_bytes = 0
        total_raw_bytes = 0
        total_entries = 0
        # Aggregate metrics across per-policy buffers. The underlying
        # buffer exposes metrics at the top level via stats().
        metric_keys = (
            "compress_time_ms",
            "backpressure_wait_ms",
            "decompress_time_ms",
        )
        agg_metrics = {k: 0.0 for k in metric_keys}
        for policy_id, replay_buffer in self.replay_buffers.items():
            policy_stats = replay_buffer.stats(debug=debug)
            total_estimated_bytes += policy_stats.get("est_size_bytes", 0)
            total_compressed_bytes += policy_stats.get("est_compressed_bytes", 0)
            total_raw_bytes += policy_stats.get("est_raw_bytes", 0)
            total_entries += policy_stats.get("num_entries", 0)
            # Aggregate per-policy metrics from top-level keys if present
            for k in metric_keys:
                if k in policy_stats:
                    try:
                        agg_metrics[k] += float(policy_stats[k])
                    except Exception:
                        pass
            stat.update(
                {"policy_{}".format(policy_id): policy_stats}
            )
        stat["est_size_bytes"] = total_estimated_bytes
        stat["est_compressed_bytes"] = total_compressed_bytes
        stat["est_raw_bytes"] = total_raw_bytes
        stat["compression_ratio"] = (
            (total_compressed_bytes / total_raw_bytes) if total_raw_bytes > 0 else 0.0
        )
        stat["num_entries"] = total_entries
        stat["metrics"] = agg_metrics
        return stat

