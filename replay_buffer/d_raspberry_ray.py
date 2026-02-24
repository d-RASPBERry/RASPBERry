"""Multi-Agent Prioritized Block Replay Buffer with Ray-based Compression.

Multi-agent wrapper for RASPBERry, providing per-policy block replay buffers.

Key operations:
- add(): Route batches to per-policy buffers
- sample(): Decompress sampled batches (weights already expanded by underlying buffer)
- update_priorities(): Aggregate transition-level TD-errors to block-level priorities

See docs/raspberr_design.md for architecture details.
"""

# ====== Section: Imports ======
# ------ Subsection: Standard library ------
import logging
from typing import Any, Dict, Optional

# ------ Subsection: Third-party ------
import numpy as np
from gymnasium.spaces import Space
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.replay_buffers import StorageUnit
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import (
    MultiAgentPrioritizedReplayBuffer,
)
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import (
    ReplayMode,
    merge_dicts_with_warning,
)
from ray.rllib.utils.typing import PolicyID, SampleBatchType
from ray.util import log_once

# ------ Subsection: Local ------
from replay_buffer.raspberry_ray import RASPBERryReplayBuffer, decompress_sample_batch

# ====== Section: Module State ======
logger = logging.getLogger(__name__)


# ====== Section: Classes ======
@DeveloperAPI
class MultiAgentRASPBERryReplayBuffer(MultiAgentPrioritizedReplayBuffer):
    """Multi-agent RASPBERry replay buffer with block-level storage and compression."""

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
            num_ray_workers: Optional[int] = None,
            compression_algorithm: str = 'zstd',
            compression_level: int = 5,
            compression_nthreads: int = 1,
            compression_mode: str = "C",  # A: sync, B: batch sync, C: async
            chunk_size: int = 10,
            max_inflight_tasks: Optional[int] = None,
            **kwargs
    ):
        if "replay_mode" in kwargs and (
                kwargs["replay_mode"] == "lockstep"
                or kwargs["replay_mode"] == ReplayMode.LOCKSTEP
        ):
            logger.error("Lockstep mode not supported, falling back to independent mode")
            kwargs["replay_mode"] = "independent"

        self.sub_buffer_size = sub_buffer_size
        self.compress_base = compress_base
        self.prioritized_replay_eps = float(prioritized_replay_eps)
        # Default to one worker per shard, but allow explicit override in config.
        if num_ray_workers is None:
            effective_num_ray_workers = max(1, int(num_shards))
        else:
            effective_num_ray_workers = max(1, int(num_ray_workers))

        pber_config = {
            "type": RASPBERryReplayBuffer,
            "action_space": action_space,
            "obs_space": obs_space,
            "storage_unit": StorageUnit.FRAGMENTS,
            "sub_buffer_size": sub_buffer_size,
            "prioritized_replay_alpha": prioritized_replay_alpha,
            "prioritized_replay_beta": prioritized_replay_beta,
            "compress_base": compress_base,
            # Keep legacy field for compatibility with existing call paths.
            # Worker concurrency is controlled by `num_ray_workers` below.
            "compress_pool_size": compress_pool_size,
            "num_ray_workers": effective_num_ray_workers,
            "compression_algorithm": compression_algorithm,
            "compression_level": compression_level,
            "compression_nthreads": compression_nthreads,
            "compression_mode": compression_mode,
            "chunk_size": chunk_size,
            "max_inflight_tasks": max_inflight_tasks,
            "prioritized_replay_eps": prioritized_replay_eps,
        }

        # Track configured capacity (in transitions) and effective capacity (in blocks)
        configured_capacity_transitions = int(capacity)
        effective_block_capacity = max(1, int(capacity // max(1, sub_buffer_size)))

        self._configured_capacity_transitions = configured_capacity_transitions
        # Protect against divide-by-zero
        self._effective_block_capacity = effective_block_capacity
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
            block_indexes = np.asarray(batch_indexes)
            td_errors_arr = np.asarray(td_errors, dtype=np.float32)
            td_errors_arr = np.nan_to_num(td_errors_arr, nan=0.0, posinf=0.0, neginf=0.0)

            unique_block_indexes = np.unique(block_indexes)
            block_priorities = []
            for block_idx in unique_block_indexes:
                mask = block_indexes == block_idx
                block_td_error = np.abs(td_errors_arr[mask]).mean()
                block_priorities.append(block_td_error)

            block_priorities = np.asarray(block_priorities, dtype=np.float32)
            block_priorities = np.nan_to_num(
                block_priorities, nan=0.0, posinf=0.0, neginf=0.0
            )

            min_priority = max(self.prioritized_replay_eps, 1e-6)
            block_priorities = block_priorities + min_priority
            # RLlib's prioritized buffer asserts `priority > 0` (strict).
            block_priorities = np.maximum(block_priorities, min_priority)
            return unique_block_indexes, block_priorities
        except Exception as e:
            if log_once(f"{type(self).__name__}:block_aggregation_failed"):
                logger.warning(
                    "Block aggregation failed (%s), using transition-level fallback", e
                )
            td_errors_arr = np.asarray(td_errors, dtype=np.float32)
            td_errors_arr = np.nan_to_num(td_errors_arr, nan=0.0, posinf=0.0, neginf=0.0)

            min_priority = max(self.prioritized_replay_eps, 1e-6)
            priorities = np.abs(td_errors_arr) + min_priority
            priorities = np.maximum(priorities, min_priority)
            return batch_indexes, priorities

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
            if log_once(f"{type(self).__name__}:empty_batch_added"):
                logger.warning(
                    "Empty batch added to %s (normal at start, check if persistent)",
                    type(self).__name__,
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
        """Sample from buffer and decompress."""
        kwargs = merge_dicts_with_warning(self.underlying_buffer_call_args, kwargs)

        beta = kwargs.pop("beta", None) or getattr(self, "prioritized_replay_beta", None) or 0.4

        with self.replay_timer:
            if self.replay_mode == ReplayMode.LOCKSTEP:
                assert policy_id is None, "`policy_id` specifier not allowed in `lockstep` mode!"
                raw_sample = self.replay_buffers["__all__"].sample(num_items, beta=beta, **kwargs)
                return decompress_sample_batch(raw_sample, self.compress_base) if raw_sample else None

            elif policy_id is not None:
                sample = self.replay_buffers[policy_id].sample(num_items, beta=beta, **kwargs)
                if sample is None:
                    return None
                sample = decompress_sample_batch(sample, self.compress_base)
                return MultiAgentBatch({policy_id: sample}, sample.count)

            else:
                samples = {}
                for pid, replay_buffer in self.replay_buffers.items():
                    sample = replay_buffer.sample(num_items, beta=beta, **kwargs)
                    if sample is not None:
                        sample = decompress_sample_batch(sample, self.compress_base)
                        samples[pid] = sample

                if samples:
                    return MultiAgentBatch(samples, sum(s.count for s in samples.values()))
                return None

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
            policy_stats = replay_buffer.stats()
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

    def apply(self, func, *args, **kwargs):
        """Apply a function to this replay buffer actor."""
        return func(self, *args, **kwargs)
