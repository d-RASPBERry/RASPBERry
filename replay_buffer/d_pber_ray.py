"""Multi-Agent Prioritized Block Experience Replay (PBER) Buffer.

Multi-agent wrapper for PBER, providing per-policy block replay buffers.

Key operations:
- add(): Route batches to per-policy buffers
- sample(): Sample from per-policy buffers (no decompression needed for PBER)
- update_priorities(): Aggregate transition-level TD-errors to block-level priorities

Design:
- Uses PrioritizedBlockReplayBuffer (PBER) for each policy
- No compression/decompression logic (pure block-level storage)
- Compatible with distributed training (Ape-X)

See Chapter 3 (PBER) and Chapter 5 (Distributed PBER) in thesis.
"""

# ====== Section: Imports ======
# ------ Subsection: Standard library ------
import logging
from typing import Dict, Optional

# ------ Subsection: Third-party ------
import numpy as np
from gymnasium.spaces import Space
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
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
from replay_buffer.pber_ray import PrioritizedBlockReplayBuffer

# ====== Section: Module State ======
logger = logging.getLogger(__name__)

# ====== Section: Classes ======
@DeveloperAPI
class MultiAgentPrioritizedBlockReplayBuffer(MultiAgentPrioritizedReplayBuffer):
    """Multi-agent PBER buffer (no compression).
    
    Distributed version of PBER for Ape-X style architectures.
    """

    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        sub_buffer_size: int = 32,
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
        **kwargs,
    ):
        """Initialize Multi-Agent PBER buffer.
        
        Args:
            obs_space: Observation space
            action_space: Action space
            sub_buffer_size: Block size (transitions per block)
            capacity: Buffer capacity in transitions
            prioritized_replay_alpha: PER alpha parameter
            prioritized_replay_beta: PER beta parameter
            prioritized_replay_eps: Small constant for numerical stability
            **kwargs: Additional args for MultiAgentPrioritizedReplayBuffer
        """
        if "replay_mode" in kwargs and (
            kwargs["replay_mode"] == "lockstep"
            or kwargs["replay_mode"] == ReplayMode.LOCKSTEP
        ):
            logger.error(
                "Lockstep mode not supported, falling back to independent mode"
            )
            kwargs["replay_mode"] = "independent"

        self.sub_buffer_size = sub_buffer_size
        self.prioritized_replay_eps = float(prioritized_replay_eps)
        self.prioritized_replay_beta = float(prioritized_replay_beta)
        self.rollout_fragment_length = rollout_fragment_length

        pber_config = {
            "type": PrioritizedBlockReplayBuffer,
            "action_space": action_space,
            "obs_space": obs_space,
            "storage_unit": StorageUnit.FRAGMENTS,
            "sub_buffer_size": sub_buffer_size,
            "prioritized_replay_alpha": prioritized_replay_alpha,
            "prioritized_replay_beta": prioritized_replay_beta,
            "prioritized_replay_eps": prioritized_replay_eps,
        }

        # Capacity management: capacity is in transitions (NOT blocks)
        # Ray's ReplayBuffer uses capacity as the threshold for _num_timesteps_added_wrap
        # Each block contributes block.count (= sub_buffer_size) transitions to this counter
        self._configured_capacity_transitions = int(capacity)
        self._top_storage_unit = storage_unit

        pber_config["capacity"] = self._configured_capacity_transitions

        super(MultiAgentPrioritizedBlockReplayBuffer, self).__init__(
            capacity=self._configured_capacity_transitions,
            storage_unit=StorageUnit.FRAGMENTS,
            num_shards=num_shards,
            replay_mode=replay_mode,
            replay_sequence_override=replay_sequence_override,
            replay_sequence_length=replay_sequence_length,
            replay_burn_in=replay_burn_in,
            replay_zero_init_states=replay_zero_init_states,
            underlying_buffer_config=pber_config,
            **kwargs,
        )

        self._capacity = self._configured_capacity_transitions

    def _ensure_sample_batch(
        self, sample: Optional[SampleBatchType]
    ) -> Optional[SampleBatchType]:
        """Normalize sampled outputs to SampleBatch when needed."""
        if sample is None:
            return None
        if isinstance(sample, SampleBatch):
            return sample
        if isinstance(sample, dict):
            return SampleBatch(sample)
        return sample

    def _maybe_split_into_policy_batches(self, batch: SampleBatchType) -> Dict:
        """Split batch into per-policy sub-batches."""
        if isinstance(batch, MultiAgentBatch):
            return batch.policy_batches
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
                    time_slices = sample_batch.timeslices(
                        size=self.rollout_fragment_length
                    )
                    for s_batch in time_slices:
                        self._add_to_underlying_buffer(policy_id, s_batch)

        self._num_added += batch.count

    @override(MultiAgentPrioritizedReplayBuffer)
    def sample(
        self,
        num_items: int,
        policy_id: PolicyID = None,
        beta: Optional[float] = None,
        **kwargs,
    ) -> Optional[SampleBatchType]:
        """Sample from the buffer.
        
        For PBER, no decompression is needed (raw numpy arrays).
        
        Args:
            num_items: Number of blocks to sample
            policy_id: Policy ID to sample from
            beta: PER beta parameter
            **kwargs: Additional sampling args
            
        Returns:
            MultiAgentBatch
        """
        # NOTE: We do NOT call `super().sample()` here.
        #
        # RLlib's MultiAgentPrioritizedReplayBuffer.sample() assumes underlying buffers
        # raise on empty. Our block buffers (PBER/RASPBERry) return None instead.
        # Calling the upstream method would crash on `sample.count` when any policy
        # buffer is empty. We therefore mirror RLlib's logic with explicit None handling.
        kwargs = merge_dicts_with_warning(self.underlying_buffer_call_args, kwargs)

        # Accept beta either as explicit arg or via kwargs (API parity).
        if beta is None:
            beta = kwargs.pop("beta", None)
        if beta is None:
            beta = getattr(self, "prioritized_replay_beta", 0.4)

        with self.replay_timer:
            # Lockstep mode: sample from all policies at the same time.
            if self.replay_mode == ReplayMode.LOCKSTEP:
                assert (
                    policy_id is None
                ), "`policy_id` specifier not allowed in `lockstep` mode!"
                raw_sample = self.replay_buffers["__all__"].sample(
                    num_items, beta=beta, **kwargs
                )
                raw_sample = self._ensure_sample_batch(raw_sample)
                return raw_sample if raw_sample is not None else None

            # Independent mode: sample from a single policy.
            if policy_id is not None:
                sample = self.replay_buffers[policy_id].sample(
                    num_items, beta=beta, **kwargs
                )
                sample = self._ensure_sample_batch(sample)
                if sample is None:
                    return None
                return MultiAgentBatch({policy_id: sample}, sample.count)

            # Independent mode: sample from all policies and merge.
            samples = {}
            for pid, replay_buffer in self.replay_buffers.items():
                sample = replay_buffer.sample(num_items, beta=beta, **kwargs)
                sample = self._ensure_sample_batch(sample)
                if sample is None:
                    continue
                samples[pid] = sample

            if not samples:
                return None
            return MultiAgentBatch(samples, sum(s.count for s in samples.values()))

    @override(MultiAgentPrioritizedReplayBuffer)
    def update_priorities(self, prio_dict: Dict) -> None:
        """Update priorities with block-level aggregation.
        
        Aggregates transition-level TD-errors to block-level priorities.
        
        Args:
            prio_dict: Dict mapping policy_id to (batch_indexes, td_errors)
        """
        with self.update_priorities_timer:
            for policy_id, (batch_indexes, td_errors) in prio_dict.items():
                buffer = self.replay_buffers.get(policy_id)
                if buffer is None:
                    continue

                block_indices, block_priorities = self._convert_to_block_priorities(
                    batch_indexes, td_errors
                )
                if len(block_indices) == 0:
                    continue
                buffer.update_priorities(block_indices, block_priorities)

    def _convert_to_block_priorities(
        self, batch_indexes: np.ndarray, td_errors: np.ndarray
    ) -> tuple:
        """Aggregate transition-level TD-errors to block-level priorities.

        Returns:
            (block_indices, block_priorities): Block indices and aggregated priorities

        Note:
            Falls back to transition-level if aggregation fails.
        """
        try:
            block_indexes = np.asarray(batch_indexes)
            td_errors_arr = np.asarray(td_errors, dtype=np.float32)

            unique_block_indexes = np.unique(block_indexes)
            if unique_block_indexes.size == 0:
                return unique_block_indexes, np.asarray([], dtype=np.float32)

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
            return np.asarray(batch_indexes), priorities

    @override(MultiAgentPrioritizedReplayBuffer)
    def stats(self, debug: bool = False) -> Dict:
        """Get buffer statistics.
        
        Returns:
            Dict with per-policy and aggregate statistics
        """
        data = {
            "add_batch_time_ms": round(1000 * self.add_batch_timer.mean, 3),
            "replay_time_ms": round(1000 * self.replay_timer.mean, 3),
            "update_priorities_time_ms": round(
                1000 * self.update_priorities_timer.mean, 3
            ),
        }
        for policy_id, replay_buffer in self.replay_buffers.items():
            data["policy_{}".format(policy_id)] = replay_buffer.stats()

        # Align with other replay buffers: expose total transitions seen.
        data["added_count"] = int(getattr(self, "_num_added", 0))

        # Add PBER-specific metadata
        data["sub_buffer_size"] = self.sub_buffer_size
        data["configured_capacity_transitions"] = (
            self._configured_capacity_transitions
        )

        return data

    def get_host(self) -> str:
        """Get host identifier for distributed training."""
        import socket
        return socket.gethostname()

    def apply(self, func, *args, **kwargs):
        """Apply a function to this replay buffer actor."""
        return func(self, *args, **kwargs)

