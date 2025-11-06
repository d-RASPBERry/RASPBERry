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

import logging
import numpy as np
from gymnasium.spaces import Space
from replay_buffer.pber_ray import PrioritizedBlockReplayBuffer
from typing import Dict, Optional, Any
from ray.rllib.utils.typing import PolicyID, SampleBatchType
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.replay_buffers import StorageUnit
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import (
    ReplayMode,
    merge_dicts_with_warning,
)
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import (
    MultiAgentPrioritizedReplayBuffer,
)
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.util.debug import log_once

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
            if log_once("lockstep_mode_not_supported"):
                logger.error(
                    "Lockstep mode not supported, falling back to independent mode"
                )
            kwargs["replay_mode"] = "independent"

        # Store block-level parameters
        self.sub_buffer_size = sub_buffer_size
        self.prioritized_replay_eps = float(prioritized_replay_eps)

        # PBER configuration
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

        # Pass capacity directly (in transitions) to underlying buffer
        pber_config["capacity"] = self._configured_capacity_transitions

        logger.info(
            "MultiAgentPBER: capacity=%d transitions (block_size=%d, max_blocks~%d)",
            self._configured_capacity_transitions,
            sub_buffer_size,
            self._configured_capacity_transitions // sub_buffer_size,
        )

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

        # Override capacity tracking (in transitions)
        self._capacity = self._configured_capacity_transitions

        logger.info(
            "Initialized MultiAgentPBER: block_size=%d, capacity=%d transitions",
            self.sub_buffer_size,
            self._configured_capacity_transitions,
        )

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
            SampleBatch or MultiAgentBatch
        """
        # Use parent's sampling logic
        sampled_batch = super(
            MultiAgentPrioritizedBlockReplayBuffer, self
        ).sample(num_items=num_items, policy_id=policy_id, beta=beta, **kwargs)

        # PBER stores raw numpy arrays, no decompression needed
        return sampled_batch

    @override(MultiAgentPrioritizedReplayBuffer)
    def update_priorities(self, prio_dict: Dict) -> None:
        """Update priorities with block-level aggregation.
        
        Aggregates transition-level TD-errors to block-level priorities.
        
        Args:
            prio_dict: Dict mapping policy_id to (batch_indexes, td_errors)
        """
        for policy_id, (batch_indexes, td_errors) in prio_dict.items():
            buffer = self.replay_buffers.get(policy_id)
            if buffer is None:
                continue

            # Convert transition-level indexes/errors to block-level
            block_indexes = batch_indexes // self.sub_buffer_size
            unique_block_indexes = np.unique(block_indexes)

            # Aggregate TD-errors per block (use max for conservative priority)
            block_priorities = []
            for block_idx in unique_block_indexes:
                mask = block_indexes == block_idx
                block_td_error = np.max(np.abs(td_errors[mask]))
                block_priorities.append(block_td_error)

            block_priorities = np.array(block_priorities, dtype=np.float32)

            # Update underlying buffer with block-level priorities
            buffer.update_priorities(unique_block_indexes, block_priorities)

            logger.info(
                "[PBER] Updated priorities for policy %s: %d transitions -> %d blocks",
                policy_id,
                len(batch_indexes),
                len(unique_block_indexes),
            )

    @override(MultiAgentPrioritizedReplayBuffer)
    def stats(self, debug: bool = False) -> Dict:
        """Get buffer statistics.
        
        Returns:
            Dict with per-policy and aggregate statistics
        """
        data = super(MultiAgentPrioritizedBlockReplayBuffer, self).stats(
            debug=debug
        )

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
        """Apply a function to all underlying buffers."""
        results = {}
        for policy_id, buffer in self.replay_buffers.items():
            results[policy_id] = func(buffer, *args, **kwargs)
        return results

