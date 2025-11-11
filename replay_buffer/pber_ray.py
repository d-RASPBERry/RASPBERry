"""Prioritized Block Experience Replay (PBER) Buffer.

Block-level replay buffer WITHOUT compression (pure PBER baseline).

Key features:
- Block-level storage: O(M/m) operations instead of O(M)
- No compression: direct numpy array storage
- Compatible with RLlib's PER interface
- Clean separation from RASPBERry (compression-enabled variant)

Design:
- Uses BlockAccumulator for simple data batching
- Stores raw numpy blocks directly (no compression overhead)
- Block-level priority management

See Chapter 3 (PBER) in thesis for algorithm details.
"""

import logging
import numpy as np
from gymnasium.spaces import Space
from typing import Dict, Optional, Any, List
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import (
    PrioritizedReplayBuffer,
)
from replay_buffer.block_accumulator import BlockAccumulator
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import concat_samples

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PrioritizedBlockReplayBuffer(PrioritizedReplayBuffer):
    """Prioritized Block Experience Replay (PBER) buffer.
    
    Pure block-level replay without compression.
    This is the PBER baseline for comparison with RASPBERry.
    """

    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        sub_buffer_size: int = 32,
        **kwargs,
    ):
        """Initialize PBER buffer.
        
        Args:
            obs_space: Observation space
            action_space: Action space
            sub_buffer_size: Block size (transitions per block)
            **kwargs: Additional args for PrioritizedReplayBuffer
        """
        super(PrioritizedBlockReplayBuffer, self).__init__(**kwargs)

        self.sub_buffer_size = sub_buffer_size
        self.obs_space = obs_space
        self.action_space = action_space

        # Block accumulator (简单的数据累积器)
        self.block_accumulator = BlockAccumulator(
            block_size=sub_buffer_size,
            obs_space=obs_space,
            action_space=action_space,
        )

        # Metrics
        self._metrics = {
            "num_blocks": 0,
            "num_transitions": 0,
        }

        logger.info(
            "Initialized PBER buffer: block_size=%d, capacity=%d blocks",
            sub_buffer_size,
            self.capacity,
        )

    def stats(self, debug: bool = False) -> Dict[str, Any]:
        """Compute memory usage statistics.
        
        Returns:
            Dict with est_size_bytes and num_entries
        """
        total_size = 0
        
        # 遍历所有存储的blocks
        for sample_batch in self._storage:
            if "obs" in sample_batch and hasattr(sample_batch["obs"], "nbytes"):
                # PBER stores raw numpy arrays
                total_size += sample_batch["obs"].nbytes
            
            if "new_obs" in sample_batch and hasattr(
                sample_batch["new_obs"], "nbytes"
            ):
                total_size += sample_batch["new_obs"].nbytes

        out = {
            "est_size_bytes": total_size,
            "num_entries": len(self._storage) * self.sub_buffer_size,
            "num_blocks": len(self._storage),
        }
        return out

    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Add a batch to the buffer, organizing into blocks.
        
        Args:
            batch: SampleBatch with transitions to add
            **kwargs: Additional args (unused)
        """
        if not isinstance(batch, SampleBatch):
            return

        # 使用BlockAccumulator填充blocks
        idx = 0
        count = len(batch)
        
        while idx < count:
            space = self.block_accumulator.block_size - self.block_accumulator.pos
            take = min(space, count - idx)
            slice_batch = batch.slice(idx, idx + take)
            self.block_accumulator.add(slice_batch)
            idx += take

            # Block满了就存储
            if self.block_accumulator.is_full():
                self._store_block()

    def _store_block(self) -> None:
        """Store the current block to buffer."""
        # Extract block data and weight
        raw_batch, weight = self.block_accumulator.flush()
        
        # Validate weight
        if np.isnan(weight) or weight <= 0:
            weight = 0.01
        
        # Store directly (no compression)
        self._add_single_batch(raw_batch, weight=weight)
        
        # Reset accumulator
        self.block_accumulator.reset()
        
        # Update metrics
        self._metrics["num_blocks"] = len(self._storage)
        self._metrics["num_transitions"] = (
            len(self._storage) * self.sub_buffer_size
        )

    def sample(
        self, num_items: int, beta: Optional[float] = None, **kwargs
    ) -> Optional[SampleBatch]:
        """Sample blocks and expand metadata to transition-level.
        
        Args:
            num_items: Number of blocks to sample
            beta: PER importance sampling exponent (if None, try to use parent's beta or default to 1.0)
            **kwargs: Additional sampling args
            
        Returns:
            SampleBatch with expanded weights/batch_indexes, or None if buffer empty
        """
        if len(self._storage) == 0:
            return None
        
        # If beta not provided, try to get from parent class or default to 1.0
        if beta is None:
            beta = getattr(self, 'beta', 1.0)
        
        try:
            batch = super(PrioritizedBlockReplayBuffer, self).sample(
                num_items, beta=beta, **kwargs
            )
        except ValueError as e:
            if "empty buffer" in str(e).lower():
                return None
            raise

        if batch is None:
            return None

        # Expand block-level weights/batch_indexes to transition-level
        num_transitions = len(batch.get("actions", batch.get("rewards", [])))
        self._expand_block_field(batch, "weights", num_transitions)
        self._expand_block_field(batch, "batch_indexes", num_transitions)

        return batch
    
    def _expand_block_field(
        self, batch: SampleBatch, field_name: str, target_size: int
    ) -> None:
        """Expand block-level field to transition-level by replicating values.
        
        Args:
            batch: SampleBatch to modify in-place
            field_name: Field to expand (weights, batch_indexes)
            target_size: Target size (number of transitions)
        """
        if field_name not in batch or len(batch[field_name]) == target_size:
            return
        
        replicate_factor = target_size // len(batch[field_name])
        batch[field_name] = np.repeat(batch[field_name], replicate_factor)

    def _encode_sample(self, idxes: List[int]) -> SampleBatch:
        """Encode samples - returns raw block data.
        
        Args:
            idxes: Block indices to retrieve
            
        Returns:
            Concatenated SampleBatch with raw numpy arrays
        """
        batch_list = []
        for i in idxes:
            self._hit_count[i] += 1
            batch_list.append(self._storage[i])

        # Concatenate blocks
        return concat_samples(batch_list)

