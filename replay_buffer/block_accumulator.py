"""Block Accumulator - Simple data buffer for PBER.

Pure block-level data accumulation without compression logic.
Used by PBER (Prioritized Block Experience Replay) to batch transitions.
"""

import numpy as np
from typing import Tuple
from gymnasium import spaces
from ray.rllib.policy.sample_batch import SampleBatch
from utils import get_obs_shape
import logging

logger = logging.getLogger(__name__)


class BlockAccumulator(object):
    """
    Block accumulator for PBER (without compression).

    Design principles:
    1) Pre-allocate all necessary memory at initialization time
    2) Clear dtype and shape definitions for all tensors
    3) No compression logic (pure PBER baseline)
    4) Support different observation/action space types
    5) Robust memory monitoring and error handling
    """

    def __init__(
            self,
            block_size: int,
            obs_space: spaces.Space,
            action_space: spaces.Space,
    ):
        """
        Initialize the block accumulator.
        Args:
            block_size: secondary block size
            obs_space: observation space (supports Box, Discrete, etc.)
            action_space: action space (supports Discrete, Box, etc.)
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got: {block_size}")

        self.block_size = block_size
        self.obs_space = obs_space
        self.action_space = action_space

        # State management
        self.pos = 0
        self.full = False

        self.obs_shape = get_obs_shape(obs_space)

        # Pre-allocate observation buffers
        obs_buffer_shape = (self.block_size,) + tuple(self.obs_shape)
        self.obs = np.zeros(obs_buffer_shape, dtype=obs_space.dtype)
        self.new_obs = np.zeros(obs_buffer_shape, dtype=obs_space.dtype)

        # Action space support (Discrete and Box)
        if isinstance(action_space, spaces.Discrete):
            self.actions = np.zeros(self.block_size, dtype=action_space.dtype)
        else:
            self.actions = np.zeros((self.block_size, *action_space.shape), dtype=action_space.dtype)

        # Standard RL fields (keep dtypes aligned with BaseBuffer)
        self.rewards = np.zeros(self.block_size, dtype=np.float32)
        self.terminateds = np.zeros(self.block_size, dtype=np.int32)
        self.truncateds = np.zeros(self.block_size, dtype=np.int32)
        self.weights = np.zeros(self.block_size, dtype=np.float32)
    
    def add(self, batch: SampleBatch) -> None:
        """Add a SampleBatch to the node's buffers."""
        if not isinstance(batch, SampleBatch) or len(batch) == 0:
            raise ValueError

        batch_size = len(batch["obs"])
        available_space = self.block_size - self.pos
        actual_size = min(batch_size, available_space)

        if actual_size == 0:
            raise ValueError

        # Store data
        end_pos = self.pos + actual_size
        slice_range = slice(self.pos, end_pos)
        data_slice = slice(None, actual_size)

        self.obs[slice_range] = batch["obs"][data_slice]
        self.new_obs[slice_range] = batch["new_obs"][data_slice]
        self.actions[slice_range] = batch["actions"][data_slice]
        self.rewards[slice_range] = batch["rewards"][data_slice]
        self.terminateds[slice_range] = batch["terminateds"][data_slice]
        self.truncateds[slice_range] = batch["truncateds"][data_slice]

        # Handle weights
        weights = batch.get("weights")
        self.weights[slice_range] = weights[data_slice] if weights is not None else 1.0

        # Update state
        self.pos = end_pos
        if self.pos >= self.block_size:
            self.full = True
    
    def flush(self) -> Tuple[SampleBatch, float]:
        """
        Sample and return raw data (PBER baseline without compression).

        Core steps:
        1) Extract data slice
        2) Compute block-level weight
        3) Return raw SampleBatch and weight

        Returns:
            Tuple[raw_sample_batch, block_weight]

        Raises:
            ValueError: if the node is empty
        """
        if not self.full and self.pos == 0:
            raise ValueError("Node is empty; cannot sample data")

        size = self.size()
        
        # Compute block-level weight (mean of sample weights)
        block_weight = float(np.mean(self.weights[:size]))
        if np.isnan(block_weight) or block_weight <= 0:
            block_weight = 0.01  # minimal weight guard

        # Create raw SampleBatch (no compression, no transpose)
        raw_batch = SampleBatch({
            "obs": self.obs[:size].copy(),
            "new_obs": self.new_obs[:size].copy(),
            "actions": self.actions[:size].copy(),
            "rewards": self.rewards[:size].copy(),
            "terminateds": self.terminateds[:size].copy(),
            "truncateds": self.truncateds[:size].copy(),
            "weights": self.weights[:size].copy(),
        })

        return raw_batch, block_weight

    def size(self) -> int:
        """Return current number of stored samples."""
        return self.block_size if self.full else self.pos

    def reset(self) -> None:
        """Reset node state (keep allocated memory)."""
        self.pos = 0
        self.full = False
        if logger.isEnabledFor(logging.INFO):
            logger.info("BlockAccumulator reset")

    def is_full(self) -> bool:
        """Whether the accumulator is full (ready to flush)."""
        return self.full

    def __repr__(self) -> str:
        return (f"BlockAccumulator(size={self.pos}/{self.block_size}, "
                f"obs_shape={self.obs_shape})")


