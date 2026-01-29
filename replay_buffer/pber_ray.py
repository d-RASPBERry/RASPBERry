"""Prioritized Block Experience Replay (PBER) buffer.

Block-level replay buffer without compression (pure PBER baseline).
"""

# ====== Section: Imports ======
# ------ Subsection: Standard library ------
import logging
from typing import Any, Dict, List, Optional

# ------ Subsection: Third-party ------
import numpy as np
from gymnasium.spaces import Space
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import (
    PrioritizedReplayBuffer,
)
from ray.rllib.utils.typing import SampleBatchType

# ------ Subsection: Local ------
from replay_buffer.block_accumulator import BlockAccumulator

# ====== Section: Module State ======
BUFFER_NAME = "pber"
logger = logging.getLogger(__name__)


# ====== Section: Prioritized Block Replay Buffer ======
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
        """Initialize the prioritized block replay buffer.

        Args:
            obs_space: Observation space.
            action_space: Action space.
            sub_buffer_size: Block size (transitions per block).
            **kwargs: Additional args for PrioritizedReplayBuffer.
        """
        # Map repo config keys to RLlib's PrioritizedReplayBuffer args.
        # - Configs use `prioritized_replay_alpha/beta`.
        # - RLlib expects `alpha`, and `beta` is provided in sample().
        if "prioritized_replay_alpha" in kwargs and "alpha" not in kwargs:
            kwargs["alpha"] = kwargs.pop("prioritized_replay_alpha")
        # Keep beta as an attribute so callers can omit beta in sample().
        if "prioritized_replay_beta" in kwargs:
            self.beta = float(kwargs.pop("prioritized_replay_beta"))

        super(PrioritizedBlockReplayBuffer, self).__init__(**kwargs)

        self.sub_buffer_size = sub_buffer_size
        self.obs_space = obs_space
        self.action_space = action_space

        self.block_accumulator = BlockAccumulator(
            block_size=sub_buffer_size,
            obs_space=obs_space,
            action_space=action_space,
        )

        self._metrics = {
            "num_blocks": 0,
            "num_transitions": 0,
        }

    def stats(self, debug: bool = False) -> Dict:
        """Compute memory usage statistics.

        Returns:
            Dict with estimated size and entry counts.
        """
        total_size = 0

        for sample_batch in self._storage:
            if "obs" in sample_batch and hasattr(sample_batch["obs"], "nbytes"):
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
        """Add a batch to the buffer, organizing transitions into blocks.

        Args:
            batch: SampleBatch with transitions to add.
            **kwargs: Additional args (unused).
        """
        if not isinstance(batch, SampleBatch):
            return

        idx = 0
        count = len(batch)

        while idx < count:
            space = self.block_accumulator.block_size - self.block_accumulator.pos
            take = min(space, count - idx)
            slice_batch = batch.slice(idx, idx + take)
            self.block_accumulator.add(slice_batch)
            idx += take

            if self.block_accumulator.is_full():
                self._store_block()

    def _store_block(self) -> None:
        """Store the current block to the buffer."""
        raw_batch, weight = self.block_accumulator.flush()

        min_weight = 0.01
        if np.isnan(weight) or weight <= 0:
            logger.warning(
                "event=invalid_block_weight buffer=%s weight=%s fallback=%s",
                BUFFER_NAME,
                weight,
                min_weight,
            )
            weight = min_weight

        self._add_single_batch(raw_batch, weight=weight)

        self.block_accumulator.reset()

        self._metrics["num_blocks"] = len(self._storage)
        self._metrics["num_transitions"] = (
                len(self._storage) * self.sub_buffer_size
        )

    def sample(
            self, num_items: int, beta: Optional[float] = None, **kwargs
    ) -> Optional[SampleBatch]:
        """Sample blocks and expand metadata to transition-level.

        Args:
            num_items: Number of blocks to sample.
            beta: PER importance sampling exponent (defaults to self.beta if not provided).
            **kwargs: Additional sampling args.

        Returns:
            SampleBatch with expanded weights/batch_indexes, or None if empty.
        """
        if len(self._storage) == 0:
            return None

        if beta is None:
            beta = getattr(self, "beta", 1.0)

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

        num_transitions = len(batch.get("actions", batch.get("rewards", [])))
        self._expand_block_field(batch, "weights", num_transitions)
        self._expand_block_field(batch, "batch_indexes", num_transitions)

        return batch

    def _expand_block_field(
            self, batch: SampleBatch, field_name: str, target_size: int
    ) -> None:
        """Expand a block-level field to transition-level by replication.

        Args:
            batch: SampleBatch to modify in-place.
            field_name: Field to expand (weights, batch_indexes).
            target_size: Target size (number of transitions).
        """
        if field_name not in batch or len(batch[field_name]) == target_size:
            return

        replicate_factor = target_size // len(batch[field_name])
        batch[field_name] = np.repeat(batch[field_name], replicate_factor)

    def _encode_sample(self, idxes: List[int]) -> SampleBatch:
        """Encode samples and return raw block data.

        Args:
            idxes: Block indices to retrieve.

        Returns:
            Concatenated SampleBatch with raw numpy arrays.
        """
        batch_list = []
        for i in idxes:
            self._hit_count[i] += 1
            batch_list.append(self._storage[i])

        return concat_samples(batch_list)
