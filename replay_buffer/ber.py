import numpy as np
import logging
from gymnasium.spaces import Space
from typing import Optional
from ray.rllib.utils.replay_buffers.utils import SampleBatchType
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer
from replay_buffer.replay_node import BaseBuffer

logger = logging.getLogger(__name__)


class BlockReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            randomly: bool = False,
            sub_buffer_size: int = 8,
            **kwargs
    ):
        super(BlockReplayBuffer, self).__init__(**kwargs)
        self.base_buffer = BaseBuffer(sub_buffer_size, obs_space, action_space, randomly)

    def sample(self, num_items: Optional[int] = None, **kwargs):
        return super(BlockReplayBuffer, self).sample(num_items, **kwargs)

    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Adds a batch of experiences to this buffer.

        Splits batch into chunks of timesteps, sequences or episodes, depending on
        `self._storage_unit`. Calls `self._add_single_batch` to add resulting slices
        to the buffer storage.

        Args:
            batch: Batch to add.
            ``**kwargs``: Forward compatibility kwargs.
        """
        if not batch.count > 0:
            return
        buffer = self.base_buffer

        self.base_buffer.add(batch)
        if buffer.full:
            data = buffer.sample()
            weight = np.mean(data.get("weights"))
            buffer.reset()
            self._add_single_batch(data, weight=weight)
