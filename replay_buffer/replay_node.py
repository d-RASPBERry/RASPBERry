from abc import ABC
from utils import get_action_dim, get_obs_shape
from ray.rllib.policy.policy import SampleBatch
from gymnasium import spaces
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)
    :param buffer_size: Max number of element in the buffer
    :param obs_space: obs space
    :param action_space: Action space to which the values will be converted
    """

    def __init__(
            self,
            buffer_size: int,
            obs_space: spaces.Space,
            action_space: spaces.Space,
            randomly: bool = False,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.action_space = get_action_dim(action_space)
        self.obs_space = get_obs_shape(obs_space)
        self.pos = 0
        self.full = False
        self.randomly = randomly
        self.obs = np.zeros(np.concatenate([[self.buffer_size], self.obs_space]), dtype=obs_space.dtype)
        self.new_obs = np.zeros(np.concatenate([[self.buffer_size], self.obs_space]), dtype=obs_space.dtype)
        self.actions = np.zeros(self.buffer_size, dtype=np.int32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminateds = np.zeros(self.buffer_size, dtype=np.int32)
        self.truncateds = np.zeros(self.buffer_size, dtype=np.int32)
        self.weights = np.zeros(self.buffer_size, dtype=np.float32)
        self.t = np.zeros(self.buffer_size, dtype=np.int32)

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, batch: SampleBatch) -> None:
        """
        Add elements to the buffer.
        """
        shape = batch.get("obs").shape[0]
        self.obs[self.pos:self.pos + shape] = np.array(batch.get("obs"))
        self.new_obs[self.pos:self.pos + shape] = np.array(batch.get("new_obs"))
        self.actions[self.pos:self.pos + shape] = batch.get("actions")
        self.rewards[self.pos:self.pos + shape] = batch.get("rewards")
        self.terminateds[self.pos:self.pos + shape] = batch.get("terminateds")
        self.truncateds[self.pos:self.pos + shape] = batch.get("truncateds")
        self.weights[self.pos:self.pos + shape] = batch.get("weights")
        self.pos += shape
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def extend(self, *args) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self):
        upper_bound = self.buffer_size if self.full else self.pos
        if self.randomly:
            batch_ids = np.random.randint(0, upper_bound, size=self.size())
        else:
            batch_ids = np.array(range(0, self.size()))

        data = SampleBatch(
            {
                "obs": self.obs[batch_ids, :],
                "new_obs": self.new_obs[batch_ids, :],
                "actions": self.actions[batch_ids],
                "rewards": self.rewards[batch_ids],
                "terminateds": self.terminateds[batch_ids],
                "truncateds": self.truncateds[batch_ids],
                "weights": self.weights[batch_ids],
                "shape": np.array([self.obs[batch_ids, :].shape])
            }
        )
        return data
