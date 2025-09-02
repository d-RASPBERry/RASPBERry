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
        self.obs_space = get_obs_shape(obs_space)
        self.pos = 0
        self.full = False
        self.randomly = randomly
        self.action_space = action_space  # 保存引用但不强制使用
        
        # 预分配观察空间（这个是确定的）
        self.obs = np.zeros(np.concatenate([[self.buffer_size], self.obs_space]), dtype=obs_space.dtype)
        self.new_obs = np.zeros(np.concatenate([[self.buffer_size], self.obs_space]), dtype=obs_space.dtype)
        
        # 按 action_space 分配动作缓冲区（严格依赖环境声明）
        if isinstance(action_space, spaces.Discrete):
            self.actions = np.zeros(self.buffer_size, dtype=np.int32)
        elif isinstance(action_space, spaces.Box):
            action_shape = action_space.shape
            self.actions = np.zeros((self.buffer_size, *action_shape), dtype=np.float32)
        elif isinstance(action_space, spaces.MultiDiscrete):
            action_shape = action_space.shape
            self.actions = np.zeros((self.buffer_size, *action_shape), dtype=np.int32)
        else:
            if hasattr(action_space, 'shape') and len(action_space.shape) > 0:
                self.actions = np.zeros((self.buffer_size, *action_space.shape), dtype=np.float32)
            else:
                self.actions = np.zeros(self.buffer_size, dtype=np.int32)
        
        # 其他字段正常初始化
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
        """添加元素到缓冲区"""
        shape = batch.get("obs").shape[0]
        
        # 边界检查
        if self.pos + shape > self.buffer_size:
            shape = self.buffer_size - self.pos
            
        # 存储数据
        self.obs[self.pos:self.pos + shape] = np.array(batch.get("obs")[:shape])
        self.new_obs[self.pos:self.pos + shape] = np.array(batch.get("new_obs")[:shape])
        
        # 动作数据（根据 action_space 分配的缓冲区写入；允许 (B,1)->(B,) 的轻量兼容）
        actions_data = batch.get("actions")[:shape]
        if self.actions.ndim == 1 and actions_data.ndim > 1:
            self.actions[self.pos:self.pos + shape] = actions_data.flatten()[:shape]
        else:
            self.actions[self.pos:self.pos + shape] = actions_data
            
        self.rewards[self.pos:self.pos + shape] = batch.get("rewards")[:shape]
        self.terminateds[self.pos:self.pos + shape] = batch.get("terminateds")[:shape]
        self.truncateds[self.pos:self.pos + shape] = batch.get("truncateds")[:shape]
        
        batch_weights = batch.get("weights")
        if batch_weights is not None:
            self.weights[self.pos:self.pos + shape] = batch_weights[:shape]
        else:
            self.weights[self.pos:self.pos + shape] = 1.0
            
        self.pos += shape
        if self.pos >= self.buffer_size:
            self.full = True

    def extend(self, *args) -> None:
        """
        Add a new batch of transitions to the buffer
        """
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
            }
        )
        return data


class CompressedReplayNode(BaseBuffer):
    """支持压缩存储的replay node，在sample时自动解压缩"""
    
    def __init__(self, *args, compress_base: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.compress_base = compress_base
        self._compressed_storage = {}  # 存储压缩后的数据
        self._compressed = False
        
    def compress_and_store(self, compressed_data, weight: float = 1.0):
        """存储压缩数据而不是原始数据"""
        self._compressed_storage = compressed_data
        self._compressed = True
        self.weights.fill(weight)  # 统一设置权重
        
    def sample(self):
        """sample时自动处理压缩数据"""
        if not self._compressed:
            return super().sample()
            
        from .mpber_ram_saver_v8 import decompress_sample_batch
        return decompress_sample_batch(self._compressed_storage, self.compress_base)
        
    def is_compressed(self) -> bool:
        """检查是否存储了压缩数据"""
        return self._compressed
