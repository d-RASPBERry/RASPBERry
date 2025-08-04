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
        
        # 增强动作空间支持，特别是连续动作空间
        if isinstance(action_space, spaces.Discrete):
            self.actions = np.zeros(self.buffer_size, dtype=np.int32)
        elif isinstance(action_space, spaces.Box):
            # SAC等连续控制算法的主要场景
            action_shape = action_space.shape
            self.actions = np.zeros((self.buffer_size, *action_shape), dtype=np.float32)
        elif isinstance(action_space, spaces.MultiDiscrete):
            action_shape = action_space.shape
            self.actions = np.zeros((self.buffer_size, *action_shape), dtype=np.int32)
        else:
            # 更智能的回退策略 - 根据action_space的属性推断
            if hasattr(action_space, 'shape') and len(action_space.shape) > 0:
                # 有形状信息，假设是连续的
                self.actions = np.zeros((self.buffer_size, *action_space.shape), dtype=np.float32)
            else:
                # 标量动作，保持原有逻辑
                self.actions = np.zeros(self.buffer_size, dtype=np.int32)
        
        # 其他字段正常初始化
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminateds = np.zeros(self.buffer_size, dtype=np.int32)
        self.truncateds = np.zeros(self.buffer_size, dtype=np.int32)
        self.weights = np.zeros(self.buffer_size, dtype=np.float32)
        self.t = np.zeros(self.buffer_size, dtype=np.int32)

    def _init_action_buffer_from_space(self, action_space):
        """保持原有的动作空间处理逻辑"""
        if isinstance(action_space, spaces.Discrete):
            self.actions = np.zeros(self.buffer_size, dtype=np.int32)
        elif isinstance(action_space, spaces.Box):
            action_shape = action_space.shape
            self.actions = np.zeros((self.buffer_size, *action_shape), dtype=np.float32)
        self._action_shape_initialized = True

    def _initialize_action_buffer(self, sample_actions):
        """根据第一个样本动态初始化动作缓冲区"""
        if self._action_shape_initialized:
            return
            
        # 从实际样本推断动作形状和类型
        if isinstance(sample_actions, np.ndarray):
            if sample_actions.ndim == 1:
                # 单个样本
                action_shape = sample_actions.shape
                action_dtype = sample_actions.dtype
            else:
                # 批量样本
                action_shape = sample_actions.shape[1:]  # 去掉batch维度
                action_dtype = sample_actions.dtype
        else:
            # 处理标量或其他类型
            action_shape = ()
            action_dtype = np.float32 if isinstance(sample_actions, (float, np.floating)) else np.int32
        
        # 分配动作缓冲区
        if action_shape == ():
            self.actions = np.zeros(self.buffer_size, dtype=action_dtype)
        else:
            self.actions = np.zeros((self.buffer_size, *action_shape), dtype=action_dtype)
        
        self._action_shape_initialized = True
        logger.debug(f"动态初始化动作缓冲区: shape={action_shape}, dtype={action_dtype}")

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, batch: SampleBatch) -> None:
        """添加元素到缓冲区，动态处理动作空间"""
        # 获取样本动作并初始化动作缓冲区
        sample_actions = batch.get("actions")
        if not self._action_shape_initialized:
            self._initialize_action_buffer(sample_actions)
        
        shape = batch.get("obs").shape[0]
        
        # 边界检查
        if self.pos + shape > self.buffer_size:
            shape = self.buffer_size - self.pos
            
        # 存储数据
        self.obs[self.pos:self.pos + shape] = np.array(batch.get("obs")[:shape])
        self.new_obs[self.pos:self.pos + shape] = np.array(batch.get("new_obs")[:shape])
        
        # 动态处理动作数据
        actions_data = sample_actions[:shape]
        if self.actions.ndim == 1 and actions_data.ndim > 1:
            # 如果缓冲区是1D但数据是多维，展平处理
            self.actions[self.pos:self.pos + shape] = actions_data.flatten()[:shape]
        else:
            self.actions[self.pos:self.pos + shape] = actions_data
            
        self.rewards[self.pos:self.pos + shape] = batch.get("rewards")[:shape]
        self.terminateds[self.pos:self.pos + shape] = batch.get("terminateds")[:shape]
        self.truncateds[self.pos:self.pos + shape] = batch.get("truncateds")[:shape]
        
        # 权重处理保持不变
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
            # 未压缩，使用父类方法
            return super().sample()
            
        # 压缩数据，需要解压缩
        from .mpber_ram_saver_v8 import decompress_sample_batch
        return decompress_sample_batch(self._compressed_storage, self.compress_base)
        
    def is_compressed(self) -> bool:
        """检查是否存储了压缩数据"""
        return self._compressed
