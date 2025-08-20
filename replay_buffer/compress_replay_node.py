import numpy as np
from typing import Tuple, Dict, Any
from gymnasium import spaces
from ray.rllib.policy.sample_batch import SampleBatch
from utils import get_obs_shape
import blosc
import logging

logger = logging.getLogger(__name__)


class CompressReplayNode(object):
    """
    压缩重放节点
    设计原则：
    1. 初始化时预分配所有必要空间
    2. 明确的数据类型和维度定义
    3. 内置压缩功能，一步到位压缩后的sample_batch
    4. 支持不同类型的观察和动作空间
    5. 完善的内存监控和错误处理
    """

    def __init__(
            self,
            buffer_size: int,
            obs_space: spaces.Space,
            action_space: spaces.Space,
            compress_base: int = -1,
            compression_level: int = 5,
            cname: str = 'zstd',
            shuffle: int = blosc.BITSHUFFLE,
            nthreads: int = 4,
    ):
        """
        初始化压缩重放节点
        Args:
            buffer_size: 次级Block size
            obs_space: 观察空间（支持Box, Discrete等）
            action_space: 动作空间（支持Discrete, Box等）
            compress_base: 压缩基准维度（-1为智能默认）
            compression_level: 压缩级别（1-9，默认5）
        """
        if buffer_size <= 0:
            raise ValueError(f"buffer_size必须大于0，当前值: {buffer_size}")

        self.buffer_size = buffer_size
        self.obs_space = obs_space
        self.action_space = action_space
        self.compress_base = compress_base
        self.compression_level = max(1, min(9, compression_level))
        self.cname = cname
        self.shuffle = shuffle
        self.nthreads = max(1, int(nthreads))
        try:
            blosc.set_nthreads(self.nthreads)
        except Exception:
            # 某些环境下不支持设置线程数，忽略即可
            pass

        # 状态管理
        self.pos = 0
        self.full = False

        self.obs_shape = get_obs_shape(obs_space)

        # 预分配观察空间
        obs_buffer_shape = (self.buffer_size,) + tuple(self.obs_shape)
        self.obs = np.zeros(obs_buffer_shape, dtype=obs_space.dtype)
        self.new_obs = np.zeros(obs_buffer_shape, dtype=obs_space.dtype)

        # 增强动作空间支持（采用BaseBuffer的完整逻辑）
        if isinstance(action_space, spaces.Discrete):
            self.actions = np.zeros(self.buffer_size, dtype=action_space.dtype)
        else:
            self.actions = np.zeros((self.buffer_size, *action_space.shape), dtype=action_space.dtype)

        # 标准RL字段（与BaseBuffer保持一致的数据类型）
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminateds = np.zeros(self.buffer_size, dtype=np.int32)
        self.truncateds = np.zeros(self.buffer_size, dtype=np.int32)
        self.weights = np.zeros(self.buffer_size, dtype=np.float32)

    def add(self, batch: SampleBatch) -> None:
        # todo: add

        """添加样本批次到缓冲区"""
        if not isinstance(batch, SampleBatch) or len(batch) == 0:
            raise ValueError

        batch_size = len(batch["obs"])
        available_space = self.buffer_size - self.pos
        actual_size = min(batch_size, available_space)

        if actual_size == 0:
            raise ValueError

        # 存储数据
        end_pos = self.pos + actual_size
        slice_range = slice(self.pos, end_pos)
        data_slice = slice(None, actual_size)

        self.obs[slice_range] = batch["obs"][data_slice]
        self.new_obs[slice_range] = batch["new_obs"][data_slice]
        self.actions[slice_range] = batch["actions"][data_slice]
        self.rewards[slice_range] = batch["rewards"][data_slice]
        self.terminateds[slice_range] = batch["terminateds"][data_slice]
        self.truncateds[slice_range] = batch["truncateds"][data_slice]

        # 处理权重
        weights = batch.get("weights")
        self.weights[slice_range] = weights[data_slice] if weights is not None else 1.0

        # 更新状态
        self.pos = end_pos
        if self.pos >= self.buffer_size:
            self.full = True

    def _prepare_for_compression(self) -> SampleBatch:
        """准备数据进行压缩"""
        size = self.size()
        obs_data = self.obs[:size]
        new_obs_data = self.new_obs[:size]
        
        # 如果需要转置优化，先转置
        rank = len(obs_data.shape)
        if rank > 1:
            if self.compress_base == -1:
                # 智能默认：将batch维度移到最后
                axes = list(range(1, rank)) + [0]
                obs_data = np.transpose(obs_data, axes)
                new_obs_data = np.transpose(new_obs_data, axes)
            elif 0 <= self.compress_base < rank:
                # 用户指定维度转置
                axes = list(range(rank))
                to_move = axes.pop(self.compress_base)
                axes.append(to_move)
                obs_data = np.transpose(obs_data, axes)
                new_obs_data = np.transpose(new_obs_data, axes)
        
        # 创建优化后的sample batch
        return SampleBatch({
            "obs": obs_data,
            "new_obs": new_obs_data,
            "actions": self.actions[:size],
            "rewards": self.rewards[:size],
            "terminateds": self.terminateds[:size],
            "truncateds": self.truncateds[:size],
            "weights": self.weights[:size],
            "compress_base": self.compress_base,
        })

    def sample(self) -> Tuple[SampleBatch, float]:
        """
        采样并返回压缩数据
        
        这是节点的核心方法：
        1. 准备数据（包含预处理）
        2. 计算块级权重
        3. 执行压缩
        4. 返回压缩数据和权重
        
        Returns:
            Tuple[压缩数据字典, 块权重]
            
        Raises:
            ValueError: 如果节点为空
        """
        if not self.full and self.pos == 0:
            raise ValueError("节点为空，无法采样压缩数据")

        # 准备数据
        prepared_batch = self._prepare_for_compression()

        # 计算块级权重（样本权重的平均值）
        block_weight = float(np.mean(prepared_batch["weights"]))
        if np.isnan(block_weight) or block_weight <= 0:
            block_weight = 0.01  # 最小权重保护

        # 执行压缩
        try:
            compressed_batch = self._compress_sample_batch(prepared_batch)
            return compressed_batch, block_weight
        except Exception as e:
            logger.exception("压缩失败")
            raise RuntimeError(f"压缩失败: {e}")

    def _compress_sample_batch(self, sample_batch: SampleBatch) -> SampleBatch:
        """压缩样本批次并返回压缩后的 SampleBatch"""
        obs_array = sample_batch["obs"]
        new_obs_array = sample_batch["new_obs"]

        # 使用blosc压缩，支持压缩级别/算法/洗牌策略配置
        compressed_obs = blosc.pack_array(
            obs_array,
            cname=self.cname,
            clevel=self.compression_level,
            shuffle=self.shuffle,
        )
        compressed_new_obs = blosc.pack_array(
            new_obs_array,
            cname=self.cname,
            clevel=self.compression_level,
            shuffle=self.shuffle,
        )

        # 返回压缩后的 SampleBatch（仅 obs/new_obs 为压缩对象数组）
        return SampleBatch({
            "obs": np.array([compressed_obs], dtype=object),
            "new_obs": np.array([compressed_new_obs], dtype=object),
            "actions": sample_batch["actions"],
            "rewards": sample_batch["rewards"],
            "terminateds": sample_batch["terminateds"],
            "truncateds": sample_batch["truncateds"],
            "weights": sample_batch["weights"],
            "compress_base": self.compress_base,
        })

    def size(self) -> int:
        """获取当前存储的样本数量"""
        return self.buffer_size if self.full else self.pos

    def reset(self) -> None:
        """重置节点状态（保留已分配的内存）"""
        self.pos = 0
        self.full = False
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("CompressReplayNode 已重置")

    def is_ready(self) -> bool:
        """检查节点是否准备好进行压缩"""
        return self.full

    def __repr__(self) -> str:
        return (f"CompressReplayNode(size={self.pos}/{self.buffer_size}, "
                f"obs_shape={self.obs_shape}, compression_level={self.compression_level})")
