import numpy as np
from typing import Tuple, Dict, Any, Optional
from gymnasium import spaces
from ray.rllib.policy.policy import SampleBatch
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
            randomly: bool = False,
            compression_level: int = 5,
            enable_memory_monitoring: bool = False
    ):
        """
        初始化压缩重放节点
        Args:
            buffer_size: 次级Block size
            obs_space: 观察空间（支持Box, Discrete等）
            action_space: 动作空间（支持Discrete, Box等）
            compress_base: 压缩基准维度（-1为智能默认）
            randomly: 是否随机采样
            compression_level: 压缩级别（1-9，默认5）
            enable_memory_monitoring: 是否启用内存监控
        """
        if buffer_size <= 0:
            raise ValueError(f"buffer_size必须大于0，当前值: {buffer_size}")
        
        self.buffer_size = buffer_size
        self.obs_space = obs_space
        self.action_space = action_space
        self.compress_base = compress_base
        self.randomly = randomly
        self.compression_level = max(1, min(9, compression_level))
        self.enable_memory_monitoring = enable_memory_monitoring

        # 状态管理
        self.pos = 0
        self.full = False

        self.obs_shape = get_obs_shape(obs_space)
        
        # 预分配观察空间（采用BaseBuffer的通用方案）
        self.obs = np.zeros(np.concatenate([[self.buffer_size], self.obs_shape]), dtype=obs_space.dtype)
        self.new_obs = np.zeros(np.concatenate([[self.buffer_size], self.obs_shape]), dtype=obs_space.dtype)

        # 增强动作空间支持（采用BaseBuffer的完整逻辑）
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
            raise ValueError(f"不支持的动作空间类型: {type(action_space)}")

        # 标准RL字段（与BaseBuffer保持一致的数据类型）
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminateds = np.zeros(self.buffer_size, dtype=np.int32)
        self.truncateds = np.zeros(self.buffer_size, dtype=np.int32)
        self.weights = np.zeros(self.buffer_size, dtype=np.float32)

        # 内存监控
        if self.enable_memory_monitoring:
            self._initial_memory = self._calculate_memory_usage()
            logger.info(f"节点初始化完成: buffer_size={buffer_size}, "
                       f"obs_shape={self.obs_shape}, 内存使用={self._initial_memory/1024/1024:.2f}MB")

    def _calculate_memory_usage(self) -> int:
        """计算当前内存使用量"""
        return (
            self.obs.nbytes +
            self.new_obs.nbytes +
            self.actions.nbytes +
            self.rewards.nbytes +
            self.terminateds.nbytes +
            self.truncateds.nbytes +
            self.weights.nbytes
        )

    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存使用信息"""
        current_memory = self._calculate_memory_usage()
        return {
            "current_memory_bytes": current_memory,
            "current_memory_mb": current_memory / 1024 / 1024,
            "buffer_utilization": self.pos / self.buffer_size,
            "obs_shape": self.obs_shape,
            "action_shape": self.actions.shape[1:] if len(self.actions.shape) > 1 else (),
            "compression_level": self.compression_level
        }

    def add(self, batch: SampleBatch) -> None:
        """
        添加样本批次到缓冲区
        
        Args:
            batch: 样本批次，包含obs, new_obs, actions, rewards等字段
            
        Raises:
            ValueError: 如果批次数据格式不正确或缺少必需字段
        """
        if not isinstance(batch, SampleBatch):
            raise ValueError("batch必须是SampleBatch类型")
        
        if len(batch) == 0:
            logger.warning("收到空批次，跳过")
            return

        # 验证必需字段
        required_fields = ["obs", "new_obs", "actions", "rewards", "terminateds", "truncateds"]
        missing_fields = [field for field in required_fields if field not in batch]
        if missing_fields:
            raise ValueError(f"批次缺少必需字段: {missing_fields}")

        batch_size = len(batch["obs"])

        # 计算实际可添加的样本数
        available_space = self.buffer_size - self.pos
        actual_size = min(batch_size, available_space)

        if actual_size == 0:
            logger.warning("缓冲区已满，无法添加更多样本")
            return

        # 存储数据到对应字段
        end_pos = self.pos + actual_size

        self.obs[self.pos:end_pos] = batch["obs"][:actual_size]
        self.new_obs[self.pos:end_pos] = batch["new_obs"][:actual_size]
        self.actions[self.pos:end_pos] = batch["actions"][:actual_size]
        self.rewards[self.pos:end_pos] = batch["rewards"][:actual_size]
        self.terminateds[self.pos:end_pos] = batch["terminateds"][:actual_size]
        self.truncateds[self.pos:end_pos] = batch["truncateds"][:actual_size]

        # 处理权重（可选字段）
        if "weights" in batch and batch["weights"] is not None:
            self.weights[self.pos:end_pos] = batch["weights"][:actual_size]
        else:
            self.weights[self.pos:end_pos] = 1.0  # 默认权重

        # 更新状态
        self.pos = end_pos
        if self.pos >= self.buffer_size:
            self.full = True

        if self.enable_memory_monitoring:
            logger.debug(f"添加{actual_size}个样本，当前使用率: {self.pos/self.buffer_size:.2%}")

    def _prepare_for_compression(self) -> SampleBatch:
        """
        准备数据进行压缩
        
        包含智能的数据预处理逻辑：
        1. 确定采样范围和策略
        2. 构建标准的SampleBatch
        3. 应用数据转置优化（提高压缩率）
        
        Returns:
            预处理后的SampleBatch，准备用于压缩
        """
        # 确定采样范围
        upper_bound = self.buffer_size if self.full else self.pos

        if self.randomly:
            # 随机采样（无重复）
            indices = np.random.choice(upper_bound, size=self.size(), replace=False)
        else:
            # 顺序采样
            indices = np.arange(self.size())

        # 构建标准的SampleBatch
        batch_data = {
            "obs": self.obs[indices],
            "new_obs": self.new_obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "terminateds": self.terminateds[indices],
            "truncateds": self.truncateds[indices],
            "weights": self.weights[indices]
        }

        batch = SampleBatch(batch_data)

        # 数据转置优化（提高压缩效率）
        obs_array = batch["obs"]
        new_obs_array = batch["new_obs"]
        rank = len(obs_array.shape)

        if self.compress_base == -1 and rank > 1:
            # 智能默认：将batch维度移到最后
            # 这通常能提高压缩率，特别是对图像数据
            axes = list(range(1, rank)) + [0]
            batch["obs"] = np.transpose(obs_array, axes)
            batch["new_obs"] = np.transpose(new_obs_array, axes)
            batch["_transpose_info"] = f"batch_to_end_{rank}d"

        elif (self.compress_base >= 0 and
              self.compress_base < rank and
              rank > 1):
            # 用户指定的维度转置
            axes = list(range(rank))
            to_move = axes.pop(self.compress_base)
            axes.append(to_move)

            batch["obs"] = np.transpose(obs_array, axes)
            batch["new_obs"] = np.transpose(new_obs_array, axes)
            batch["_transpose_info"] = f"dim_{self.compress_base}_to_end_{rank}d"

        return batch

    def sample_compressed(self) -> Tuple[Dict[str, Any], float]:
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
            compressed_data = self._compress_sample_batch(prepared_batch)
            
            if self.enable_memory_monitoring:
                original_size = sum(arr.nbytes for arr in [prepared_batch["obs"], prepared_batch["new_obs"]])
                compressed_size = len(compressed_data["obs"][0]) + len(compressed_data["new_obs"][0])
                compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
                logger.debug(f"压缩完成: 原始大小={original_size/1024:.1f}KB, "
                           f"压缩后={compressed_size/1024:.1f}KB, 压缩比={compression_ratio:.2%}")
            
            return compressed_data, block_weight
        except Exception as e:
            raise RuntimeError(f"压缩失败: {e}")

    def _compress_sample_batch(self, sample_batch: SampleBatch) -> Dict[str, Any]:
        """压缩样本批次"""
        obs_array = sample_batch["obs"]
        new_obs_array = sample_batch["new_obs"]

        # 使用blosc压缩，支持压缩级别配置
        compressed_obs = blosc.pack_array(obs_array, cname='blosclz', clevel=self.compression_level)
        compressed_new_obs = blosc.pack_array(new_obs_array, cname='blosclz', clevel=self.compression_level)

        # 返回压缩数据
        return {
            "obs": [compressed_obs],  # 保持与v8兼容的格式
            "new_obs": [compressed_new_obs],
            "actions": sample_batch["actions"],
            "rewards": sample_batch["rewards"],
            "terminateds": sample_batch["terminateds"],
            "truncateds": sample_batch["truncateds"],
            "weights": sample_batch["weights"]
        }

    def size(self) -> int:
        """获取当前存储的样本数量"""
        return self.buffer_size if self.full else self.pos

    def reset(self) -> None:
        """重置节点状态（保留已分配的内存）"""
        self.pos = 0
        self.full = False
        # 注意：不清空数据，只重置指针，提高效率
        if self.enable_memory_monitoring:
            logger.debug("节点已重置")

    def is_ready(self) -> bool:
        """检查节点是否准备好进行压缩"""
        return self.full

    def __repr__(self) -> str:
        return (f"CompressReplayNode(size={self.pos}/{self.buffer_size}, "
                f"obs_shape={self.obs_shape}, compression_level={self.compression_level})")
