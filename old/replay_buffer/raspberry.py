import logging
import numpy as np
import time
import blosc
import sys
from gymnasium.spaces import Space
from typing import Dict, Optional, Any, List
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from replay_buffer.compress_replay_node import CompressReplayNode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import concat_samples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decompress_sample_batch(ma_batch: SampleBatch, compress_base: int = -1) -> SampleBatch:
    """使用 blosc.unpack_array 解压缩，并根据 compress_base 参数恢复原始格式。
    
    支持任意维度的观察数据，不再限制于图像数据。
    
    Args:
        ma_batch: 压缩的批次数据
        compress_base: 压缩时移动到最后的维度索引
                      -1: 智能默认行为（将第0维移到最后，适用于大多数时序数据）
                      >=0: 将指定维度从最后位置恢复到原始位置
    """
    t0 = time.time()

    # 优先读取批次内的 compress_base 元数据
    compress_base_used = ma_batch.get("compress_base", compress_base)

    # 使用 unpack_array 恢复压缩的数组
    decompressed_obs_transposed = blosc.unpack_array(ma_batch["obs"][0])
    decompressed_new_obs_transposed = blosc.unpack_array(ma_batch["new_obs"][0])

    # 获取数据维度信息
    rank = len(decompressed_obs_transposed.shape)

    if compress_base_used == -1:
        # 智能默认行为：将第0维（通常是batch维度）从最后位置移回开头
        if rank <= 1:
            decompressed_obs = decompressed_obs_transposed
            decompressed_new_obs = decompressed_new_obs_transposed
            transpose_type = "no_transpose"
        else:
            # 多维数据：将最后一个维度（原第0维）移回开头
            axes = [rank - 1] + list(range(rank - 1))
            decompressed_obs = np.transpose(decompressed_obs_transposed, axes)
            decompressed_new_obs = np.transpose(decompressed_new_obs_transposed, axes)
            transpose_type = f"default_dim0_to_front_{rank}D"
    else:
        # 用户指定的维度恢复
        if compress_base_used >= rank or rank <= 1:
            decompressed_obs = decompressed_obs_transposed
            decompressed_new_obs = decompressed_new_obs_transposed
            transpose_type = "no_transpose"
        else:
            # 将最后一个维度插入到 compress_base 位置
            axes = list(range(rank))
            axes.pop()  # 移除最后一个维度
            axes.insert(compress_base_used, rank - 1)  # 插入到指定位置

            decompressed_obs = np.transpose(decompressed_obs_transposed, axes)
            decompressed_new_obs = np.transpose(decompressed_new_obs_transposed, axes)
            transpose_type = f"dim_{compress_base_used}_from_end_{rank}D"

    t1 = time.time()
    logger.debug(f"[Decompression] Blosc unpack_array & {transpose_type} transpose took: {t1 - t0:.4f}s")

    data_dict = {
        "obs": decompressed_obs,
        "new_obs": decompressed_new_obs,
        "actions": ma_batch["actions"],
        "rewards": ma_batch["rewards"],
        "terminateds": ma_batch["terminateds"],
        "truncateds": ma_batch["truncateds"],
        "weights": ma_batch["weights"],
    }

    if "batch_indexes" in ma_batch:
        data_dict["batch_indexes"] = ma_batch["batch_indexes"]

    return SampleBatch(data_dict)


class PrioritizedBlockReplayBuffer(PrioritizedReplayBuffer):
    """带压缩的优先级重放缓冲区"""

    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            sub_buffer_size: int = 32,
            compress_base: int = -1,
            **kwargs
    ):
        # 严格遵循 Ray：将优先级参数（prioritized_replay_alpha/prioritized_replay_beta）
        # 直接通过 **kwargs 透传给父类，不做别名或默认处理。
        super(PrioritizedBlockReplayBuffer, self).__init__(**kwargs)

        self.sub_buffer_size = sub_buffer_size
        self.compress_base = compress_base
        self.compress_node = CompressReplayNode(
            buffer_size=sub_buffer_size,
            obs_space=obs_space,
            action_space=action_space,
            compress_base=compress_base,
        )

    def stats(self, debug: bool = False) -> Dict[str, Any]:
        """计算压缩块的实际内存占用"""
        total_size = 0
        for sample_batch in self._storage:
            if "obs" in sample_batch and hasattr(sample_batch["obs"], '__getitem__'):
                total_size += sys.getsizeof(sample_batch["obs"][0])
            if "new_obs" in sample_batch and hasattr(sample_batch["new_obs"], '__getitem__'):
                total_size += sys.getsizeof(sample_batch["new_obs"][0])

        return {
            "est_size_bytes": total_size,
            "num_entries": len(self._storage) * self.sub_buffer_size,
        }

    def sample(self, num_items: int, beta: float, **kwargs) -> Optional[SampleBatch]:
        """重写采样方法以正确处理权重扩展（块权重 -> transition 权重）"""
        batch = super(PrioritizedBlockReplayBuffer, self).sample(num_items, beta=beta, **kwargs)

        if batch is not None:
            # 展开块级权重到 transition 级
            if "weights" in batch and 0 < batch.count != len(batch["weights"]):
                num_blocks = len(batch["weights"])
                samples_per_block = batch.count // num_blocks
                expanded_weights = np.repeat(batch["weights"], samples_per_block)
                batch["weights"] = expanded_weights
            # 展开块级 batch_indexes 到 transition 级，便于 RLlib 的通用优先级更新工具使用
            if "batch_indexes" in batch and 0 < batch.count != len(batch["batch_indexes"]):
                num_blocks_idx = len(batch["batch_indexes"])
                samples_per_block_idx = batch.count // num_blocks_idx
                expanded_idx = np.repeat(batch["batch_indexes"], samples_per_block_idx)
                batch["batch_indexes"] = expanded_idx

        return batch

    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """添加批次到缓冲区"""
        if not isinstance(batch, SampleBatch):
            return

        # 采用 CompressReplayNode 即时压缩入库：切片填充块，ready 即写入
        idx = 0
        count = len(batch)
        while idx < count:
            space = self.compress_node.buffer_size - self.compress_node.pos
            take = min(space, count - idx)
            slice_batch = batch.slice(idx, idx + take)
            self.compress_node.add(slice_batch)
            idx += take

            if self.compress_node.is_ready():
                compressed_data, weight = self.compress_node.sample()
                try:
                    comp_obs_len = len(compressed_data["obs"][0])
                    comp_new_obs_len = len(compressed_data["new_obs"][0])
                    logger.debug(
                        f"[RB.add] block_ready: size={self.sub_buffer_size}, block_weight={weight:.4f}, "
                        f"comp_obs_bytes={comp_obs_len}, comp_new_obs_bytes={comp_new_obs_len}"
                    )
                except Exception:
                    pass
                self._add_single_batch(compressed_data, weight=weight)
                self.compress_node.reset()

    def _encode_sample(self, idxes: List[int]) -> SampleBatch:
        """编码样本（同步解压 + 一次性拼接）。本方法由父类在采样时调用，必须保留。"""
        t0 = time.time()
        compressed_list = []
        for i in idxes:
            self._hit_count[i] += 1
            compressed_list.append(self._storage[i])

        decompressed_list: List[SampleBatch] = [
            decompress_sample_batch(compressed_sample, self.compress_base)
            for compressed_sample in compressed_list
        ]
        out = concat_samples(decompressed_list)
        dt = (time.time() - t0) * 1000.0
        try:
            logger.debug(
                f"[RB._encode_sample] blocks={len(idxes)}, transitions={out.count}, time_ms={dt:.2f}"
            )
        except Exception:
            pass
        return out
