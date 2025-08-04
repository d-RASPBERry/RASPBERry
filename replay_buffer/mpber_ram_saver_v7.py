import ray
import logging
import numpy as np
import time
import blosc
import sys
from itertools import chain
from gymnasium.spaces import Space
from typing import Dict, Optional, Any, List
from ray.rllib.utils.typing import PolicyID
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.replay_buffers import StorageUnit
from ray.rllib.utils.replay_buffers.utils import SampleBatchType
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import ReplayMode, merge_dicts_with_warning
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import MultiAgentPrioritizedReplayBuffer
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from replay_buffer.replay_node import BaseBuffer
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.util.debug import log_once
from utils import split_list_into_n_parts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decompress_sample_batch(ma_batch, compress_base=-1):
    """使用 blosc.unpack_array 解压缩，并根据 compress_base 参数恢复原始格式。
    
    支持任意维度的观察数据，不再限制于图像数据。
    
    Args:
        ma_batch: 压缩的批次数据
        compress_base: 压缩时移动到最后的维度索引
                      -1: 智能默认行为（将第0维移到最后，适用于大多数时序数据）
                      >=0: 将指定维度从最后位置恢复到原始位置
    """
    t0 = time.time()
    
    # 使用 unpack_array 恢复压缩的数组
    decompressed_obs_transposed = blosc.unpack_array(ma_batch["obs"][0])
    decompressed_new_obs_transposed = blosc.unpack_array(ma_batch["new_obs"][0])

    # 获取数据维度信息
    rank = len(decompressed_obs_transposed.shape)
    
    if compress_base == -1:
        # 智能默认行为：将第0维（通常是batch维度）从最后位置移回开头
        # 这适用于大多数时序数据，因为batch内的样本在时间上相近，压缩效果好
        if rank <= 1:
            # 1D或0D数据，无需转置
            decompressed_obs = decompressed_obs_transposed
            decompressed_new_obs = decompressed_new_obs_transposed
            transpose_type = "no_transpose"
        else:
            # 多维数据：将最后一个维度（原第0维）移回开头
            # 转置轴：[rank-1, 0, 1, 2, ..., rank-2]
            axes = [rank - 1] + list(range(rank - 1))
            decompressed_obs = np.transpose(decompressed_obs_transposed, axes)
            decompressed_new_obs = np.transpose(decompressed_new_obs_transposed, axes)
            transpose_type = f"default_dim0_to_front_{rank}D"
    else:
        # 用户指定的维度恢复：将最后一个维度移回到 compress_base 位置
        if compress_base >= rank:
            logger.warning(f"compress_base={compress_base} >= rank={rank}, using no transpose")
            decompressed_obs = decompressed_obs_transposed
            decompressed_new_obs = decompressed_new_obs_transposed
            transpose_type = "invalid_compress_base"
        elif rank <= 1:
            # 1D或0D数据，无需转置
            decompressed_obs = decompressed_obs_transposed
            decompressed_new_obs = decompressed_new_obs_transposed
            transpose_type = "no_transpose"
        else:
            # 将最后一个维度插入到 compress_base 位置
            axes = list(range(rank))
            axes.pop()  # 移除最后一个维度
            axes.insert(compress_base, rank - 1)  # 插入到指定位置
            
            decompressed_obs = np.transpose(decompressed_obs_transposed, axes)
            decompressed_new_obs = np.transpose(decompressed_new_obs_transposed, axes)
            transpose_type = f"dim_{compress_base}_from_end_{rank}D"

    t1 = time.time()
    logger.debug(f"[Decompression] Blosc unpack_array & {transpose_type} transpose took: {t1 - t0:.4f}s")
    logger.debug(f"[Decompression] Shape: {decompressed_obs_transposed.shape} -> {decompressed_obs.shape}")
    
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


@ray.remote(num_cpus=1, max_calls=50, num_returns=1)
def compress_sample_batch_loop(samples):
    _ = []
    for each in samples:
        _.append(compress_sample_batch(each[0], each[1], each[2]))  # 添加 compress_base 参数
    return _


def compress_sample_batch(sample_batch, weight, compress_base=-1):
    """
    Args:
        sample_batch: 要压缩的样本批次
        weight: 样本权重
        compress_base: 要移动到最后的维度索引
                      -1: 智能默认行为（将第0维移到最后，适用于大多数时序数据）
                      >=0: 将指定维度移动到最后
    """
    obs_array = sample_batch["obs"]
    new_obs_array = sample_batch["new_obs"]
    
    # 获取数据维度信息
    rank = len(obs_array.shape)
    
    if compress_base == -1:
        # 智能默认行为：将第0维（通常是batch维度）移到最后
        # 这样相近时间点的样本在内存中相邻，压缩效果更好
        if rank <= 1:
            # 1D或0D数据，无需转置
            obs_transposed = obs_array
            new_obs_transposed = new_obs_array
            transpose_type = "no_transpose"
        else:
            # 多维数据：将第0维移到最后
            # 转置轴：[1, 2, 3, ..., rank-1, 0]
            axes = list(range(1, rank)) + [0]
            obs_transposed = np.transpose(obs_array, axes)
            new_obs_transposed = np.transpose(new_obs_array, axes)
            transpose_type = f"default_dim0_to_end_{rank}D"
    else:
        # 用户指定的维度转置：将 compress_base 维度移动到最后
        if compress_base >= rank:
            logger.warning(f"compress_base={compress_base} >= rank={rank}, using no transpose")
            obs_transposed = obs_array
            new_obs_transposed = new_obs_array
            transpose_type = "invalid_compress_base"
        elif rank <= 1:
            # 1D或0D数据，无需转置
            obs_transposed = obs_array
            new_obs_transposed = new_obs_array
            transpose_type = "no_transpose"
        else:
            # 将 compress_base 维度移动到最后
            axes = list(range(rank))
            axes.pop(compress_base)  # 移除指定维度
            axes.append(compress_base)  # 添加到最后
            
            obs_transposed = np.transpose(obs_array, axes)
            new_obs_transposed = np.transpose(new_obs_array, axes)
            transpose_type = f"dim_{compress_base}_to_end_{rank}D"

    logger.debug(f"[Compression] {transpose_type}: {obs_array.shape} -> {obs_transposed.shape}")

    # 使用 pack_array 打包并压缩转置后的数组
    compressed_obs = blosc.pack_array(obs_transposed, clevel=5, cname='lz4', shuffle=blosc.SHUFFLE)
    compressed_new_obs = blosc.pack_array(new_obs_transposed, clevel=5, cname='lz4', shuffle=blosc.SHUFFLE)

    # 存储压缩后的数据块
    data = SampleBatch({
        "obs": np.array([compressed_obs], dtype=object),
        "new_obs": np.array([compressed_new_obs], dtype=object),
        "actions": sample_batch["actions"],
        "rewards": sample_batch["rewards"],
        "terminateds": sample_batch["terminateds"],
        "truncateds": sample_batch["truncateds"],
        "weights": sample_batch["weights"],
    })
    
    return data, weight


class PrioritizedBlockReplayBuffer(PrioritizedReplayBuffer):
    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            randomly: bool = False,
            sub_buffer_size: int = 32,
            beta=0.6,
            num_save=200,
            split_mini_batch=10,
            compress_base: int = -1,  # 新增参数
            **kwargs
    ):
        super(PrioritizedBlockReplayBuffer, self).__init__(**kwargs)
        self.beta = beta
        self.sub_buffer_size =sub_buffer_size
        self.num_save = num_save
        self.split_mini_batch = split_mini_batch
        self.compress_base = compress_base  # 存储压缩基准维度
        self.base_buffer = BaseBuffer(sub_buffer_size, obs_space, action_space, randomly)
        self._sub_store = []

    def stats(self, debug: bool = False) -> Dict[str, Any]:
        """
        覆盖父类的 stats 方法，以正确计算 Blosc 压缩块的内存大小。
        """
        total_size = 0
        for sample_batch in self._storage:
            # 累加每个 SampleBatch 中 'obs' 和 'new_obs' 压缩块的实际内存占用
            if "obs" in sample_batch and hasattr(sample_batch["obs"], '__getitem__'):
                total_size += sys.getsizeof(sample_batch["obs"][0])
            if "new_obs" in sample_batch and hasattr(sample_batch["new_obs"], '__getitem__'):
                total_size += sys.getsizeof(sample_batch["new_obs"][0])
        
        # 返回与父类结构类似的统计字典
        return {
            "est_size_bytes": total_size,
            "num_entries": len(self._storage) * self.sub_buffer_size,
        }

    def sample(self, num_items: int, **kwargs):
        """
        覆盖父类的 sample 方法，以处理块状数据的权重扩展问题。
        
        1. 调用父类 sample 方法，获得一个包含未扩展权重的批次。
           - 父类方法会调用我们重写的 _encode_sample，它返回一个由多个块拼接成的大批次。
           - 父类方法会附加一个 `weights` 键，其长度等于采样出的块数量 (num_items)。
        2. 检查权重数量是否与样本总数不匹配。
        3. 如果不匹配，将每个块的权重重复 `sub_buffer_size` 次，使其与样本对齐。
        """
        batch = super(PrioritizedBlockReplayBuffer, self).sample(num_items, **kwargs)

        # 当采样出的样本总数与权重数量不匹配时，进行扩展
        if "weights" in batch and batch.count > 0 and len(batch["weights"]) != batch.count:
            num_blocks = len(batch["weights"])
            samples_per_block = batch.count // num_blocks
            expanded_weights = np.repeat(batch["weights"], samples_per_block)
            batch["weights"] = expanded_weights
        
        return batch

    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Adds a batch to the buffer."""
        buffer = self.base_buffer
        self.base_buffer.add(batch)
        if buffer.full:
            data = buffer.sample()
            weight = np.mean(data.get("weights"))
            if np.isnan(weight):
                weight = 0.01
            buffer.reset()
            self._sub_store.append([data, weight, self.compress_base])  # 传递 compress_base
        if len(self._sub_store) == self.num_save:
            _list = split_list_into_n_parts(self._sub_store, n=self.split_mini_batch)
            result_ids = [compress_sample_batch_loop.remote(batch) for batch in _list]
            results = ray.get(result_ids)
            results = list(chain(*results))
            for each in results:
                self._add_single_batch(each[0], weight=each[1])

            self._sub_store = []

    def _encode_sample(self, idxes):
        """编码样本，解压压缩块并正确拼接。"""
        samples = []
        for i in idxes:
            self._hit_count[i] += 1
            samples.append(self._storage[i])

        if samples:
            # 修复：不能直接在压缩数据上使用 concat_samples
            # 需要先解压每个压缩块，再拼接解压后的数据
            if len(samples) == 1:
                # 单个样本，直接返回（仍然是压缩的，将在上层解压）
                out = samples[0]
            else:
                # 多个样本，需要解压后拼接
                decompressed_samples = []
                for compressed_sample in samples:
                    # 解压每个压缩块
                    decompressed = decompress_sample_batch(compressed_sample, self.compress_base)
                    decompressed_samples.append(decompressed)
                
                # 拼接解压后的数据
                from ray.rllib.policy.sample_batch import concat_samples
                out = concat_samples(decompressed_samples)
        else:
            out = SampleBatch()
        
        return out


@DeveloperAPI
class MultiAgentPrioritizedBlockReplayBuffer(MultiAgentPrioritizedReplayBuffer):
    """A prioritized replay buffer with Blosc compression for multi-agent setups.

    This buffer supports arbitrary-dimensional observation data (not limited to images).
    It uses block compression with configurable transpose strategies to exploit temporal
    locality in sample batches for high compression ratios.

    Key Features:
    - Universal data type support: Works with any dimensional observation data
    - Configurable compression strategy via compress_base parameter
    - High compression ratios by exploiting temporal similarity in sample batches
    - Parallel compression using Ray for performance

    The core insight is that samples within a batch are temporally close, so they
    exhibit high similarity. By rearranging dimensions strategically before compression,
    we can significantly improve compression ratios.
    """

    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            sub_buffer_size: int = 1,
            rollout_fragment_length: int = 4,
            capacity: int = 10000,
            storage_unit: str = "timesteps",
            num_shards: int = 1,
            replay_mode: str = "independent",
            replay_sequence_override: bool = True,
            replay_sequence_length: int = 1,
            replay_burn_in: int = 0,
            replay_zero_init_states: bool = True,
            prioritized_replay_alpha: float = 0.6,
            prioritized_replay_beta: float = 0.4,
            prioritized_replay_eps: float = 1e-6,
            split_mini_batch=10,
            num_save=200,
            store=2000,
            compress_base: int = -1,  # 新增参数
            **kwargs
    ):
        """Initializes a MultiAgentPrioritizedBlockReplayBuffer instance.

        Args:
            compress_base: Dimension index to move to the end for compression.
                          -1: Smart default (move batch dimension to end, works for most data)
                          >=0: Move specified dimension to end for compression
                          
                          Examples:
                          - For images (N,C,H,W): compress_base=0 moves N to end
                          - For time series (N,T,F): compress_base=1 moves T to end  
                          - For vectors (N,D): compress_base=0 moves N to end
                          
                          The key is to choose the dimension that varies least within
                          a temporally-close batch to maximize compression.
                          
            obs_space: Observation space (supports any Box space dimensionality)
            action_space: Action space
            sub_buffer_size: Size of sub-buffers for batching before compression
            capacity: Total buffer capacity
            其他参数: Standard RLlib replay buffer parameters
        """

        if "replay_mode" in kwargs and (
                kwargs["replay_mode"] == "lockstep"
                or kwargs["replay_mode"] == ReplayMode.LOCKSTEP
        ):
            if log_once("lockstep_mode_not_supported"):
                logger.error(
                    "Replay mode `lockstep` is not supported for "
                    "MultiAgentPrioritizedReplayBuffer. "
                    "This buffer will run in `independent` mode."
                )
            kwargs["replay_mode"] = "independent"
        
        self.compress_base = compress_base  # 存储压缩基准维度
        
        pber_config = {
            "type": PrioritizedBlockReplayBuffer,
            "action_space": action_space,
            "obs_space": obs_space,
            "storage_unit": StorageUnit.FRAGMENTS,
            "randomly": False,
            "sub_buffer_size": sub_buffer_size,
            "alpha": prioritized_replay_alpha,
            "beta": prioritized_replay_beta,
            "split_mini_batch": split_mini_batch,
            "num_save": num_save,
            "store": store,
            "compress_base": compress_base,  # 传递给底层 buffer
        }
        MultiAgentPrioritizedReplayBuffer.__init__(
            self,
            capacity=capacity,
            storage_unit=storage_unit,
            num_shards=num_shards,
            replay_mode=replay_mode,
            replay_sequence_override=replay_sequence_override,
            replay_sequence_length=replay_sequence_length,
            replay_burn_in=replay_burn_in,
            replay_zero_init_states=replay_zero_init_states,
            underlying_buffer_config=pber_config,
            prioritized_replay_alpha=prioritized_replay_alpha,
            prioritized_replay_beta=prioritized_replay_beta,
            prioritized_replay_eps=prioritized_replay_eps,
            **kwargs,
        )
        self.rollout_fragment_length = rollout_fragment_length

    @DeveloperAPI
    @override(ReplayBuffer)
    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Adds a batch to the appropriate policy's replay buffer.

        Turns the batch into a MultiAgentBatch of the DEFAULT_POLICY_ID if
        it is not a MultiAgentBatch. Subsequently, adds the individual policy
        batches to the storage.

        Args:
            batch : The batch to be added.
            ``**kwargs``: Forward compatibility kwargs.
        """
        if batch is None:
            if log_once("empty_batch_added_to_buffer"):
                logger.info(
                    "A batch that is `None` was added to {}. This can be "
                    "normal at the beginning of execution but might "
                    "indicate an issue.".format(type(self).__name__)
                )
            return
        # Make a copy so the replay buffer doesn't pin plasma memory.
        batch = batch.copy()
        # Handle everything as if multi-agent.
        batch = batch.as_multi_agent()
        with self.add_batch_timer:
            pids_and_batches = self._maybe_split_into_policy_batches(batch)
            for policy_id, sample_batch in pids_and_batches.items():
                if len(sample_batch) == 1:
                    self._add_to_underlying_buffer(policy_id, sample_batch)
                else:
                    _ = sample_batch.timeslices(size=self.rollout_fragment_length)
                    for s_batch in _:
                        self._add_to_underlying_buffer(policy_id, s_batch)

        self._num_added += batch.count

    @DeveloperAPI
    @override(PrioritizedReplayBuffer)
    def update_priorities(self, prio_dict: Dict) -> None:
        """Updates the priorities of underlying replay buffers.

        Computes new priorities from td_errors and prioritized_replay_eps.
        These priorities are used to update underlying replay buffers per
        policy_id.

        Args:
            prio_dict: A dictionary containing td_errors for
            batches saved in underlying replay buffers.
        """
        with self.update_priorities_timer:
            for policy_id, (batch_indexes, td_errors) in prio_dict.items():
                new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                
                # 如果优先级数量与块索引数量不匹配，则进行聚合
                if len(batch_indexes) != len(new_priorities):
                    num_blocks = len(batch_indexes)
                    samples_per_block = len(new_priorities) // num_blocks
                    # 将每个块内样本的优先级取平均值，作为该块的新优先级
                    block_priorities = np.mean(
                        new_priorities.reshape((num_blocks, samples_per_block)),
                        axis=1
                    )
                    self.replay_buffers[policy_id].update_priorities(
                        batch_indexes, block_priorities
                    )
                else:
                    # 如果数量匹配，直接更新
                    self.replay_buffers[policy_id].update_priorities(
                        batch_indexes, new_priorities
                    )

    def _maybe_split_into_policy_batches(self, batch: SampleBatchType):
        """Returns a dict of policy IDs and batches, depending on our replay mode.

        This method helps with splitting up MultiAgentBatches only if the
        self.replay_mode requires it.
        """
        return batch.policy_batches

    @DeveloperAPI
    @override(ReplayBuffer)
    def sample(
            self, num_items: int, policy_id: Optional[PolicyID] = None, **kwargs
    ) -> Optional[SampleBatchType]:
        t_start = time.time()
        logger.debug(f"Starting sample for policy '{policy_id}' with {num_items} items.")
        
        kwargs = merge_dicts_with_warning(self.underlying_buffer_call_args, kwargs)

        with self.replay_timer:
            # Lockstep mode: Sample from all policies at the same time an
            # equal amount of steps.
            if self.replay_mode == ReplayMode.LOCKSTEP:
                assert (
                        policy_id is None
                ), "`policy_id` specifier not allowed in `lockstep` mode!"
                
                t0 = time.time()
                raw_sample = self.replay_buffers["__all__"].sample(num_items, **kwargs)
                t1 = time.time()
                logger.debug(f"[Sample] Underlying buffer sample took: {t1 - t0:.4f}s")
                
                decompressed_sample = decompress_sample_batch(raw_sample, self.compress_base)
                t2 = time.time()
                logger.debug(f"[Sample] Decompression took: {t2 - t1:.4f}s")
                logger.debug(f"[Sample] Total time for lockstep sample: {t2 - t_start:.4f}s")
                return decompressed_sample
                
            elif policy_id is not None:
                t0 = time.time()
                sample = self.replay_buffers[policy_id].sample(num_items, **kwargs)
                t1 = time.time()
                logger.debug(f"[Sample] Underlying buffer sample for policy '{policy_id}' took: {t1 - t0:.4f}s")

                # 智能判断是否需要解压：如果 obs 字段是 object 类型，说明还是压缩的
                if 'obs' in sample and isinstance(sample['obs'], np.ndarray) and sample['obs'].dtype == object:
                    # 仍然是压缩的，需要解压
                    sample = decompress_sample_batch(sample, self.compress_base)
                    t2 = time.time()
                    logger.debug(f"[Sample] Decompression for policy '{policy_id}' took: {t2 - t1:.4f}s")
                else:
                    # 已经是解压后的数据，无需再次解压
                    t2 = t1
                    logger.debug(f"[Sample] Data already decompressed for policy '{policy_id}'")
                
                ma_batch = MultiAgentBatch({policy_id: sample}, sample.count)
                t_end = time.time()
                logger.debug(f"[Sample] Total time for policy '{policy_id}': {t_end - t_start:.4f}s")
                return ma_batch

            else:
                # Default case: Sample from all policies independently.
                samples = {}
                t0 = time.time()
                for pid, replay_buffer in self.replay_buffers.items():
                    # 1. Sample compressed data from underlying buffers
                    sample = replay_buffer.sample(num_items, **kwargs)
                    samples[pid] = sample
                t1 = time.time()
                logger.debug(f"[Sample] Underlying buffer sample for all policies took: {t1 - t0:.4f}s")

                # 2. 智能解压samples
                decompressed_samples = {}
                t2 = time.time()
                for pid, sample in samples.items():
                    # 智能判断是否需要解压
                    if 'obs' in sample and isinstance(sample['obs'], np.ndarray) and sample['obs'].dtype == object:
                        # 仍然是压缩的，需要解压
                        decompressed_samples[pid] = decompress_sample_batch(sample, self.compress_base)
                    else:
                        # 已经是解压后的数据，无需再次解压
                        decompressed_samples[pid] = sample
                t3 = time.time()
                logger.debug(f"[Sample] Smart decompression for all policies took: {t3 - t2:.4f}s")

                ma_batch = MultiAgentBatch(decompressed_samples, sum(s.count for s in decompressed_samples.values()))
                t_end = time.time()
                logger.debug(f"[Sample] Total time for all policies: {t_end - t_start:.4f}s")
                return ma_batch

    def stats(self, debug: bool = False) -> Dict[str, Any]:
        """
        Returns stats about the replay buffer.

        Returns:
            A dictionary of stats.
        """
        stat = {
            "add_batch_time_ms": round(1000 * self.add_batch_timer.mean, 3),
            "replay_time_ms": round(1000 * self.replay_timer.mean, 3),
            "update_priorities_time_ms": round(
                1000 * self.update_priorities_timer.mean, 3
            ),
            "est_size_bytes": 0
        }
        total_estimated_bytes = 0
        for policy_id, replay_buffer in self.replay_buffers.items():
            # 调用我们重写后的 stats 方法
            policy_stats = replay_buffer.stats(debug=debug)
            total_estimated_bytes += policy_stats.get("est_size_bytes", 0)
            stat.update(
                {"policy_{}".format(policy_id): policy_stats}
            )
        stat["est_size_bytes"] = total_estimated_bytes
        return stat
