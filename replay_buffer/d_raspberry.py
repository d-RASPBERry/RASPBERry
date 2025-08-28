import ray
import logging
import numpy as np
import time
from gymnasium.spaces import Space
from replay_buffer.raspberry import PrioritizedBlockReplayBuffer, decompress_sample_batch
from typing import Dict, Optional, Any, List, Union
from ray.rllib.utils.typing import PolicyID, SampleBatchType
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.replay_buffers import StorageUnit
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import ReplayMode, merge_dicts_with_warning
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import MultiAgentPrioritizedReplayBuffer
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from replay_buffer.compress_replay_node import CompressReplayNode
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.util.debug import log_once
from utils import split_list_into_n_parts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@DeveloperAPI
class MultiAgentPrioritizedBlockReplayBuffer(MultiAgentPrioritizedReplayBuffer):
    """多智能体优先级块重放缓冲区"""

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
            split_mini_batch: int = 10,
            num_save: int = 200,
            store: int = 2000,
            compress_base: int = -1,
            **kwargs
    ):
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

        # 存储块级参数以供后续使用
        self.sub_buffer_size = sub_buffer_size
        self.compress_base = compress_base
        # 确保 prioritized_replay_eps 是数值类型
        self.prioritized_replay_eps = float(prioritized_replay_eps)

        pber_config = {
            "type": PrioritizedBlockReplayBuffer,
            "action_space": action_space,
            "obs_space": obs_space,
            "storage_unit": StorageUnit.FRAGMENTS,
            "sub_buffer_size": sub_buffer_size,
            "prioritized_replay_alpha": prioritized_replay_alpha,
            "prioritized_replay_beta": prioritized_replay_beta,
            "split_mini_batch": split_mini_batch,
            "compress_base": compress_base,
            "prioritized_replay_eps": prioritized_replay_eps,
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
    @override(MultiAgentPrioritizedReplayBuffer)
    def update_priorities(self, prio_dict: Dict) -> None:
        """更新底层重放缓冲区的优先级，直接使用块级逻辑"""
        with self.update_priorities_timer:
            for policy_id, (batch_indexes, td_errors) in prio_dict.items():

                # 直接执行块级逻辑转换
                block_indices, block_priorities = self._convert_to_block_priorities(
                    batch_indexes, td_errors
                )
                logger.debug(f"Block priority update: "
                             f"{len(batch_indexes)} samples -> {len(block_indices)} blocks")

                # 更新底层缓冲区
                if hasattr(self.replay_buffers[policy_id], 'update_priorities'):
                    self.replay_buffers[policy_id].update_priorities(
                        block_indices, block_priorities
                    )
                else:
                    logger.warning(f"Policy {policy_id} replay buffer does not support priority updates")

    def _convert_to_block_priorities(self, batch_indexes: np.ndarray, td_errors: np.ndarray) -> tuple:
        """将样本级数据转换为块级数据
        
        Args:
            batch_indexes: 原始批次索引 [batch_size]
            td_errors: 原始TD误差 [batch_size]
            
        Returns:
            tuple: (block_indices, block_priorities)
        """
        try:
            # 重新整形为块结构
            block_indices = batch_indexes.reshape(-1, self.sub_buffer_size)[:, 0]
            block_td_errors = td_errors.reshape(-1, self.sub_buffer_size).mean(axis=1)

            # 计算块级优先级
            block_priorities = np.abs(block_td_errors) + self.prioritized_replay_eps

            logger.debug(f"Block conversion: {batch_indexes.shape} -> {block_indices.shape}, "
                         f"TD errors: {td_errors.shape} -> {block_td_errors.shape}")

            return block_indices, block_priorities

        except Exception as e:
            logger.warning(f"Block conversion failed: {e}, falling back to direct processing")
            # 回退到原始处理方式
            return batch_indexes, np.abs(td_errors) + self.prioritized_replay_eps

    def _maybe_split_into_policy_batches(self, batch: SampleBatchType) -> Dict:
        """根据重放模式拆分批次"""
        if isinstance(batch, MultiAgentBatch):
            return batch.policy_batches
        else:
            # 如果是单个SampleBatch，转换为多智能体格式
            return {"__default_policy__": batch}

    @DeveloperAPI
    @override(MultiAgentPrioritizedReplayBuffer)
    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """添加批次到适当的策略重放缓冲区"""
        if batch is None:
            if log_once("empty_batch_added_to_buffer"):
                logger.info(
                    "A batch that is `None` was added to {}. This can be "
                    "normal at the beginning of execution but might "
                    "indicate an issue.".format(type(self).__name__)
                )
            return

        batch = batch.copy()
        batch = batch.as_multi_agent()

        with self.add_batch_timer:
            pids_and_batches = self._maybe_split_into_policy_batches(batch)
            for policy_id, sample_batch in pids_and_batches.items():
                if len(sample_batch) == 1:
                    self._add_to_underlying_buffer(policy_id, sample_batch)
                else:
                    time_slices = sample_batch.timeslices(size=self.rollout_fragment_length)
                    for s_batch in time_slices:
                        self._add_to_underlying_buffer(policy_id, s_batch)

        self._num_added += batch.count

    @DeveloperAPI
    @override(MultiAgentPrioritizedReplayBuffer)
    def sample(
            self, num_items: int, policy_id: Optional[PolicyID] = None, **kwargs
    ) -> Optional[SampleBatchType]:
        """采样数据并智能解压"""
        t_start = time.time()
        logger.debug(f"Starting sample for policy '{policy_id}' with {num_items} items.")

        kwargs = merge_dicts_with_warning(self.underlying_buffer_call_args, kwargs)

        # 从 kwargs 或父类/底层缓冲动态获取 beta（RLlib 可能以 kwargs 方式传入退火后的 beta）
        beta = kwargs.pop("beta", None)
        if beta is None:
            beta = getattr(self, "prioritized_replay_beta", None)
        if beta is None:
            try:
                any_buf = next(iter(self.replay_buffers.values()))
                beta = getattr(any_buf, "prioritized_replay_beta", getattr(any_buf, "beta", None))
            except Exception:
                beta = None
        if beta is None:
            beta = 0.4

        with self.replay_timer:
            if self.replay_mode == ReplayMode.LOCKSTEP:
                assert policy_id is None, "`policy_id` specifier not allowed in `lockstep` mode!"

                raw_sample = self.replay_buffers["__all__"].sample(
                    num_items, beta=beta, **kwargs
                )
                if raw_sample is None:
                    return None

                # 若已经是解压后的样本则直接返回
                if not self._is_compressed(raw_sample):
                    return raw_sample

                return decompress_sample_batch(raw_sample, self.compress_base)

            elif policy_id is not None:
                sample = self.replay_buffers[policy_id].sample(
                    num_items, beta=beta, **kwargs
                )
                if sample is None:
                    return None

                # 智能判断是否需要解压
                if self._is_compressed(sample):
                    sample = decompress_sample_batch(sample, self.compress_base)

                ma_batch = MultiAgentBatch({policy_id: sample}, sample.count)
                return ma_batch

            else:
                # 从所有策略独立采样
                samples = {}
                for pid, replay_buffer in self.replay_buffers.items():
                    sample = replay_buffer.sample(num_items, beta=beta, **kwargs)
                    if sample is not None:
                        # 智能解压
                        if self._is_compressed(sample):
                            sample = decompress_sample_batch(sample, self.compress_base)
                        samples[pid] = sample

                if samples:
                    total_count = sum(s.count for s in samples.values())
                    ma_batch = MultiAgentBatch(samples, total_count)
                    return ma_batch
                else:
                    return None

    def _is_compressed(self, sample: SampleBatch) -> bool:
        """判断样本是否仍然是压缩状态"""
        return ('obs' in sample and
                isinstance(sample['obs'], np.ndarray) and
                sample['obs'].dtype == object)

    def stats(self, debug: bool = False) -> Dict[str, Any]:
        """返回重放缓冲区的统计信息"""
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
            policy_stats = replay_buffer.stats(debug=debug)
            total_estimated_bytes += policy_stats.get("est_size_bytes", 0)
            stat.update(
                {"policy_{}".format(policy_id): policy_stats}
            )
        stat["est_size_bytes"] = total_estimated_bytes
        return stat


@ray.remote(num_cpus=1, max_calls=50)
def _parallel_node_sample(node_data, compress_base=-1):
    """并行调用node的sample方法"""
    from .replay_node import CompressedReplayNode
    # 创建临时node并存储压缩数据
    temp_node = CompressedReplayNode(buffer_size=1, obs_space=None, action_space=None, compress_base=compress_base)
    temp_node.compress_and_store(node_data, weight=0.01)
    return temp_node.sample()
