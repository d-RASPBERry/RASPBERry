import os
import sys
import time
import unittest
from typing import List, Tuple, Dict, Any
import statistics

import blosc
import numpy as np
import warnings
import ray

from ray.rllib.policy.policy import SampleBatch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import env_creator
from replay_buffer.raspberry import PrioritizedBlockReplayBuffer, decompress_sample_batch

# 过滤测试中无关的警告输出
warnings.filterwarnings("ignore", message=r".*Using the latest versioned environment.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*SwigPy.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*swigvarlink.*", category=DeprecationWarning)


def _collect_transitions(env, num_steps: int):
    """收集真实环境的transitions - 复用test_compress_node.py的逻辑"""
    obs_list, new_obs_list = [], []
    actions_list, rewards_list = [], []
    terms_list, truncs_list = [], []

    obs, _ = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)

        obs_list.append(obs)
        new_obs_list.append(next_obs)
        actions_list.append(action)
        rewards_list.append(float(reward))
        terms_list.append(bool(terminated))
        truncs_list.append(bool(truncated))

        obs = env.reset()[0] if (terminated or truncated) else next_obs

    return obs_list, new_obs_list, actions_list, rewards_list, terms_list, truncs_list


def _transitions_to_batch(obs_list, new_obs_list, actions_list, rewards_list, terms_list, truncs_list) -> SampleBatch:
    """将transitions转换为SampleBatch"""
    return SampleBatch({
        "obs": np.asarray(obs_list),
        "new_obs": np.asarray(new_obs_list),
        "actions": np.asarray(actions_list),
        "rewards": np.asarray(rewards_list, dtype=np.float32),
        "terminateds": np.asarray(terms_list),
        "truncateds": np.asarray(truncs_list),
        "weights": np.ones((len(obs_list),), dtype=np.float32),
    })


class TestRASPBERryPerformance(unittest.TestCase):
    """基于真实环境数据的RASPBERry性能测试 - 使用优化后的默认配置(nthread=4)"""

    def setUp(self):
        """设置测试环境"""
        # 确保ray已初始化但不重复初始化
        if not ray.is_initialized():
            ray.init(local_mode=True, include_dashboard=False)

    def tearDown(self):
        """清理测试环境"""
        if ray.is_initialized():
            ray.shutdown()

    def test_atari_pong_performance(self):
        """测试Atari Pong环境的压缩性能"""
        print("\n" + "=" * 60)
        print("ATARI PONG 真实性能测试")
        print("=" * 60)

        env = env_creator({"id": "Pong"})

        # 创建buffer，使用默认的4线程配置
        buffer = PrioritizedBlockReplayBuffer(
            obs_space=env.observation_space,
            action_space=env.action_space,
            sub_buffer_size=32,
            beta=0.6,
            capacity=100000,
            alpha=0.6
        )

        print("收集真实环境数据...")
        # todo: 加一个统计组

        # 测试多个block的填充性能
        num_blocks = 5000
        fill_times = []

        for block_i in range(num_blocks):
            # 收集20步真实transitions
            obs_list, new_obs_list, actions_list, rewards_list, terms_list, truncs_list = _collect_transitions(env, 20)
            batch = _transitions_to_batch(obs_list, new_obs_list, actions_list, rewards_list, terms_list, truncs_list)

            # 测量填充时间
            start_time = time.time()
            buffer.add(batch)
            fill_time = time.time() - start_time
            fill_times.append(fill_time)

            if (block_i + 1) % 20 == 0:
                print(f"  已填充 {block_i + 1}/{num_blocks} blocks")

        # 测试采样性能
        print("测试采样性能...")
        sample_times = []
        decompress_times = []

        for i in range(500):
            # 采样压缩数据
            start_time = time.time()
            compressed_batch = buffer.sample(512)  # 指定采样数量
            if compressed_batch is not None:
                sample_time = time.time() - start_time
                sample_times.append(sample_time)

                # 解压数据
                start_time = time.time()
                decompressed_batch = decompress_sample_batch(compressed_batch)
                decompress_time = time.time() - start_time
                decompress_times.append(decompress_time)
                # todo: 验证一下 decompress的结果，加到统计组中

        # 计算统计结果
        # todo: 这里的逻辑移动到统计组中
        avg_fill_time = statistics.mean(fill_times) * 1000 if fill_times else 0  # ms
        avg_sample_time = statistics.mean(sample_times) * 1000 if sample_times else 0  # ms
        avg_decompress_time = statistics.mean(decompress_times) * 1000 if decompress_times else 0  # ms
        total_transitions = num_blocks * 32
        fill_throughput = total_transitions / sum(fill_times) if fill_times else 0
        sample_throughput = len(sample_times) * 512 / sum(sample_times) if sample_times else 0  # batch_size=512

        # 输出结果
        print(f"\n--- ATARI PONG 性能结果 ---")
        print(f"填充性能:")
        print(f"  平均填充时间: {avg_fill_time:.2f}ms/block (32 transitions)")
        print(f"  填充吞吐量: {fill_throughput:.0f} transitions/sec")
        print(f"采样性能:")
        print(f"  平均采样时间: {avg_sample_time:.2f}ms")
        print(f"  平均解压时间: {avg_decompress_time:.2f}ms")
        print(f"  采样吞吐量: {sample_throughput:.0f} transitions/sec")
        print(f"总transitions数: {total_transitions}")

        # 获取压缩统计
        stats = buffer.stats()
        memory_gb = stats.get('estimated_memory_gb', 0)
        print(f"内存使用: {memory_gb:.2f}GB")

        # 性能断言
        self.assertLess(avg_fill_time, 50.0)  # 填充时间<50ms
        self.assertLess(avg_sample_time, 20.0)  # 采样时间<20ms
        self.assertGreater(fill_throughput, 1000)  # 填充吞吐量>1000 t/s

    # def test_car_racing_performance(self):
    #     """测试Car Racing环境的压缩性能"""
    #     print("\n" + "="*60)
    #     print("CAR RACING 真实性能测试")
    #     print("="*60)
    #
    #     env = env_creator({"id": "CarRacing"})
    #
    #     # 创建buffer，使用默认的4线程配置
    #     buffer = PrioritizedBlockReplayBuffer(
    #         obs_space=env.observation_space,
    #         action_space=env.action_space,
    #         sub_buffer_size=16,  # 较小block size
    #         beta=0.4,
    #         capacity=5000,  # 较小容量因为CarRacing图像更大
    #         alpha=0.6
    #     )
    #
    #     print("收集真实环境数据...")
    #
    #     # 测试多个block的填充性能
    #     num_blocks = 50  # 较少blocks因为CarRacing更耗时
    #     fill_times = []
    #
    #     for block_i in range(num_blocks):
    #         # 收集16步真实transitions
    #         obs_list, new_obs_list, actions_list, rewards_list, terms_list, truncs_list = _collect_transitions(env, 16)
    #         batch = _transitions_to_batch(obs_list, new_obs_list, actions_list, rewards_list, terms_list, truncs_list)
    #
    #         # 测量填充时间
    #         start_time = time.time()
    #         buffer.add(batch)
    #         fill_time = time.time() - start_time
    #         fill_times.append(fill_time)
    #
    #         if (block_i + 1) % 10 == 0:
    #             print(f"  已填充 {block_i + 1}/{num_blocks} blocks")
    #
    #     # 测试采样性能
    #     print("测试采样性能...")
    #     sample_times = []
    #     decompress_times = []
    #
    #     for i in range(30):  # 较少采样次数
    #         # 采样压缩数据
    #         start_time = time.time()
    #         compressed_batch = buffer.sample(256)  # CarRacing使用256采样数量
    #         if compressed_batch is not None:
    #             sample_time = time.time() - start_time
    #             sample_times.append(sample_time)
    #
    #             # 解压数据
    #             start_time = time.time()
    #             decompressed_batch = decompress_sample_batch(compressed_batch)
    #             decompress_time = time.time() - start_time
    #             decompress_times.append(decompress_time)
    #
    #     # 计算统计结果
    #     avg_fill_time = statistics.mean(fill_times) * 1000  # ms
    #     avg_sample_time = statistics.mean(sample_times) * 1000  # ms
    #     avg_decompress_time = statistics.mean(decompress_times) * 1000  # ms
    #
    #     total_transitions = num_blocks * 16
    #     fill_throughput = total_transitions / sum(fill_times)
    #     sample_throughput = len(sample_times) * 256 / sum(sample_times)  # batch_size=256
    #
    #     # 输出结果
    #     print(f"\n--- CAR RACING 性能结果 ---")
    #     print(f"填充性能:")
    #     print(f"  平均填充时间: {avg_fill_time:.2f}ms/block (16 transitions)")
    #     print(f"  填充吞吐量: {fill_throughput:.0f} transitions/sec")
    #     print(f"采样性能:")
    #     print(f"  平均采样时间: {avg_sample_time:.2f}ms")
    #     print(f"  平均解压时间: {avg_decompress_time:.2f}ms")
    #     print(f"  采样吞吐量: {sample_throughput:.0f} transitions/sec")
    #     print(f"总transitions数: {total_transitions}")
    #
    #     # 获取压缩统计
    #     stats = buffer.stats()
    #     memory_gb = stats.get('estimated_memory_gb', 0)
    #     print(f"内存使用: {memory_gb:.2f}GB")
    #
    #     # 性能断言
    #     self.assertLess(avg_fill_time, 100.0)  # 填充时间<100ms
    #     self.assertLess(avg_sample_time, 50.0)  # 采样时间<50ms
    #     self.assertGreater(fill_throughput, 200)  # 填充吞吐量>200 t/s


if __name__ == '__main__':
    unittest.main(verbosity=2)
