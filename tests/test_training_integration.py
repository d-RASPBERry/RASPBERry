import os
import unittest
import sys
import logging
import psutil

# 将项目根目录加入 sys.path，确保从任意执行目录均可导入项目内模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray
import warnings

from ray.tune.registry import register_env

from utils import env_creator, flatten_dict

# Algorithms (use new config-style API to avoid deprecation warnings)
from ray.rllib.algorithms.dqn import DQNConfig

from replay_buffer.d_raspberry import MultiAgentPrioritizedBlockReplayBuffer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRASPBERryEndToEndTrainingAtari(unittest.TestCase):
    """端到端训练集成测试：env -> store -> train（基于 RLlib Algorithm 对象）"""

    def setUp(self) -> None:
        # 抑制 Ray 启动期间的资源告警（文件句柄/子进程未关闭等）
        warnings.simplefilter("ignore", ResourceWarning)
        # 屏蔽上游库的弃用/冗余告警（不影响功能，仅减少噪音）
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module=r"ray\.tune\.logger(\.unified)?"
        )
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module=r"ray\.rllib\.models\.torch\.visionnet"
        )
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module=r"ray\.rllib\.execution\.train_ops"
        )
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module=r"ray\.rllib\.algorithms\.simple_q"
        )
        if not ray.is_initialized():
            # 关闭 dashboard 以减小开销；避免使用已弃用的 local_mode
            ray.init(include_dashboard=False, num_gpus=1, num_cpus=12)
        # 参考 atari_dper.py：在 Ray 初始化后注册一次环境入口
        register_env("Atari", env_creator)

    def tearDown(self) -> None:
        if ray.is_initialized():
            ray.shutdown()

    def _build_rb_config(self, env_example, sub_buffer_size: int = 8, compress_base: int = -1):
        return {
            "type": MultiAgentPrioritizedBlockReplayBuffer,
            "capacity": 10000,
            "obs_space": env_example.observation_space,
            "action_space": env_example.action_space,
            "sub_buffer_size": sub_buffer_size,
            "rollout_fragment_length": sub_buffer_size,
            "storage_unit": "timesteps",
            "replay_mode": "independent",
            "prioritized_replay_alpha": 0.5,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_eps": 1e-6,
            "compress_base": compress_base,
        }

    def test_dqn_end_to_end(self):
        """DQN + RASPBERry 回放缓冲区（Atari）"""
        # 使用带版本的 Atari 环境，避免 Gymnasium 关于未指定版本的告警
        env_cfg = {"id": "Atari-Pong-v4"}
        env_example = env_creator(env_cfg)

        sub_buffer_size = 8
        # 使用 DQNConfig 构建算法，避免旧式 API 的弃用告警
        cfg = (
            DQNConfig()
            .environment(env="Atari", env_config=env_cfg)
            .framework("torch")
            .resources(num_gpus=1, num_cpus_per_worker=1, num_gpus_per_worker=0.01)
            .rollouts(
                num_rollout_workers=1,
                num_envs_per_worker=20,
                rollout_fragment_length=sub_buffer_size,
            )
            .exploration(exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "epsilon_timesteps": 200000,
                "final_epsilon": 0.01,
            })
            .training(
                replay_buffer_config=self._build_rb_config(env_example, sub_buffer_size=sub_buffer_size),
                train_batch_size=max(1, 32 // sub_buffer_size),
                target_network_update_freq=5000,
                num_steps_sampled_before_learning_starts=10000,
                adam_epsilon=0.00015,
                double_q=True,
                dueling=False,
                lr=6.25e-05,
                n_step=1,
                noisy=False,
                num_atoms=1,
            )
            .reporting(
                min_sample_timesteps_per_iteration=10000
            )
        )

        algo = cfg.build()
        last_result = None
        for i in range(1000):
            last_result = algo.train()

            # 每5次迭代打印一次内存信息
            if i % 5 == 0:
                print(f"\n=== 迭代 {i + 1} 内存-+报告 ===")

                # 直接从算法对象获取缓冲区统计信息
                try:
                    if hasattr(algo, 'local_replay_buffer') and hasattr(algo.local_replay_buffer, 'stats'):
                        buffer_stats = algo.local_replay_buffer.stats()
                        print("📊 缓冲区统计:")
                        print(f"  估计大小: {buffer_stats.get('est_size_bytes', 0)} bytes")
                        print(f"  估计大小: {buffer_stats.get('est_size_bytes', 0) / 1e9:.3f} GB")
                        print(f"  添加批次时间: {buffer_stats.get('add_batch_time_ms', 0):.2f} ms")
                        print(f"  重放时间: {buffer_stats.get('replay_time_ms', 0):.2f} ms")
                        print(f"  更新优先级时间: {buffer_stats.get('update_priorities_time_ms', 0):.2f} ms")

                        # 打印策略级别的统计信息
                        for key, value in buffer_stats.items():
                            if key.startswith('policy_'):
                                print(f"  {key}: {value}")
                    else:
                        print("⚠️  无法访问本地缓冲区统计信息")

                except Exception as e:
                    print(f"❌ 获取缓冲区统计信息失败: {e}")

                # 从训练结果获取性能信息
                perf = last_result.get('perf', {})
                print(f"💾 系统性能:")
                print(f"  CPU使用率: {perf.get('cpu_util_percent', 0):.1f}%")
                print(f"  内存使用率: {perf.get('ram_util_percent', 0):.1f}%")

                # 打印基本训练统计
                print(f"🎯 训练统计:")
                print(f"  环境步数: {last_result.get('num_env_steps_sampled', 0)}")
                print(f"  训练步数: {last_result.get('num_env_steps_trained', 0)}")
                print(f"  训练时间: {last_result.get('time_total_s', 0):.2f}s")
                print(f"  平均奖励: {last_result.get('episode_reward_mean', 'N/A')}")
                print(f"  完成回合数: {last_result.get('episodes_total', 0)}")

        # 断言：训练返回结果且包含关键统计字段
        self.assertIsInstance(last_result, dict)
        self.assertIn("episode_reward_mean", last_result)
        self.assertGreaterEqual(last_result.get("num_env_steps_sampled", 0), 0)

        algo.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
