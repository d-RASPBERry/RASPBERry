#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
env_creator 函数的观察空间和动作空间单元测试
专注于验证 obs 和 act 的正确性
"""

import sys
import os
import unittest
import numpy as np

# 添加父目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import env_creator


class TestEnvObsAct(unittest.TestCase):
    """测试环境的观察空间和动作空间"""

    def _test_step_return_format(self, env, env_type):
        """测试 step 返回值格式 (done vs terminated/truncated)"""
        print(f"  --- 测试 {env_type} step 返回值格式 ---")

        # 重置环境
        obs, _ = env.reset()

        # 采样一个动作
        if hasattr(env.action_space, 'sample'):
            action = env.action_space.sample()
        elif hasattr(env.action_space, 'n'):
            action = 0  # 使用第一个动作
        else:
            self.fail("无法确定动作格式")

        # 执行一步
        step_result = env.step(action)
        step_len = len(step_result)

        print(f"  step() 返回长度: {step_len}")
        print(f"  step() 返回类型: {[type(x).__name__ for x in step_result]}")

        if step_len == 4:
            # 旧版 Gym API: (obs, reward, done, info)
            obs, reward, done, info = step_result
            print(f"  ✓ 旧版 Gym API 格式: (obs, reward, done, info)")
            print(f"  done 类型: {type(done)}, 值: {done}")

            # 验证类型
            self.assertIsInstance(done, (bool, np.bool_), "done 应该是布尔类型")

        elif step_len == 5:
            # 新版 Gymnasium API: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = step_result
            print(f"  ✓ 新版 Gymnasium API 格式: (obs, reward, terminated, truncated, info)")
            print(f"  terminated 类型: {type(terminated)}, 值: {terminated}")
            print(f"  truncated 类型: {type(truncated)}, 值: {truncated}")

            # 验证类型
            self.assertIsInstance(terminated, (bool, np.bool_), "terminated 应该是布尔类型")
            self.assertIsInstance(truncated, (bool, np.bool_), "truncated 应该是布尔类型")

            # 计算等效的 done
            done = terminated or truncated
            print(f"  等效 done: {done}")

        else:
            self.fail(f"step() 返回了意外的长度: {step_len}, 期望 4 或 5")

        # 验证基本返回值
        self.assertTrue(hasattr(obs, 'shape'), "观察值应该有 shape 属性")
        self.assertIsInstance(reward, (int, float, np.number), "奖励应该是数值类型")
        self.assertIsInstance(info, dict, "info 应该是字典类型")

        print(f"  obs shape: {obs.shape}, reward: {reward}")
        return step_len

    def test_atari_obs_act(self):
        """测试 Atari 环境的 obs 和 act"""
        print("\n=== 测试 Atari 环境 ===")

        env_config = {"id": "Atari-PongNoFrameskip-v4"}
        env = env_creator(env_config)

        # 测试观察空间
        obs_space = env.observation_space
        print(f"观察空间: {obs_space}")
        print(f"obs shape: {obs_space.shape}")
        print(f"obs dtype: {obs_space.dtype}")

        # 测试动作空间
        act_space = env.action_space
        print(f"动作空间: {act_space}")
        print(f"动作数量: {act_space.n if hasattr(act_space, 'n') else 'N/A'}")

        # 实际重置环境验证
        obs, info = env.reset()
        print(f"实际 obs shape: {obs.shape}")
        print(f"实际 obs dtype: {obs.dtype}")
        print(f"obs 范围: [{obs.min():.3f}, {obs.max():.3f}]")

        # 验证观察空间一致性
        self.assertEqual(obs.shape, obs_space.shape)
        self.assertEqual(obs.dtype, obs_space.dtype)

        # 验证动作空间
        self.assertTrue(hasattr(act_space, 'n'))
        self.assertGreater(act_space.n, 0)

        # 测试 step 返回值格式
        step_format = self._test_step_return_format(env, "Atari")
        print(f"Atari 环境使用 {'新版 Gymnasium' if step_format == 5 else '旧版 Gym'} API")

    def test_minigrid_obs_act(self):
        """测试 MiniGrid 环境的 obs 和 act"""
        print("\n=== 测试 MiniGrid 环境 ===")

        env_config = {
            "id": "MiniGrid-Empty-5x5-v0",
            "tile_size": 8,
            "img_size": 84,
            "max_steps": 100
        }
        env = env_creator(env_config)

        # 测试观察空间
        obs_space = env.observation_space
        print(f"观察空间: {obs_space}")
        print(f"obs shape: {obs_space.shape}")
        print(f"obs dtype: {obs_space.dtype}")

        # 测试动作空间
        act_space = env.action_space
        print(f"动作空间: {act_space}")
        print(f"动作数量: {act_space.n if hasattr(act_space, 'n') else 'N/A'}")

        # 实际重置环境验证
        obs, info = env.reset()
        print(f"实际 obs shape: {obs.shape}")
        print(f"实际 obs dtype: {obs.dtype}")
        print(f"obs 范围: [{obs.min():.3f}, {obs.max():.3f}]")

        # 验证观察空间一致性
        self.assertEqual(obs.shape, obs_space.shape)
        self.assertEqual(obs.dtype, obs_space.dtype)

        # 验证动作空间
        self.assertTrue(hasattr(act_space, 'n'))
        self.assertGreater(act_space.n, 0)

        # 测试 step 返回值格式
        step_format = self._test_step_return_format(env, "MiniGrid")
        print(f"MiniGrid 环境使用 {'新版 Gymnasium' if step_format == 5 else '旧版 Gym'} API")

    def test_car_racing_obs_act(self):
        """测试 CarRacing 环境的 obs 和 act（如果可用）"""
        print("\n=== 测试 CarRacing 环境 ===")

        env_config = {"id": "CarRacing"}

        try:
            env = env_creator(env_config)

            # 测试观察空间
            obs_space = env.observation_space
            print(f"观察空间: {obs_space}")
            print(f"obs shape: {obs_space.shape}")
            print(f"obs dtype: {obs_space.dtype}")

            # 测试动作空间
            act_space = env.action_space
            print(f"动作空间: {act_space}")
            if hasattr(act_space, 'shape'):
                print(f"动作维度: {act_space.shape}")
            if hasattr(act_space, 'high'):
                print(f"动作范围: [{act_space.low}, {act_space.high}]")

            # 实际重置环境验证
            obs, info = env.reset()
            print(f"实际 obs shape: {obs.shape}")
            print(f"实际 obs dtype: {obs.dtype}")
            print(f"obs 范围: [{obs.min():.3f}, {obs.max():.3f}]")

            # 验证观察空间一致性
            self.assertEqual(obs.shape, obs_space.shape)
            self.assertEqual(obs.dtype, obs_space.dtype)

            # 测试 step 返回值格式
            step_format = self._test_step_return_format(env, "CarRacing")
            print(f"CarRacing 环境使用 {'新版 Gymnasium' if step_format == 5 else '旧版 Gym'} API")

        except Exception as e:
            print(f"CarRacing 环境不可用: {e}")
            self.skipTest("CarRacing 需要 Box2D 依赖")

    def test_user_scenario_obs_act(self):
        """测试用户具体使用场景的 obs 和 act"""
        print("\n=== 测试用户场景 ===")

        # 模拟用户的使用方式
        env_name = "Atari-PongNoFrameskip-v4"
        env_str = env_name.split("-")[1].replace("NoFrameskip", "")

        print(f"设置环境: {env_name}")
        print(f"环境字符串: {env_str}")

        # 创建环境
        _env = env_creator({"id": env_name})
        reset_result = _env.reset()

        print(f"reset()返回类型: {type(reset_result)}, 长度: {len(reset_result)}")

        _env_rest = reset_result[0]

        print(f"✓ 环境 {env_str} 注册成功, obs: {_env_rest.shape} {str(_env_rest.dtype)}, act: {_env.action_space}")
        print(f"obs范围: [{_env_rest.min():.3f}, {_env_rest.max():.3f}]")
        print(f"environment wrapper链: {_env}")

        # 验证期望的形状和类型
        expected_shape = (84, 84, 4)
        self.assertEqual(_env_rest.shape, expected_shape,
                         f"期望 obs 形状 {expected_shape}, 实际 {_env_rest.shape}")

        # 验证数据类型（根据之前的测试，应该是 uint8）
        self.assertIn(_env_rest.dtype, [np.uint8, np.float32, np.float64],
                      f"obs dtype {_env_rest.dtype} 不在预期范围内")

        # 验证动作空间
        self.assertTrue(hasattr(_env.action_space, 'n'), "Atari 环境应该有离散动作空间")
        self.assertGreater(_env.action_space.n, 0, "动作数量应该大于0")

        # 测试 step 返回值格式
        step_format = self._test_step_return_format(_env, "用户场景")
        print(f"用户场景环境使用 {'新版 Gymnasium' if step_format == 5 else '旧版 Gym'} API")

    def test_multiple_atari_envs_obs_act(self):
        """测试多个 Atari 环境的 obs 和 act 一致性"""
        print("\n=== 测试多个 Atari 环境 ===")

        atari_envs = [
            "Atari-PongNoFrameskip-v4",
            "Atari-BreakoutNoFrameskip-v4",
            # 只测试这两个，避免测试时间过长
        ]

        results = []
        step_formats = []

        for env_id in atari_envs:
            print(f"\n测试 {env_id}:")
            try:
                env = env_creator({"id": env_id})
                obs, _ = env.reset()

                # 测试 step 返回值格式
                step_format = self._test_step_return_format(env, f"Atari-{env_id.split('-')[1]}")
                step_formats.append(step_format)

                result = {
                    "id": env_id,
                    "obs_shape": obs.shape,
                    "obs_dtype": obs.dtype,
                    "act_space": env.action_space.n,
                    "obs_range": [obs.min(), obs.max()],
                    "step_format": step_format
                }

                results.append(result)

                print(f"  obs: {result['obs_shape']} {result['obs_dtype']}")
                print(f"  act: {result['act_space']} 个动作")
                print(f"  范围: [{result['obs_range'][0]:.1f}, {result['obs_range'][1]:.1f}]")
                print(f"  API: {'新版 Gymnasium' if step_format == 5 else '旧版 Gym'}")

            except Exception as e:
                print(f"  ❌ 失败: {e}")

        # 验证所有 Atari 环境的观察空间形状一致
        if len(results) > 1:
            shapes = [r['obs_shape'] for r in results]
            dtypes = [r['obs_dtype'] for r in results]

            self.assertTrue(all(s == shapes[0] for s in shapes),
                            f"不同 Atari 环境的 obs 形状不一致: {shapes}")
            print(f"\n✓ 所有 Atari 环境 obs 形状一致: {shapes[0]}")

            # 验证所有 Atari 环境的 step 格式一致
            if step_formats:
                self.assertTrue(all(f == step_formats[0] for f in step_formats),
                                f"不同 Atari 环境的 step 返回格式不一致: {step_formats}")
                api_type = "新版 Gymnasium" if step_formats[0] == 5 else "旧版 Gym"
                print(f"✓ 所有 Atari 环境 step 格式一致: {api_type} API")

    def run_comprehensive_test(self):
        """运行综合测试报告"""
        print("\n" + "=" * 50)
        print("环境 obs 和 act 综合验证报告")
        print("=" * 50)

        test_results = {}

        # 测试配置
        test_configs = [
            {
                "name": "Atari-Pong",
                "config": {"id": "Atari-PongNoFrameskip-v4"},
                "expected_obs_shape": (84, 84, 4),
                "expected_obs_dtype": np.uint8
            },
            {
                "name": "MiniGrid-Empty",
                "config": {
                    "id": "MiniGrid-Empty-5x5-v0",
                    "tile_size": 8,
                    "img_size": 84,
                    "max_steps": 100
                },
                "expected_obs_shape": (84, 84, 3),
                "expected_obs_dtype": np.uint8
            }
        ]

        for test_config in test_configs:
            print(f"\n【{test_config['name']}】")
            try:
                env = env_creator(test_config['config'])
                obs, _ = env.reset()

                # 收集结果
                result = {
                    "success": True,
                    "obs_shape": obs.shape,
                    "obs_dtype": obs.dtype,
                    "action_space": str(env.action_space),
                    "obs_range": [float(obs.min()), float(obs.max())],
                    "memory_usage_kb": obs.nbytes / 1024
                }

                # 验证期望
                shape_match = obs.shape == test_config['expected_obs_shape']
                dtype_match = obs.dtype == test_config['expected_obs_dtype']

                print(
                    f"  obs 形状: {obs.shape} {'✓' if shape_match else '✗ 期望 ' + str(test_config['expected_obs_shape'])}")
                print(
                    f"  obs 类型: {obs.dtype} {'✓' if dtype_match else '✗ 期望 ' + str(test_config['expected_obs_dtype'])}")
                print(f"  动作空间: {env.action_space}")
                print(f"  obs 范围: [{obs.min():.1f}, {obs.max():.1f}]")
                print(f"  内存占用: {result['memory_usage_kb']:.1f} KB")

                test_results[test_config['name']] = result

            except Exception as e:
                print(f"  ❌ 失败: {e}")
                test_results[test_config['name']] = {"success": False, "error": str(e)}

        return test_results


def main():
    """主函数：运行所有测试"""
    print("开始运行 env_creator obs/act 单元测试...")

    # 创建测试实例
    test_instance = TestEnvObsAct()

    # 运行综合测试
    results = test_instance.run_comprehensive_test()

    # 运行标准单元测试
    print("\n" + "=" * 50)
    print("运行标准单元测试")
    print("=" * 50)

    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    main()
