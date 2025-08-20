import os
import sys
import time
import unittest

import blosc
import numpy as np
import warnings

from ray.rllib.policy.policy import SampleBatch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import env_creator
from replay_buffer.compress_replay_node import CompressReplayNode

# 过滤测试中无关的警告输出
warnings.filterwarnings(
    "ignore",
    message=r".*Using the latest versioned environment.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*SwigPy.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*swigvarlink.*",
    category=DeprecationWarning,
)

def _add_to_node(node: CompressReplayNode, obs_list, new_obs_list, actions_list, rewards_list, terms_list,
                 truncs_list):
    num = len(obs_list)
    for i in range(num):
        batch = SampleBatch({
            "obs": np.expand_dims(np.asarray(obs_list[i]), 0),
            "new_obs": np.expand_dims(np.asarray(new_obs_list[i]), 0),
            "actions": np.expand_dims(np.asarray(actions_list[i]), 0),
            "rewards": np.asarray([rewards_list[i]], dtype=np.float32),
            "terminateds": np.asarray([terms_list[i]]),
            "truncateds": np.asarray([truncs_list[i]]),
        })
        node.add(batch)


def _make_baseline_batch(obs_list, new_obs_list, actions_list, rewards_list, terms_list,
                         truncs_list) -> SampleBatch:
    # 在外部构造一份“拼接后的原始批次”，不改变 node 内部行为
    return SampleBatch({
        "obs": np.asarray(obs_list),
        "new_obs": np.asarray(new_obs_list),
        "actions": np.asarray(actions_list),
        "rewards": np.asarray(rewards_list, dtype=np.float32),
        "terminateds": np.asarray(terms_list),
        "truncateds": np.asarray(truncs_list),
        "weights": np.ones((len(obs_list),), dtype=np.float32),
    })


def _assert_compressed_matches_original(compressed, baseline_batch: SampleBatch, compress_base: int):
    # 解压 obs/new_obs
    obs_unpacked = blosc.unpack_array(compressed["obs"][0])
    new_obs_unpacked = blosc.unpack_array(compressed["new_obs"][0])

    # 根据 compress_base 规则：-1 将 batch 维(0)移到最后；>=0 将指定维移到末尾
    def transpose_by_base(arr, base):
        rank = arr.ndim
        if rank <= 1:
            return arr
        if base == -1:
            axes = list(range(1, rank)) + [0]
        else:
            axes = list(range(rank))
            to_move = axes.pop(base)
            axes.append(to_move)
        return np.transpose(arr, axes)

    expected_obs = transpose_by_base(baseline_batch["obs"], compress_base)
    expected_new_obs = transpose_by_base(baseline_batch["new_obs"], compress_base)

    np.testing.assert_array_equal(obs_unpacked, expected_obs)
    np.testing.assert_array_equal(new_obs_unpacked, expected_new_obs)

    # 其他字段值相等
    np.testing.assert_array_equal(compressed["actions"], baseline_batch["actions"])
    np.testing.assert_allclose(compressed["rewards"], baseline_batch["rewards"], rtol=0, atol=0)
    np.testing.assert_array_equal(compressed["terminateds"], baseline_batch["terminateds"])
    np.testing.assert_array_equal(compressed["truncateds"], baseline_batch["truncateds"])
    np.testing.assert_array_equal(compressed["weights"], baseline_batch["weights"])


def _collect_transitions(env, num_steps: int):
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

    # 返回原始列表，更简洁，转换放在使用处
    return obs_list, new_obs_list, actions_list, rewards_list, terms_list, truncs_list


class TestCompressReplayNode(unittest.TestCase):
    """仅测试两种环境：Atari(Pong) 与 CarRacing，均基于真实环境转移收集32步。"""

    def _run_case(self, env_id: str, compression_level: int):
        env = env_creator({"id": env_id})

        # 收集 32 步转移（外部baseline将用到）
        obs_list, new_obs_list, actions_list, rewards_list, terms_list, truncs_list = _collect_transitions(env, 32)

        # 构建节点并逐条加入
        node = CompressReplayNode(
            buffer_size=32,
            obs_space=env.observation_space,
            action_space=env.action_space,
            compress_base=-1,
            compression_level=compression_level,
        )
        _add_to_node(node, obs_list, new_obs_list, actions_list, rewards_list, terms_list, truncs_list)
        self.assertTrue(node.is_ready())

        t0 = time.time()
        compressed, weight = node.sample()
        dt = time.time() - t0

        baseline_batch = _make_baseline_batch(
            obs_list, new_obs_list, actions_list, rewards_list, terms_list, truncs_list
        )
        _assert_compressed_matches_original(compressed, baseline_batch, compress_base=-1)

        # 最小结构与权重断言
        self.assertIsInstance(compressed["obs"][0], (bytes, bytearray))
        self.assertIsInstance(weight, float)
        self.assertGreater(weight, 0)

        # 简单压缩率打印（不作硬性断言）
        orig_obs_bytes = baseline_batch["obs"].nbytes
        orig_new_obs_bytes = baseline_batch["new_obs"].nbytes
        comp_obs_bytes = len(compressed["obs"][0])
        comp_new_obs_bytes = len(compressed["new_obs"][0])
        obs_rate = comp_obs_bytes / max(1, orig_obs_bytes)
        new_obs_rate = comp_new_obs_bytes / max(1, orig_new_obs_bytes)

        print(f"{env_id}:")
        print(f"  compress_time={dt * 1000:.2f} ms, weight={weight:.4f}")
        print(f"  obs_compress_rate={obs_rate:.3f} ({comp_obs_bytes}/{orig_obs_bytes})")
        print(f"  new_obs_compress_rate={new_obs_rate:.3f} ({comp_new_obs_bytes}/{orig_new_obs_bytes})")

    def test_atari_pong(self):
        self._run_case("Pong", compression_level=6)

    def test_car_racing(self):
        self._run_case("CarRacing", compression_level=5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
