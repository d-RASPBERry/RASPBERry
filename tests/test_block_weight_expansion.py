import unittest
import numpy as np
from gymnasium import spaces

from ray.rllib.policy.sample_batch import SampleBatch

from replay_buffer.raspberry import PrioritizedBlockReplayBuffer


class TestBlockWeightExpansion(unittest.TestCase):
    def test_weights_are_shared_within_block(self):
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(2)

        sub_buffer_size = 4
        rb = PrioritizedBlockReplayBuffer(
            obs_space=obs_space,
            action_space=action_space,
            sub_buffer_size=sub_buffer_size,
            capacity=100,
            alpha=0.6,
            storage_unit="timesteps",
        )

        # 构造两块，每块 4 条 transition，设置不同的样本权重以便区分
        def make_batch(start_weight: float) -> SampleBatch:
            obs = np.random.randn(sub_buffer_size, 4).astype(np.float32)
            new_obs = np.random.randn(sub_buffer_size, 4).astype(np.float32)
            actions = np.random.randint(0, 2, size=(sub_buffer_size,)).astype(np.int32)
            rewards = np.random.randn(sub_buffer_size).astype(np.float32)
            terminateds = np.zeros(sub_buffer_size, dtype=np.int32)
            truncateds = np.zeros(sub_buffer_size, dtype=np.int32)
            weights = np.full((sub_buffer_size,), start_weight, dtype=np.float32)
            return SampleBatch(
                {
                    "obs": obs,
                    "new_obs": new_obs,
                    "actions": actions,
                    "rewards": rewards,
                    "terminateds": terminateds,
                    "truncateds": truncateds,
                    "weights": weights,
                }
            )

        rb.add(make_batch(0.1))  # 第1块
        rb.add(make_batch(0.9))  # 第2块

        out = rb.sample(num_items=2)
        self.assertIsNotNone(out)
        self.assertIn("weights", out)
        # 展开后的权重应当与 transition 数量一致
        self.assertEqual(out.count, len(out["weights"]))

        # 每个唯一权重出现的次数应当是 sub_buffer_size 的倍数（块内共享）
        unique_vals, counts = np.unique(out["weights"], return_counts=True)
        for c in counts:
            self.assertEqual(c % sub_buffer_size, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)


