import os
import ray
from pathlib import Path
from dynaconf import Dynaconf
from tqdm import tqdm
from typing import Dict, Any

from utils import env_creator
from ray.tune.logger import JsonLogger
from ray.tune.registry import register_env
from algorithms.apex_ddqn import ApexDDQNWithDPBER  # 自定义 Ape-X DDQN 版本，支持RASPBERry
from replay_buffer.mpber_ram_saver_v8 import MultiAgentPrioritizedBlockReplayBuffer

class RASPBERryApexTrainer:
    """Ape-X DDQN with RASPBERry block replay buffer"""
    def __init__(self, config: Dict[str, Any]):
        self.env_name = config.get("env_name", "BreakoutNoFrameskip-v4")
        self.run_name = config.get("run_name", "APEX_RASPBERry_test")
        self.sub_buffer_size = config.get("sub_buffer_size", 16)
        self.compress_base = config.get("compress_base", -1)

        # Paths
        self.base = Path("~/data/BER/WSL").expanduser()
        self.log_dir = self.base / "logs" / self.run_name
        self.ckpt_dir = self.base / "checkpoints" / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        ray.init(num_cpus=20, num_gpus=1, include_dashboard=False)

        register_env("Atari", env_creator)
        env_cfg = {"id": self.env_name}
        env_example = env_creator(env_cfg)

        default_cfg = Dynaconf(settings_files="settings/ddqn_atari.yml").hyper_parameters.to_dict()
        default_cfg["env_config"] = env_cfg
        default_cfg["logger_config"] = {"type": JsonLogger, "logdir": str(self.ckpt_dir)}

        # Inject RASPBERry buffer
        rb_cfg = {
            **default_cfg["replay_buffer_config"],
            "type": MultiAgentPrioritizedBlockReplayBuffer,
            "obs_space": env_example.observation_space,
            "action_space": env_example.action_space,
            "sub_buffer_size": self.sub_buffer_size,
            "worker_side_prioritization": False,
            "replay_buffer_shards_colocated_with_driver": True,
            "rollout_fragment_length": self.sub_buffer_size,
            "num_save": 100,
            "split_mini_batch": 4,
            "compress_base": self.compress_base,
        }
        default_cfg["replay_buffer_config"] = rb_cfg
        default_cfg["train_batch_size"] = int(default_cfg["train_batch_size"] / self.sub_buffer_size)
        default_cfg["optimizer"] = {"num_replay_buffer_shards": 4}
        default_cfg["num_workers"] = 4
        default_cfg["num_envs_per_worker"] = 10

        trainer = ApexDDQNWithDPBER(config=default_cfg, env="Atari")

        for i in tqdm(range(2000), desc="Ape-X Training"):
            res = trainer.train()
            if i % 20 == 0:
                print(f"Iter {i} | reward_mean: {res['episode_reward_mean']:.2f}")
            if i % 500 == 0 and i > 0:
                trainer.save(checkpoint_dir=str(self.ckpt_dir))

        ray.shutdown()


if __name__ == "__main__":
    cfg = {}
    RASPBERryApexTrainer(cfg).run() 