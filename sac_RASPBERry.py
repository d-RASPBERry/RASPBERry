import os
import ray
from pathlib import Path
from dynaconf import Dynaconf
from tqdm import tqdm
from typing import Dict, Any
import time

from utils import env_creator
from ray.tune.logger import JsonLogger
from ray.tune.registry import register_env
from ray.rllib.algorithms.sac import SAC
from replay_buffer.mpber_ram_saver_v8 import MultiAgentPrioritizedBlockReplayBuffer

# -------------------------------------------------------------
#  Simple SAC + RASPBERry entry script
#  NOTE: SAC 原生支持 replay_buffer_config，我们仅注入 RASPBERry 参数
# -------------------------------------------------------------

class RASPBERrySACTrainer:
    """Soft Actor-Critic with RASPBERry replay buffer"""
    def __init__(self, config: Dict[str, Any]):
        self.env_name = config.get("env_name", "BipedalWalker-v3")
        self.run_name = config.get("run_name", "SAC_RASPBERry_test")
        self.sub_buffer_size = config.get("sub_buffer_size", 8)
        self.compress_base = config.get("compress_base", -1)

        # Paths
        self.base = Path("~/data/BER/WSL").expanduser()
        self.log_dir = self.base / "logs" / self.run_name
        self.ckpt_dir = self.base / "checkpoints" / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        # ---------- Ray ----------
        ray.init(num_cpus=8, num_gpus=1, include_dashboard=False)

        # ---------- Env ----------
        register_env("Default", env_creator)
        env_cfg = {"id": self.env_name}
        env_example = env_creator(env_cfg)

        # ---------- Config ----------
        default_cfg = {
            "env_config": env_cfg,
            "logger_config": {"type": JsonLogger, "logdir": str(self.ckpt_dir)},
            "framework": "torch",
            "num_workers": 0,
            "replay_buffer_config": {
                "type": MultiAgentPrioritizedBlockReplayBuffer,
                "capacity": 1_000_000,
                "obs_space": env_example.observation_space,
                "action_space": env_example.action_space,
                "sub_buffer_size": self.sub_buffer_size,
                "compress_base": self.compress_base,
                "rollout_fragment_length": self.sub_buffer_size,
                "num_save": 50,
                "split_mini_batch": 4,
            },
            "train_batch_size": 256 // self.sub_buffer_size,
        }

        # 允许用户 YAML 覆盖
        try:
            setting = Dynaconf(settings_files="settings/sac.yml")
            user_hp = setting.get("hyper_parameters", {})
            default_cfg.update(user_hp)
        except Exception:
            pass

        trainer = SAC(config=default_cfg, env="Default")

        # ---------- Training loop ----------
        for i in tqdm(range(1000), desc="SAC Training"):
            res = trainer.train()
            if i % 50 == 0:
                print(f"Iter {i:4d} | reward: {res['episode_reward_mean']:.2f}")
            if i % 200 == 0 and i > 0:
                trainer.save(checkpoint_dir=str(self.ckpt_dir))

        ray.shutdown()


if __name__ == "__main__":
    cfg = {
        "env_name": "BipedalWalker-v3",
    }
    RASPBERrySACTrainer(cfg).run() 