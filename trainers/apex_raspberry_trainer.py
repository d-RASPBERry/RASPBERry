"""
Ape-X DQN Trainer with RASPBERry block-prioritized replay buffer.
"""

from typing import Dict, Any, Optional
from gymnasium.spaces import Space
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig
from utils import convert_np_arrays

from replay_buffer.d_raspberry import MultiAgentPrioritizedBlockReplayBuffer
import time
import json

from .base_trainer import BaseTrainer


class ApexDQNRaspberryTrainer(BaseTrainer):
    """
    Ape-X DQN trainer using RASPBERry MultiAgent Block PER buffer.
    """

    def __init__(self,
                 config: str,
                 env_name: str,
                 run_name: str,
                 log_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 mlflow: str = None,
                 obs_space: Space = None,
                 action_space: Space = None):
        super().__init__(config, env_name, run_name, log_path, checkpoint_path, use_mlflow=mlflow)
        self.obs_space = obs_space
        self.action_space = action_space

    def init_algorithm(self) -> Any:
        """
        Create and return Ape-X DQN with RASPBERry replay buffer.
        """
        self.log("Creating Ape-X DQN + RASPBERry algorithm...", "TRAIN")
        env_id = self.setup_environment()

        hyper_parameters = self.config["hyper_parameters"]
        self.log(f"hyper-parameters: {hyper_parameters}", "TRAIN")

        # Validate spaces required by RASPBERry buffer
        if self.obs_space is None or self.action_space is None:
            raise ValueError("RASPBERry buffer requires obs_space and action_space. Please pass both.")

        # Replay buffer configuration from YAML + inject spaces
        buffer_config = hyper_parameters["replay_buffer_config"]
        buffer_config["obs_space"] = self.obs_space
        buffer_config["action_space"] = self.action_space

        # Ensure numeric types if provided
        if "prioritized_replay_eps" in buffer_config:
            buffer_config["prioritized_replay_eps"] = float(buffer_config["prioritized_replay_eps"])
        if "prioritized_replay_alpha" in buffer_config:
            buffer_config["prioritized_replay_alpha"] = float(buffer_config["prioritized_replay_alpha"])
        if "prioritized_replay_beta" in buffer_config:
            buffer_config["prioritized_replay_beta"] = float(buffer_config["prioritized_replay_beta"])

        # Ensure RASPBERry buffer type
        if buffer_config["type"] == "MultiAgentPrioritizedBlockReplayBuffer":
            buffer_config["type"] = MultiAgentPrioritizedBlockReplayBuffer
        else:
            raise ValueError("Must use MultiAgentPrioritizedBlockReplayBuffer")

        env_config = {"id": self.env_name}

        apex_config = ApexDQNConfig()

        apex_config = apex_config.environment(env=env_id, env_config=env_config)

        apex_config = apex_config.framework(hyper_parameters["framework"])

        training_kwargs = {
            "lr": hyper_parameters["lr"],
            "gamma": hyper_parameters["gamma"],
            "double_q": hyper_parameters["double_q"],
            "dueling": hyper_parameters["dueling"],
            "hiddens": hyper_parameters["hiddens"],
            "target_network_update_freq": hyper_parameters["target_network_update_freq"],
            "replay_buffer_config": buffer_config,
            "train_batch_size": hyper_parameters["train_batch_size"],
            "n_step": hyper_parameters["n_step"],
            "adam_epsilon": hyper_parameters["adam_epsilon"],
            "num_steps_sampled_before_learning_starts": hyper_parameters["num_steps_sampled_before_learning_starts"],
            "noisy": hyper_parameters["noisy"],
            "num_atoms": hyper_parameters["num_atoms"],
        }
        apex_config = apex_config.training(**training_kwargs)

        apex_config = apex_config.reporting(
            min_sample_timesteps_per_iteration=hyper_parameters["min_sample_timesteps_per_iteration"]
        )

        apex_config = apex_config.exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": hyper_parameters["final_epsilon"],
                "epsilon_timesteps": hyper_parameters["epsilon_timesteps"],
            }
        )

        apex_config = apex_config.resources(
            num_gpus=hyper_parameters["num_gpus"],
            num_gpus_per_worker=hyper_parameters.get("num_gpus_per_worker", 0),
            num_cpus_per_worker=hyper_parameters.get("num_cpus_per_worker", 1),
        )

        apex_config = apex_config.rollouts(
            num_rollout_workers=hyper_parameters.get("num_workers", 4),
            num_envs_per_worker=hyper_parameters["num_envs_per_worker"],
            preprocessor_pref=None,
            rollout_fragment_length=hyper_parameters["rollout_fragment_length"],
        )

        # Optimizer configuration (Ape-X specific)
        optimizer_cfg = hyper_parameters.get("optimizer")
        if optimizer_cfg:
            apex_config = apex_config.update_from_dict({"optimizer": optimizer_cfg})

        self.trainer = apex_config.build()
        self.log("✓ Ape-X DQN + RASPBERry ready", "TRAIN")
        return self.trainer

    def _filter_result(self, iteration: int, result: Dict[str, Any]) -> Dict[str, Any]:
        stats = {
            "reward": result.get("episode_reward_mean", "N/A"),
            "timesteps": result.get("timesteps_total", "N/A")
        }

        if self.log_path:
            try:
                result_with_meta = result.copy()
                result_with_meta["iteration"] = iteration
                result_with_meta["timestamp"] = time.time()
                result_with_meta["episodes"] = result.get("episodes_total", "N/A")
                result_with_meta["time_s"] = result.get("time_total_s", "N/A")

                if hasattr(self.trainer, 'local_replay_buffer') and self.trainer.local_replay_buffer is not None:
                    buffer_stats = self.trainer.local_replay_buffer.stats()
                    result_with_meta["buffer_stats"] = buffer_stats
                elif "info" in result and "replay_shard_0" in result["info"]:
                    # Use Ape-X distributed buffer stats
                    result_with_meta["buffer_stats"] = result["info"]["replay_shard_0"]

                processed_data = convert_np_arrays(result_with_meta)

                result_file = self.log_path / f"result_{iteration:06d}.json"
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(processed_data, f, indent=2)

            except Exception as e:
                self.log(f"Failed to save result file: {e}")

        return stats


