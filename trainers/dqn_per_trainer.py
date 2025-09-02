"""
DQN Trainer implementation using Ray RLlib.

This module provides DQNTrainer that uses Ray's native PER replay buffer.
"""

from typing import Dict, Any, Optional
from ray.rllib.algorithms.dqn import DQNConfig
from utils import convert_np_arrays

from ray.rllib.utils.replay_buffers import MultiAgentPrioritizedReplayBuffer
import time
import json

from .base_trainer import BaseTrainer


class DQNTrainer(BaseTrainer):
    """
    DQN trainer using Ray's native MultiAgent PER.
    
    This trainer creates DQN algorithms with Ray's built-in prioritized
    experience replay buffer.
    """

    def __init__(self,
                 config: str,
                 env_name: str,
                 run_name: str,
                 log_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 mlflow: str = None):
        """
        Initialize DQN trainer with Ray's native PER buffer.

        Args:
            config: Path to YAML configuration file
            env_name: Environment name for logging and run identification
            log_path: Optional root directory to store logs
            checkpoint_path: Optional root directory to store checkpoints
        """
        super().__init__(config, env_name, run_name, log_path, checkpoint_path, use_mlflow=mlflow)

    def init_algorithm(self) -> Any:
        """
        Create and return RL algorithm instance.
        
        Returns:
            Configured DQN algorithm instance
        """
        # Setup environment
        env_id = self.setup_environment()

        # Get hyperparameters
        hyper_parameters = self.config["hyper_parameters"]
        # Use replay buffer configuration directly from YAML
        buffer_config = hyper_parameters["replay_buffer_config"]

        # Ensure critical parameters have correct types
        if "prioritized_replay_eps" in buffer_config:
            buffer_config["prioritized_replay_eps"] = float(buffer_config["prioritized_replay_eps"])
        if "prioritized_replay_alpha" in buffer_config:
            buffer_config["prioritized_replay_alpha"] = float(buffer_config["prioritized_replay_alpha"])
        if "prioritized_replay_beta" in buffer_config:
            buffer_config["prioritized_replay_beta"] = float(buffer_config["prioritized_replay_beta"])

        self.log(f"Using buffer config: {buffer_config}", "BUFFER")
        # Ensure buffer_config type is object rather than string
        if buffer_config["type"] == "MultiAgentPrioritizedReplayBuffer":
            buffer_config["type"] = MultiAgentPrioritizedReplayBuffer
        else:
            raise ValueError()

        # Environment configuration
        env_config = {
            "id": self.env_name
        }

        # Create DQN configuration
        dqn_config = DQNConfig()

        # Environment settings
        dqn_config = dqn_config.environment(
            env=env_id,
            env_config=env_config,
        )

        # Framework settings
        dqn_config = dqn_config.framework(hyper_parameters["framework"])

        # Training parameter settings
        training_kwargs = {
            "lr": hyper_parameters["lr"],
            "gamma": hyper_parameters["gamma"],
            "double_q": hyper_parameters["double_q"],
            "dueling": hyper_parameters["dueling"],
            "hiddens": hyper_parameters["hiddens"],
            "target_network_update_freq": hyper_parameters["target_network_update_freq"],
            "replay_buffer_config": buffer_config,
            "train_batch_size": hyper_parameters["train_batch_size"]
        }
        dqn_config = dqn_config.training(**training_kwargs)

        # Reporting settings
        dqn_config = dqn_config.reporting(
            min_sample_timesteps_per_iteration=hyper_parameters["min_sample_timesteps_per_iteration"]
        )

        # Exploration settings
        dqn_config = dqn_config.exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": hyper_parameters["final_epsilon"],
                "epsilon_timesteps": hyper_parameters["epsilon_timesteps"],
            }
        )

        # Resource configuration
        dqn_config = dqn_config.resources(
            num_gpus=hyper_parameters["num_gpus"],
            num_gpus_per_worker=hyper_parameters["num_gpus_per_worker"],
            num_cpus_per_worker=hyper_parameters["num_cpus_per_worker"],
        )

        # Rollouts configuration
        dqn_config = dqn_config.rollouts(
            num_envs_per_worker=hyper_parameters["num_envs_per_worker"],
            preprocessor_pref=None,
            rollout_fragment_length=hyper_parameters["rollout_fragment_length"]
        )

        # Build algorithm
        self.trainer = dqn_config.build()
        self.log("✓ DQN ready", "TRAIN")

        return self.trainer

    def _filter_result(self, iteration: int, result: Dict[str, Any]) -> Dict[str, Any]:
        """Filter training results and return statistics for logging."""
        # Extract key statistics for BaseTrainer logging
        stats = {
            "reward": result.get("episode_reward_mean", "N/A"),
            "timesteps": result.get("timesteps_total", "N/A")
        }

        # If log path exists, save complete results to file
        if self.log_path:
            try:
                result_with_meta = result.copy()
                result_with_meta["iteration"] = iteration
                result_with_meta["timestamp"] = time.time()
                result_with_meta["episodes"] = result.get("episodes_total", "N/A")
                result_with_meta["time_s"] = result.get("time_total_s", "N/A")

                # Add buffer statistics (if available)
                if hasattr(self.trainer, 'local_replay_buffer'):
                    buffer_stats = self.trainer.local_replay_buffer.stats()
                    result_with_meta["buffer_stats"] = buffer_stats

                processed_data = convert_np_arrays(result_with_meta)

                result_file = self.log_path / f"result_{iteration:06d}.json"
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(processed_data, f, indent=2)

            except Exception as e:
                self.log(f"Failed to save result file: {e}")

        return stats
