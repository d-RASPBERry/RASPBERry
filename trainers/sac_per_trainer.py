"""
SAC Trainer implementation using Ray RLlib.

This module provides SACTrainer that uses Ray's native PER replay buffer.
"""

from typing import Dict, Any, Optional
from ray.rllib.algorithms.sac import SACConfig
from utils import convert_np_arrays

from ray.rllib.utils.replay_buffers import MultiAgentPrioritizedReplayBuffer
import time
import json

from .base_trainer import BaseTrainer


class SACTrainer(BaseTrainer):
    """
    SAC trainer using Ray's native MultiAgent PER.
    
    This trainer creates SAC algorithms with Ray's built-in prioritized
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
        Initialize SAC trainer with Ray's native PER buffer.

        Args:
            config: Path to YAML configuration file
            env_name: Environment name for logging and run identification
            log_path: Optional root directory to store logs
            checkpoint_path: Optional root directory to store checkpoints
        """
        super().__init__(config, env_name, run_name, log_path, checkpoint_path, use_mlflow=mlflow)

    def init_algorithm(self) -> Any:
        """
        Create and return the RL algorithm instance.

        Returns:
            Configured SAC algorithm instance.
        """
        self.log("Creating SAC algorithm...", "TRAIN")

        # Set up environment
        env_id = self.setup_environment()

        # Retrieve hyperparameters
        hyper_parameters = self.config["hyper_parameters"]
        self.log(f"hyper-parameters: {hyper_parameters}", "TRAIN")
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

        # Ensure `type` in buffer_config is a class object, not a string
        if buffer_config["type"] == "MultiAgentPrioritizedReplayBuffer":
            buffer_config["type"] = MultiAgentPrioritizedReplayBuffer
        else:
            raise ValueError()

        # Environment configuration
        env_config = {
            "id": self.env_name
        }

        # Create SAC configuration
        sac_config = SACConfig()

        # Environment settings
        sac_config = sac_config.environment(
            env=env_id,
            env_config=env_config
        )

        # Framework settings
        sac_config = sac_config.framework(hyper_parameters["framework"])

        # Training parameter settings
        sac_config = sac_config.training(
            lr=hyper_parameters["lr"],
            gamma=hyper_parameters["gamma"],
            tau=hyper_parameters.get("tau", 0.005),
            target_entropy=hyper_parameters.get("target_entropy", "auto"),
            initial_alpha=hyper_parameters.get("initial_alpha", 1.0),
            n_step=hyper_parameters.get("n_step", 1),
            train_batch_size=hyper_parameters.get("train_batch_size", 256),
            replay_buffer_config=buffer_config
        )

        # Resource configuration
        sac_config = sac_config.resources(
            num_gpus=hyper_parameters["num_gpus"]
        )

        # SAC-specific rollout configuration
        sac_config = sac_config.rollouts(
            num_rollout_workers=hyper_parameters.get("num_workers", 0),
            rollout_fragment_length=hyper_parameters.get("rollout_fragment_length", 1),
        )

        # Build algorithm
        self.trainer = sac_config.build()
        self.log(f"✓ SAC algorithm created successfully", "TRAIN")

        return self.trainer

    def _filter_result(self, iteration: int, result: Dict[str, Any]) -> Dict[str, Any]:
        """Filter training results and return stats for logging."""
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

                # Add replay buffer statistics if available
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
