"""
DQN Trainer implementation using RASPBERry.

This module provides DQNRaspberryTrainer that uses RASPBERry's 
RAM-saving prioritized block experience replay buffer.
"""

from typing import Dict, Any, Optional
from ray.rllib.algorithms.dqn import DQNConfig
from utils import convert_np_arrays
from replay_buffer.d_raspberry import MultiAgentPrioritizedBlockReplayBuffer
from gymnasium.spaces import Space

import time
import json

from .base_trainer import BaseTrainer


class DQNRaspberryTrainer(BaseTrainer):
    """
    DQN trainer using RASPBERry replay buffer.
    
    This trainer creates DQN algorithms with RASPBERry's block-based
    prioritized experience replay buffer for memory efficiency.
    """

    def __init__(self,
                 config: str,
                 env_name: str,
                 run_name: str,
                 log_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 obs_space: Space = None,
                 action_space: Space = None,):
        """
        Initialize DQN RASPBERry trainer.
        
        Args:
            config: Path to YAML configuration file
            env_name: Environment name for logging and run identification
            log_path: Optional root directory to store logs
            checkpoint_path: Optional root directory to store checkpoints
        """
        super().__init__(config, env_name, run_name, log_path, checkpoint_path)
        self.obs_space = obs_space
        self.action_space = action_space

    def init_algorithm(self) -> Any:
        """
        Create DQN algorithm with RASPBERry buffer.
        
        Returns:
            Configured DQN algorithm instance
        """
        self.log("Creating DQN algorithm with RASPBERry...", "TRAIN")

        # Setup environment first
        env_id = self.setup_environment()

        # Get and configure hyperparameters
        hyper_parameters = self.config["hyper_parameters"]
        self.log(f"hyper-parameters: {hyper_parameters}", "TRAIN")
        # Use replay buffer configuration directly from YAML
        buffer_config = hyper_parameters["replay_buffer_config"]
        buffer_config["action_space"] = self.action_space
        buffer_config["obs_space"] = self.obs_space
        
        # 确保关键参数的类型正确
        if "prioritized_replay_eps" in buffer_config:
            buffer_config["prioritized_replay_eps"] = float(buffer_config["prioritized_replay_eps"])
        if "prioritized_replay_alpha" in buffer_config:
            buffer_config["prioritized_replay_alpha"] = float(buffer_config["prioritized_replay_alpha"])
        if "prioritized_replay_beta" in buffer_config:
            buffer_config["prioritized_replay_beta"] = float(buffer_config["prioritized_replay_beta"])
            
        self.log(f"Using buffer config: {buffer_config}", "BUFFER")

        # Ensure buffer_config type is object rather than string
        if buffer_config["type"] == "MultiAgentPrioritizedBlockReplayBuffer":
            buffer_config["type"] = MultiAgentPrioritizedBlockReplayBuffer
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
            env_config=env_config
        )

        # Framework settings
        dqn_config = dqn_config.framework(hyper_parameters["framework"])

        # Training parameter settings
        dqn_config = dqn_config.training(
            lr=hyper_parameters["lr"],
            gamma=hyper_parameters["gamma"],
            double_q=hyper_parameters["double_q"],
            dueling=hyper_parameters["dueling"],
            hiddens=hyper_parameters["hiddens"],
            target_network_update_freq=hyper_parameters["target_network_update_freq"],
            replay_buffer_config=buffer_config
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
            num_gpus=hyper_parameters["num_gpus"]
        )

        # Build algorithm
        self.trainer = dqn_config.build()
        self.log(f"✓ DQN algorithm with RASPBERRY created successfully", "TRAIN")

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
