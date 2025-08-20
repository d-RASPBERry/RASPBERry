"""
DQN Trainer implementation using RASPBERry.

This module provides DQNRaspberryTrainer that uses RASPBERry's 
RAM-saving prioritized block experience replay buffer.
"""

from typing import Dict, Any
from dynaconf import Dynaconf
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
import json
import time

from .base_trainer import BaseTrainer


class DQNRaspberryTrainer(BaseTrainer):
    """
    DQN trainer using RASPBERry replay buffer.
    
    This trainer creates DQN algorithms with RASPBERry's block-based
    prioritized experience replay buffer for memory efficiency.
    """
    
    def __init__(self, config: Dynaconf, **kwargs):
        """
        Initialize DQN RASPBERry trainer.
        
        Args:
            config: Dynaconf configuration object
            **kwargs: Additional arguments passed to BaseTrainer
        """
        # Force multi-agent-raspberry mode for this trainer
        super().__init__(config, buffer_mode="multi-agent-raspberry", **kwargs)
    
    def setup_environment(self) -> None:
        """Setup and register environment for training."""
        self.log("Setting up environment...", "TRAIN")
        
        # Import env_creator from utils (avoiding circular import)
        from utils import env_creator
        
        # Register environment creator
        register_env("RLEnvironment", env_creator)
        
        self.log("✓ Environment registered", "TRAIN")
    
    def configure_replay_buffer(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure RASPBERry buffer settings for DQN.
        
        Args:
            config_dict: Configuration dictionary containing buffer parameters
            
        Returns:
            Updated configuration dictionary with RASPBERry settings
        """
        raspberry_config = config_dict.copy()
        
        # RASPBERry buffer configuration
        if "replay_buffer_config" not in raspberry_config:
            raspberry_config["replay_buffer_config"] = {}
        
        buffer_config = raspberry_config["replay_buffer_config"]
        
        # RASPBERry specific parameters
        buffer_config.setdefault("type", "MultiAgentPrioritizedBlockReplayBuffer") 
        buffer_config.setdefault("capacity", config_dict.get("replay_capacity", 1000000))
        buffer_config.setdefault("prioritized_replay_alpha", config_dict.get("alpha", 0.6))
        buffer_config.setdefault("prioritized_replay_beta", config_dict.get("beta", 0.4))
        buffer_config.setdefault("prioritized_replay_eps", 1e-6)
        
        # RASPBERry specific parameters
        buffer_config.setdefault("block_size", config_dict.get("block_size", 8))
        buffer_config.setdefault("compress_base", config_dict.get("compress_base", -1))
        buffer_config.setdefault("split_mini_batch", config_dict.get("split_mini_batch", 4))
        buffer_config.setdefault("num_save", config_dict.get("num_save", 100))
        buffer_config.setdefault("store", config_dict.get("store", 2000))
        buffer_config.setdefault("worker_side_prioritization", False)
        buffer_config.setdefault("replay_buffer_shards_colocated_with_driver", True)
        
        # Distributed buffer settings for multi-worker setups
        if config_dict.get("num_workers", 0) > 1:
            raspberry_config.setdefault("optimizer", {})
            raspberry_config["optimizer"]["num_replay_buffer_shards"] = config_dict.get("num_replay_buffer_shards", 1)
        
        self.log(f"Configured RASPBERry buffer: capacity={buffer_config['capacity']}, "
                 f"alpha={buffer_config['prioritized_replay_alpha']}, "
                 f"beta={buffer_config['prioritized_replay_beta']}, "
                 f"block_size={buffer_config['block_size']}", "BUFFER")
        
        return raspberry_config
    
    def create_algorithm(self) -> Any:
        """
        Create DQN algorithm with RASPBERry buffer.
        
        Returns:
            Configured DQN algorithm instance
        """
        self.log("Creating DQN algorithm with RASPBERry...", "TRAIN")
        
        # Setup environment first
        self.setup_environment()
        
        # Get and configure hyperparameters
        hyper_parameters = self.config.hyper_parameters.to_dict()
        configured_params = self.configure_replay_buffer(hyper_parameters)
        
        # Extract environment configuration
        env_config = {
            "id": configured_params.get("env_id", "Atari-PongNoFrameskip-v4")
        }
        
        # Create a temporary environment to get spaces
        from utils import env_creator
        temp_env = env_creator(env_config)
        obs_space = temp_env.observation_space
        action_space = temp_env.action_space
        temp_env.close()
        
        # Update buffer config with actual spaces
        configured_params["replay_buffer_config"]["obs_space"] = obs_space
        configured_params["replay_buffer_config"]["action_space"] = action_space
        
        # RASPBERry train_batch_size adjustment
        block_size = configured_params.get("block_size", 8)
        base_batch_size = configured_params.get("base_train_batch_size", 
                                               configured_params.get("train_batch_size", 32))
        adjusted_batch_size = max(1, int(base_batch_size / block_size))
        
        self.log(f"Adjusting train_batch_size: {base_batch_size} / {block_size} = {adjusted_batch_size}", "BUFFER")
        
        # Create DQN config
        dqn_config = (
            DQNConfig()
            .environment(
                env="RLEnvironment",
                env_config=env_config
            )
            .framework(configured_params.get("framework", "torch"))
            .training(
                # Core DQN parameters
                lr=configured_params.get("lr", 6.25e-5),
                train_batch_size=adjusted_batch_size,
                gamma=configured_params.get("gamma", 0.99),
                double_q=configured_params.get("double_q", True),
                dueling=configured_params.get("dueling", False),
                n_step=configured_params.get("n_step", 1),
                noisy=configured_params.get("noisy", False),
                num_atoms=configured_params.get("num_atoms", 1),
                
                # Network architecture
                hiddens=configured_params.get("hiddens", [512]),
                
                # Adam optimizer
                adam_epsilon=configured_params.get("adam_epsilon", 1.5e-4),
                
                # Target network update
                target_network_update_freq=configured_params.get("target_network_update_freq", 8000),
                
                # RASPBERry buffer configuration
                replay_buffer_config=configured_params.get("replay_buffer_config", {}),
                
                # Learning schedule
                num_steps_sampled_before_learning_starts=configured_params.get(
                    "num_steps_sampled_before_learning_starts", 10000
                )
            )
            .rollouts(
                num_rollout_workers=configured_params.get("num_workers", 1),
                num_envs_per_worker=configured_params.get("num_envs_per_worker", 1),
                rollout_fragment_length=configured_params.get("rollout_fragment_length", 4),
            )
            .exploration(
                exploration_config={
                    "type": "EpsilonGreedy",
                    "initial_epsilon": 1.0,
                    "final_epsilon": configured_params.get("final_epsilon", 0.01),
                    "epsilon_timesteps": configured_params.get("epsilon_timesteps", 200000),
                }
            )
            .resources(
                num_gpus=configured_params.get("num_gpus", 1),
                num_cpus_per_worker=1,
            )
            .reporting(
                min_sample_timesteps_per_iteration=configured_params.get(
                    "min_sample_timesteps_per_iteration", 10000
                )
            )
        )
        
        # Build the algorithm
        algorithm = dqn_config.build()
        
        self.log(f"✓ DQN algorithm created with RASPBERry buffer", "TRAIN")
        
        return algorithm

    def _save_training_result(self, iteration: int, result: Dict[str, Any]) -> Dict[str, Any]:
        """Save RASPBERry training result and return stats for logging."""
        # Extract stats first
        stats = {
            "reward": result.get("episode_reward_mean", "N/A"),
            "timesteps": result.get("timesteps_total", "N/A"),
            "episodes": result.get("episodes_total", "N/A"),
            "time_s": result.get("time_total_s", "N/A")
        }
        
        # Save to file if path exists
        if self.metrics_path:
            try:
                from utils import convert_np_arrays
                
                result_with_meta = result.copy()
                result_with_meta["iteration"] = iteration
                result_with_meta["timestamp"] = time.time()
                
                # Add RASPBERry buffer stats - this is the key info!
                if hasattr(self.trainer, 'local_replay_buffer'):
                    buffer_stats = self.trainer.local_replay_buffer.stats()
                    result_with_meta["buffer_stats"] = buffer_stats
                
                processed_data = convert_np_arrays(result_with_meta)
                
                file_path = self.metrics_path / f"iter_{iteration:06d}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(processed_data, f, indent=2)
                    
            except Exception as e:
                self.log(f"Failed to save iteration {iteration} metrics: {e}", "WARNING")
        
        return stats