"""
DQN Trainer implementation using RASPBERry.

This module provides DQNRaspberryTrainer that uses RASPBERry's 
RAM-saving prioritized block experience replay buffer.
"""

from typing import Any, Optional
from ray.rllib.algorithms.dqn import DQNConfig
from replay_buffer.d_raspberry import MultiAgentPrioritizedBlockReplayBuffer
from gymnasium.spaces import Space

from .dqn_per_trainer import DQNTrainer


class DQNRaspberryTrainer(DQNTrainer):
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
                 mlflow: str = None,
                 obs_space: Space = None,
                 action_space: Space = None, ):
        """
        Initialize DQN RASPBERry trainer.
        
        Args:
            config: Path to YAML configuration file
            env_name: Environment name for logging and run identification
            log_path: Optional root directory to store logs
            checkpoint_path: Optional root directory to store checkpoints
        """
        super().__init__(config, env_name, run_name, log_path, checkpoint_path, mlflow)
        self.obs_space = obs_space
        self.action_space = action_space

    def init_algorithm(self) -> Any:
        """
        Create DQN algorithm with RASPBERry buffer.
        
        Returns:
            Configured DQN algorithm instance
        """
        # Log and setup environment
        self.log("Creating DQN+RASPBERry algorithm...", "TRAIN")
        # Setup environment first
        env_id = self.setup_environment()

        # Validate spaces required by RASPBERry buffer
        if self.obs_space is None or self.action_space is None:
            raise ValueError("RASPBERry buffer requires obs_space and action_space. Please pass both.")

        # Get and configure hyperparameters
        hyper_parameters = self.config["hyper_parameters"]
        # Use replay buffer configuration directly from YAML
        buffer_config = hyper_parameters["replay_buffer_config"]
        buffer_config["action_space"] = self.action_space
        buffer_config["obs_space"] = self.obs_space

        # Ensure critical parameters have correct types
        if "prioritized_replay_eps" in buffer_config:
            buffer_config["prioritized_replay_eps"] = float(buffer_config["prioritized_replay_eps"])
        if "prioritized_replay_alpha" in buffer_config:
            buffer_config["prioritized_replay_alpha"] = float(buffer_config["prioritized_replay_alpha"])
        if "prioritized_replay_beta" in buffer_config:
            buffer_config["prioritized_replay_beta"] = float(buffer_config["prioritized_replay_beta"])
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

        # Training parameter settings (allow batch sizes from YAML)
        training_kwargs = {
            "lr": hyper_parameters["lr"],
            "gamma": hyper_parameters["gamma"],
            "double_q": hyper_parameters["double_q"],
            "dueling": hyper_parameters["dueling"],
            "hiddens": hyper_parameters["hiddens"],
            "target_network_update_freq": hyper_parameters["target_network_update_freq"],
            "replay_buffer_config": buffer_config,
            "train_batch_size": hyper_parameters["train_batch_size"],
            # Additional fields from YAML
            "n_step": hyper_parameters["n_step"],
            "adam_epsilon": hyper_parameters["adam_epsilon"],
            "num_steps_sampled_before_learning_starts": hyper_parameters["num_steps_sampled_before_learning_starts"],
            "noisy": hyper_parameters["noisy"],
            "num_atoms": hyper_parameters["num_atoms"],
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
        self.log("✓ DQN+RASPBERry ready", "TRAIN")

        return self.trainer
