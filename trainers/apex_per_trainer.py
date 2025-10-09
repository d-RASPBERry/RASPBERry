"""
Ape-X DQN Trainer implementation using Ray RLlib.

This module provides ApexDQNTrainer that uses Ray's native distributed
prioritized experience replay (PER) with Ape-X.
"""

from typing import Dict, Any, Optional
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig
from ray.rllib.utils.replay_buffers import MultiAgentPrioritizedReplayBuffer

from .base_trainer import BaseTrainer


class ApexDQNTrainer(BaseTrainer):
    """
    Ape-X DQN trainer using Ray's native MultiAgent PER in distributed mode.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 env_name: str,
                 run_name: str,
                 log_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 mlflow_cfg: Optional[Dict[str, Any]] = None):
        """
        Initialize Ape-X DQN trainer with Ray's native PER buffer.

        Args:
            config: Path to YAML configuration file
            env_name: Environment name for logging and run identification
            log_path: Optional root directory to store logs
            checkpoint_path: Optional root directory to store checkpoints
        """
        super().__init__(config, env_name, run_name, log_path, checkpoint_path, mlflow_cfg=mlflow_cfg)

    def init_algorithm(self) -> Any:
        """
        Create and return RL algorithm instance.

        Returns:
            Configured Ape-X DQN algorithm instance
        """
        # Log and setup environment
        self.log("Creating Ape-X DQN algorithm...", "TRAIN")
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
            raise ValueError("Must use MultiAgentPrioritizedReplayBuffer")

        # Environment configuration
        env_config = {
            "id": self.env_name
        }

        # Create Ape-X DQN configuration
        apex_config = ApexDQNConfig()

        # Environment settings
        apex_config = apex_config.environment(
            env=env_id,
            env_config=env_config,
        )

        # Framework settings
        apex_config = apex_config.framework(hyper_parameters["framework"])

        # Training parameter settings
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

        # Reporting settings
        apex_config = apex_config.reporting(
            min_sample_timesteps_per_iteration=hyper_parameters["min_sample_timesteps_per_iteration"]
        )

        # Exploration settings
        apex_config = apex_config.exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": hyper_parameters["final_epsilon"],
                "epsilon_timesteps": hyper_parameters["epsilon_timesteps"],
            }
        )

        # Resource configuration
        apex_config = apex_config.resources(
            num_gpus=hyper_parameters["num_gpus"],
            num_gpus_per_worker=hyper_parameters.get("num_gpus_per_worker", 0),
            num_cpus_per_worker=hyper_parameters.get("num_cpus_per_worker", 1),
        )

        # Rollouts configuration: Ape-X requires rollout workers > 0
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

        # Build algorithm
        self.trainer = apex_config.build()
        self.log("✓ Ape-X DQN ready", "TRAIN")

        return self.trainer


