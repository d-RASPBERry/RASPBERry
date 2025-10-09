"""
SAC Trainer implementation with RASPBERry replay buffer.

This module provides SAC trainer that uses the custom RASPBERry 
(MultiAgentPrioritizedBlockReplayBuffer) for memory-efficient training.
"""

from typing import Dict, Any, Optional
from gymnasium.spaces import Space
from ray.rllib.algorithms.sac import SACConfig
from replay_buffer.d_raspberry import MultiAgentPrioritizedBlockReplayBuffer

from .sac_per_trainer import SACTrainer as SACPERTrainer


class SACRaspberryTrainer(SACPERTrainer):
    """
    SAC trainer using RASPBERry MultiAgent Block PER buffer.
    
    This trainer creates SAC algorithms with the custom memory-efficient
    RASPBERry replay buffer that uses block compression.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 env_name: str,
                 run_name: str,
                 log_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 mlflow_cfg: Optional[Dict[str, Any]] = None,
                 obs_space: Space = None,
                 action_space: Space = None):
        """
        Initialize SAC trainer with RASPBERry buffer.

        Args:
            config: Path to YAML configuration file
            env_name: Environment name for logging and run identification
            log_path: Optional root directory to store logs
            checkpoint_path: Optional root directory to store checkpoints
        """
        super().__init__(config, env_name, run_name, log_path, checkpoint_path, mlflow_cfg=mlflow_cfg)
        self.obs_space = obs_space
        self.action_space = action_space

    def init_algorithm(self) -> Any:
        """
        Create and return the RL algorithm instance.

        Returns:
            Configured SAC algorithm instance.
        """
        self.log("Creating SAC RASPBERry algorithm...", "TRAIN")

        # Set up environment
        env_id = self.setup_environment()

        # Retrieve hyperparameters
        hyper_parameters = self.config["hyper_parameters"]
        self.log(f"hyper-parameters: {hyper_parameters}", "TRAIN")
        # Use replay buffer configuration directly from YAML
        buffer_config = hyper_parameters["replay_buffer_config"]
        # Ensure numeric types if provided
        if "prioritized_replay_eps" in buffer_config:
            buffer_config["prioritized_replay_eps"] = float(buffer_config["prioritized_replay_eps"])
        if "prioritized_replay_alpha" in buffer_config:
            buffer_config["prioritized_replay_alpha"] = float(buffer_config["prioritized_replay_alpha"])
        if "prioritized_replay_beta" in buffer_config:
            buffer_config["prioritized_replay_beta"] = float(buffer_config["prioritized_replay_beta"])
        # Inject spaces for custom buffer
        if self.obs_space is not None and self.action_space is not None:
            buffer_config["obs_space"] = self.obs_space
            buffer_config["action_space"] = self.action_space
        self.log(f"Using RASPBERry buffer config: {buffer_config}", "BUFFER")

        # Ensure `type` in buffer_config is a class object, not a string
        if buffer_config["type"] == "MultiAgentPrioritizedBlockReplayBuffer":
            buffer_config["type"] = MultiAgentPrioritizedBlockReplayBuffer
        else:
            raise ValueError("Must use MultiAgentPrioritizedBlockReplayBuffer")

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

        # Training parameter settings (allow batch sizes from YAML)
        sac_config = sac_config.training(
            lr=hyper_parameters["lr"],
            gamma=hyper_parameters["gamma"],
            tau=hyper_parameters.get("tau", 0.005),
            target_entropy=hyper_parameters.get("target_entropy", "auto"),
            initial_alpha=hyper_parameters.get("initial_alpha", 1.0),
            n_step=hyper_parameters.get("n_step", 1),
            train_batch_size=hyper_parameters.get("train_batch_size", 256),
            replay_buffer_config=buffer_config,
        )

        # Resource configuration
        sac_config = sac_config.resources(
            num_gpus=hyper_parameters["num_gpus"],
            num_gpus_per_worker=hyper_parameters.get("num_gpus_per_worker", 0),
            num_cpus_per_worker=hyper_parameters.get("num_cpus_per_worker", 1),
        )

        # SAC-specific rollout configuration
        sac_config = sac_config.rollouts(
            num_rollout_workers=hyper_parameters.get("num_workers", 0),
            num_envs_per_worker=hyper_parameters.get("num_envs_per_worker", 1),
            rollout_fragment_length=hyper_parameters.get("rollout_fragment_length", 1),
        )

        # Build algorithm
        self.trainer = sac_config.build()
        self.log(f"✓ SAC RASPBERry algorithm created successfully", "TRAIN")

        return self.trainer
