"""
Custom neural network models for RLlib algorithms.

This module provides custom model architectures optimized for different
observation types (images, vectors) and algorithms (SAC, DQN, etc.).
"""

import torch
import torch.nn as nn
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.annotations import override


class SACLightweightCNN(SACTorchModel):
    """
    CNN encoder for image observations with SAC.
    
    - CNN backbone: Hardcoded Nature-CNN (32-64-64 filters)
    - Dense layers: Configurable via policy_model_config / q_model_config
    
    This model properly inherits from SACTorchModel to work with SAC algorithm.
    """
    
    # Hardcoded Nature-CNN architecture
    CONV_FILTERS = [
        [32, [8, 8], 4],   # 32 filters, 8x8 kernel, stride 4
        [64, [4, 4], 2],   # 64 filters, 4x4 kernel, stride 2
        [64, [3, 3], 1],   # 64 filters, 3x3 kernel, stride 1
    ]
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
                 policy_model_config=None, q_model_config=None, **kwargs):
        # Pop any custom_model_config keys to avoid passing unsupported args to parent
        kwargs.pop("feature_dim", None)
        kwargs.pop("twin_q", None)
        kwargs.pop("initial_alpha", None)
        kwargs.pop("target_entropy", None)

        # Initialize SACTorchModel parent
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
            policy_model_config=policy_model_config or {},
            q_model_config=q_model_config or {},
        )
        
    def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
        """Build the policy network with CNN feature extractor."""
        # Merge hardcoded CNN with configurable dense layers
        config = {
            # Hardcoded CNN backbone
            "conv_filters": self.CONV_FILTERS,
            "conv_activation": "relu",
            # Dense layers from config (with defaults)
            "post_fcnet_hiddens": policy_model_config.get("post_fcnet_hiddens", [256, 256]),
            "post_fcnet_activation": policy_model_config.get("post_fcnet_activation", "relu"),
            "fcnet_hiddens": policy_model_config.get("fcnet_hiddens", [256, 256]),
            "fcnet_activation": policy_model_config.get("fcnet_activation", "relu"),
        }
        
        model = ModelCatalog.get_model_v2(
            obs_space,
            self.action_space,
            num_outputs,
            config,
            framework="torch",
            name=name,
        )
        return model
    
    def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
        """Build the Q-network with CNN feature extractor."""
        # Merge hardcoded CNN with configurable dense layers
        config = {
            # Hardcoded CNN backbone
            "conv_filters": self.CONV_FILTERS,
            "conv_activation": "relu",
            # Dense layers from config (with defaults)
            "post_fcnet_hiddens": q_model_config.get("post_fcnet_hiddens", [256, 256]),
            "post_fcnet_activation": q_model_config.get("post_fcnet_activation", "relu"),
            "fcnet_hiddens": q_model_config.get("fcnet_hiddens", [256, 256]),
            "fcnet_activation": q_model_config.get("fcnet_activation", "relu"),
        }
        
        # Handle input space for Q-network (obs + action concatenation)
        self.concat_obs_and_actions = False
        if self.discrete:
            input_space = obs_space
        else:
            from gymnasium.spaces import Box
            orig_space = getattr(obs_space, "original_space", obs_space)
            # For image observations, use Tuple space (obs, action)
            if isinstance(orig_space, Box) and len(orig_space.shape) == 1:
                # Vector observation: concatenate obs and action
                input_space = Box(
                    float("-inf"),
                    float("inf"),
                    shape=(orig_space.shape[0] + action_space.shape[0],),
                )
                self.concat_obs_and_actions = True
            else:
                # Image observation: use Tuple(obs, action)
                import gymnasium as gym
                input_space = gym.spaces.Tuple([orig_space, action_space])
        
        model = ModelCatalog.get_model_v2(
            input_space,
            action_space,
            num_outputs,
            config,
            framework="torch",
            name=name,
        )
        return model


__all__ = [
    "SACLightweightCNN",
]
