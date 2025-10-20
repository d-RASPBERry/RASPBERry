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
    Lightweight CNN encoder for smaller image observations with SAC.
    
    Suitable for 64x64 or 80x80 images. Faster training than Nature CNN.
    
    This model properly inherits from SACTorchModel to work with SAC algorithm.
    It provides custom CNN feature extraction for both policy and Q networks.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
                 policy_model_config=None, q_model_config=None, **kwargs):
        # Initialize SACTorchModel parent
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
            policy_model_config=policy_model_config or {},
            q_model_config=q_model_config or {},
            **kwargs
        )
        
    def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
        """Build the policy network with CNN feature extractor."""
        # Use ModelCatalog to build a CNN model for the policy
        policy_model_config = policy_model_config.copy()
        if "conv_filters" not in policy_model_config:
            # Use lightweight CNN architecture
            policy_model_config["conv_filters"] = [
                [32, [3, 3], 2],   # 32 filters, 3x3 kernel, stride 2
                [64, [3, 3], 2],   # 64 filters, 3x3 kernel, stride 2
                [128, [3, 3], 2],  # 128 filters, 3x3 kernel, stride 2
            ]
            policy_model_config["conv_activation"] = "relu"
            policy_model_config["post_fcnet_hiddens"] = [256, 256]
            policy_model_config["post_fcnet_activation"] = "relu"
        
        model = ModelCatalog.get_model_v2(
            obs_space,
            self.action_space,
            num_outputs,
            policy_model_config,
            framework="torch",
            name=name,
        )
        return model
    
    def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
        """Build the Q-network with CNN feature extractor."""
        # Use ModelCatalog to build a CNN model for Q-network
        q_model_config = q_model_config.copy()
        if "conv_filters" not in q_model_config:
            # Use same lightweight CNN architecture as policy
            q_model_config["conv_filters"] = [
                [32, [3, 3], 2],   # 32 filters, 3x3 kernel, stride 2
                [64, [3, 3], 2],   # 64 filters, 3x3 kernel, stride 2
                [128, [3, 3], 2],  # 128 filters, 3x3 kernel, stride 2
            ]
            q_model_config["conv_activation"] = "relu"
            q_model_config["post_fcnet_hiddens"] = [256, 256]
            q_model_config["post_fcnet_activation"] = "relu"
        
        # Handle input space for Q-network (obs + action concatenation)
        self.concat_obs_and_actions = False
        if self.discrete:
            input_space = obs_space
        else:
            from gymnasium.spaces import Box, Tuple as TupleSpace
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
            q_model_config,
            framework="torch",
            name=name,
        )
        return model


__all__ = [
    "SACLightweightCNN",
]
