"""
Custom neural network models for RLlib algorithms.

This module provides custom model architectures optimized for different
observation types (images, vectors) and algorithms (SAC, DQN, etc.).
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
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
    It implements custom Policy and Q networks to handle Image inputs correctly
    (avoiding VisionNetwork + Tuple issues).
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
                 policy_model_config=None, q_model_config=None, twin_q=True,
                 initial_alpha=1.0, target_entropy=None, **kwargs):
        # Pop custom_model_config keys that are not used by parent
        kwargs.pop("feature_dim", None)

        # Initialize SACTorchModel parent with required SAC parameters
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
            policy_model_config=policy_model_config or {},
            q_model_config=q_model_config or {},
            twin_q=twin_q,
            initial_alpha=initial_alpha,
            target_entropy=target_entropy,
        )
        
    def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
        """Build the policy network with CNN feature extractor."""
        return PolicyCNN(obs_space, self.action_space, num_outputs, policy_model_config, name)
    
    def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
        """Build the Q-network with CNN feature extractor."""
        # Handle input space for Q-network
        # For image observations, we use Tuple space (obs, action)
        # We explicitly set concat_obs_and_actions=False so SACTorchModel
        # passes the structured input to our Q-net.
        
        orig_space = getattr(obs_space, "original_space", obs_space)
        if isinstance(orig_space, Box) and len(orig_space.shape) == 1:
            # Vector observation: concatenate obs and action (Legacy behavior, not used for CarRacing)
            input_space = Box(
                float("-inf"),
                float("inf"),
                shape=(orig_space.shape[0] + action_space.shape[0],),
            )
            self.concat_obs_and_actions = True
        else:
            # Image observation: use Tuple(obs, action)
            input_space = gym.spaces.Tuple([orig_space, action_space])
            self.concat_obs_and_actions = False
            
        return QCNN(input_space, action_space, num_outputs, q_model_config, name)


class PolicyCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Hardcoded 84x84 Nature-CNN variant
        # 84x84 -> 20x20 -> 9x9 -> 7x7
        self.convs = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # 64 * 7 * 7 = 3136
        self.feature_dim = 3136
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        # Ensure [B, C, H, W]
        if x.shape[-1] == 4:
            x = x.permute(0, 3, 1, 2)
        
        features = self.convs(x)
        logits = self.fc(features)
        return logits, state


class QCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Same CNN backbone
        self.convs = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.feature_dim = 3136
        
        # Q-network takes features + action
        # Action dim calculation
        if hasattr(action_space, 'shape'):
            action_dim = int(np.prod(action_space.shape))
        else:
            action_dim = 1
            
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Single Q-value
        )

    def forward(self, input_dict, state, seq_lens):
        # Handle Tuple(obs, action) input from SAC
        # input_dict["obs"] is a tuple of tensors
        # Note: RLlib might wrap this differently depending on version.
        # But for Tuple space, input_dict["obs"] is usually a tuple/list of tensors.
        
        obs_input = input_dict["obs"]
        if isinstance(obs_input, (tuple, list)):
            obs = obs_input[0]
            action = obs_input[1]
        else:
            # Fallback if it's already flattened or something (unlikely with concat=False)
            raise ValueError(f"QCNN expected Tuple input, got {type(obs_input)}")
        
        x = obs.float()
        if x.shape[-1] == 4:
            x = x.permute(0, 3, 1, 2)
            
        features = self.convs(x)
        
        # Concatenate features and action
        q_input = torch.cat([features, action], dim=1)
        
        q_val = self.fc(q_input)
        return q_val, state


__all__ = [
    "SACLightweightCNN",
]
