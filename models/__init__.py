"""
Custom neural network models for RLlib algorithms.

This module provides custom model architectures optimized for different
observation types (images, vectors) and algorithms (SAC, DQN, etc.).
"""

import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class SACImageEncoder(TorchModelV2, nn.Module):
    """
    CNN encoder for SAC with image observations.
    
    Processes 84x84x3 RGB images through convolutional layers.
    Nature CNN architecture adapted for SAC.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Extract image dimensions
        h, w, c = obs_space.shape
        
        # Nature CNN architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        
        # Calculate conv output size
        conv_out_h = ((h - 8) // 4 + 1 - 4) // 2 + 1 - 3 + 1
        conv_out_w = ((w - 8) // 4 + 1 - 4) // 2 + 1 - 3 + 1
        conv_out_size = conv_out_h * conv_out_w * 64
        
        # Feature extraction
        self.feature_dim = model_config.get("custom_model_config", {}).get("feature_dim", 512)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, self.feature_dim),
            nn.ReLU(),
        )
        
        self.output_layer = nn.Linear(self.feature_dim, num_outputs)
        self._features = None
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        obs = obs.permute(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)
        obs = obs / 255.0  # Normalize
        
        x = self.conv_layers(obs)
        x = x.reshape(x.size(0), -1)
        self._features = self.fc(x)
        output = self.output_layer(self._features)
        
        return output, state
    
    @override(TorchModelV2)
    def value_function(self):
        return torch.zeros(self._features.shape[0])


class SACLightweightCNN(TorchModelV2, nn.Module):
    """
    Lightweight CNN encoder for smaller image observations.
    
    Suitable for 64x64 or 80x80 images. Faster training than Nature CNN.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        h, w, c = obs_space.shape
        
        # Lightweight conv architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),  # H/2 x W/2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # H/4 x W/4
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # H/8 x W/8
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 4))  # Fixed 4x4 output
        )
        
        # Feature extraction (4x4x128 = 2048)
        self.feature_dim = model_config.get("custom_model_config", {}).get("feature_dim", 256)
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 128, self.feature_dim),
            nn.ReLU(),
        )
        
        self.output_layer = nn.Linear(self.feature_dim, num_outputs)
        self._features = None
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        obs = obs.permute(0, 3, 1, 2)
        obs = obs / 255.0
        
        x = self.conv_layers(obs)
        x = x.reshape(x.size(0), -1)
        self._features = self.fc(x)
        output = self.output_layer(self._features)
        
        return output, state
    
    @override(TorchModelV2)
    def value_function(self):
        return torch.zeros(self._features.shape[0])


__all__ = [
    "SACImageEncoder",
    "SACLightweightCNN",
]
