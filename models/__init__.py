"""
Custom neural network models for RLlib algorithms.

This module provides custom model architectures optimized for different
observation types (images, vectors) and algorithms (SAC, DQN, etc.).
"""

import math
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


def _ensure_int_outputs(num_outputs):
    """将 RLlib 传入的输出维度规范化为整数。"""
    if isinstance(num_outputs, (tuple, list)):
        if not num_outputs:
            raise ValueError("num_outputs 维度为空，无法构建线性层")
        return int(math.prod(num_outputs))
    if isinstance(num_outputs, (int, float)):
        return int(num_outputs)
    raise TypeError(f"无法解析的 num_outputs 类型: {type(num_outputs)}")


class SACImageEncoder(TorchModelV2, nn.Module):
    """
    CNN encoder for SAC with image observations.
    
    Processes 84x84x3 RGB images through convolutional layers.
    Nature CNN architecture adapted for SAC.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # Store additional keyword arguments (e.g., policy_model_config) for compatibility
        self._extra_model_kwargs = kwargs
        
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
        
        output_dim = _ensure_int_outputs(num_outputs)
        if output_dim == 0:
            self.output_layer = nn.Identity()
            self._logits_dim = self.feature_dim
        else:
            self.output_layer = nn.Linear(self.feature_dim, output_dim)
            self._logits_dim = output_dim
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
        device = self._features.device if self._features is not None else torch.device("cpu")
        batch = self._features.shape[0] if self._features is not None else 1
        return torch.zeros(batch, device=device)


class SACLightweightCNN(TorchModelV2, nn.Module):
    """
    Lightweight CNN encoder for smaller image observations.
    
    Suitable for 64x64 or 80x80 images. Faster training than Nature CNN.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self._extra_model_kwargs = kwargs
        
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
        
        output_dim = _ensure_int_outputs(num_outputs)
        if output_dim == 0:
            self.output_layer = nn.Identity()
            self._logits_dim = self.feature_dim
        else:
            self.output_layer = nn.Linear(self.feature_dim, output_dim)
            self._logits_dim = output_dim
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
        device = self._features.device if self._features is not None else torch.device("cpu")
        batch = self._features.shape[0] if self._features is not None else 1
        return torch.zeros(batch, device=device)


__all__ = [
    "SACImageEncoder",
    "SACLightweightCNN",
]
