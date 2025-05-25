from gymnasium.spaces.discrete import Discrete
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn


class CNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space: Discrete, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(obs_space.shape[-1], 32, kernel_size=3, stride=2, padding=1),  # Output: 40x40x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 20x20x64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 10x10x128
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: 7x7x256
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self._features = None

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def forward(self, input_dict, state, seq_lens):
        # logging.info(input_dict)
        # logging.info(input_dict["obs"].shape)
        self._features = input_dict["obs"].float()
        # permute b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        self._features = self.conv_layers(self._features)
        return self._features.flatten(1), state

    def value_function(self):
        pass
