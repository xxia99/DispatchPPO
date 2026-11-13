import torch.nn as nn
from .util import init

"""CNN Modules and utils."""

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU,
                 kernel_size=3, stride=1, cnn_dim=64):
        super(CNNLayer, self).__init__()

        input_channel = obs_shape[0]

        # CNN for grid state: flexible channel size
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, cnn_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_dim, cnn_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_dim, hidden_size, kernel_size=1)
        )

    def forward(self, x):
        x = self.cnn(x)
        return x


class CNNBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size

        self.cnn = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        x = self.cnn(x)
        return x
