import torch
import torch.nn as nn
import torch.nn.functional as F

class DispatchEncoder(nn.Module):
    def __init__(self, grid_size, n_dirs=9, n_channels=3, cnn_dim=32, hidden_dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.grid_size = grid_size
        self.n_dirs = n_dirs
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        # Transformer encoder for inter-UAV reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, grid_features, uav_positions):
        """
        grid_obs: [C, grid_size, grid_size] tensor (channels: cusum, uav_map, uncertainty, ...)
        uav_positions: [N, 2] tensor (x, y) positions (row, col convention)
        Returns:
            actions: [N] tensor (directions 0..8)
            new_positions: [N, 2] tensor (next [x, y] for each UAV)
        """
        B, N, _ = uav_positions.shape
        device = uav_positions.device

        # # 1. CNN spatial encoding
        # grid_features = self.cnn(grid_obs) # [B, hidden_dim, grid_size, grid_size]

        # 2. Gather feature vector for each UAV's current position in batch
        # Prepare batch indices for fancy indexing
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)  # [B, N]
        uav_x = uav_positions[..., 0].long()  # [B, N]
        uav_y = uav_positions[..., 1].long()  # [B, N]
        # [B, N, hidden_dim]
        uav_feats = grid_features.permute(0, 2, 3, 1)[batch_idx, uav_x, uav_y]  # [B, N, hidden_dim]

        # 3. Transformer encoder (allows each UAV to "see" others in batch)
        tfm_out = self.transformer(uav_feats)  # [B, N, hidden_dim]
        return tfm_out


class EncoderBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(EncoderBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size

        n_channels = obs_shape[0]
        grid_size = obs_shape[1]

        self.encoder = DispatchEncoder(n_channels=n_channels,
                                       grid_size=grid_size,
                                       hidden_dim=self.hidden_size)

    def forward(self, x, uav_positions):
        x = self.encoder(x, uav_positions)
        return x