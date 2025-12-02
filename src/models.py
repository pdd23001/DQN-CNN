import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(m: nn.Module):
    # Stable defaults for small CNNs + MLPs
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class _StateEncoder(nn.Module):
    """
    CNN encoder (paper-ish):
      - Conv(16), MaxPool after first conv
      - Conv(16)
      - Conv(32)
      - kernel=2, stride=1, padding=1
    """

    def __init__(self, in_channels: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        return x


def _infer_emb_dim(encoder: nn.Module, in_channels: int, input_hw: tuple[int, int], device=None) -> int:
    H, W = input_hw
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, H, W, device=device)
        z = encoder(dummy)
        return int(z.shape[1])


class RewardCNN(nn.Module):
    """
    Late-fusion reward model r_hat(s,a):
      - encode state with CNN -> embedding
      - concat (embedding, one_hot(action))
      - MLP -> scalar

    Optional stabilization:
      - output_squash: if True, applies tanh and scales by reward_scale
        (useful if proxy rewards explode and destabilize DQN).
    """

    def __init__(
        self,
        action_dim: int,
        in_channels: int = 5,
        input_hw: tuple[int, int] = (10, 10),
        output_squash: bool = False,
        reward_scale: float = 1.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.encoder = _StateEncoder(in_channels=in_channels)

        emb_dim = _infer_emb_dim(self.encoder, in_channels, input_hw, device=None)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.mlp.apply(_init_weights)

        self.output_squash = bool(output_squash)
        self.reward_scale = float(reward_scale)

    def forward(self, s: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        # s: [B,C,H,W], a_onehot: [B,A]
        # Make sure types are consistent
        if s.dtype != torch.float32:
            s = s.float()
        if a_onehot.dtype != torch.float32:
            a_onehot = a_onehot.float()

        z = self.encoder(s)
        x = torch.cat([z, a_onehot], dim=1)
        r = self.mlp(x).squeeze(-1)  # [B]

        # Optional squash for stability when using as DQN reward
        if self.output_squash:
            r = torch.tanh(r) * self.reward_scale

        return r


class QNetwork(nn.Module):
    """
    DQN network Q(s,a):
      - same CNN encoder
      - MLP head -> Q-values for each action
    """

    def __init__(
        self,
        action_dim: int,
        in_channels: int = 5,
        input_hw: tuple[int, int] = (10, 10),
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.encoder = _StateEncoder(in_channels=in_channels)

        emb_dim = _infer_emb_dim(self.encoder, in_channels, input_hw, device=None)

        self.head = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )
        self.head.apply(_init_weights)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if s.dtype != torch.float32:
            s = s.float()
        z = self.encoder(s)
        return self.head(z)
