import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    """
    정책 네트워크 (게임 상태 → 행동 확률)
    Chloe RL 전용 MLP
    """

    def __init__(self, state_dim=8, hidden_dim=128, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class ValueNetwork(nn.Module):
    """
    가치 네트워크 (PPO용)
    """
    def __init__(self, state_dim: int = 8, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.network(state)

