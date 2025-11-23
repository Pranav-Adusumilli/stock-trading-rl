# src/models_td3.py
"""
TD3 Actor and Critic networks (PyTorch).

- Actor: MLP -> outputs actions in [-1, 1] via tanh.
- Critic: two separate critic networks (you can instantiate two copies).
  Critic.forward(state, action) -> Q-value (batch, 1)

Architectural choices:
- 2 hidden layers (default [256,256]) with ReLU
- Small weight init scaling for stability
- Compatible with state vectors (flat numpy arrays)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden_sizes, activate_final=False):
    layers = []
    last = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    if activate_final:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(256, 256), final_act=nn.Tanh):
        super().__init__()
        self.net = _mlp(state_dim, list(hidden_sizes))
        self.mu = nn.Linear(hidden_sizes[-1], action_dim)
        # small init
        nn.init.xavier_uniform_(self.mu.weight, gain=1e-1)
        self.final_act = final_act()

    def forward(self, state):
        """
        state: tensor (batch, state_dim)
        returns actions in [-1,1] (batch, action_dim)
        """
        h = self.net(state)
        a = self.mu(h)
        return self.final_act(a)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        # Critic takes state and action concatenated
        self.net = _mlp(state_dim + action_dim, list(hidden_sizes))
        self.q = nn.Linear(hidden_sizes[-1], 1)
        # init
        nn.init.xavier_uniform_(self.q.weight, gain=1e-1)

    def forward(self, state, action):
        """
        state: (batch, state_dim)
        action: (batch, action_dim)
        returns (batch, 1)
        """
        x = torch.cat([state, action], dim=1)
        h = self.net(x)
        q = self.q(h)
        return q
