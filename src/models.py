import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPQNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_sizes=[256,256], dueling=True):
        super().__init__()
        self.dueling = dueling
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        if dueling:
            self.fc_value = nn.Linear(hidden_sizes[1], 1)
            self.fc_adv = nn.Linear(hidden_sizes[1], n_actions)
        else:
            self.fc_out = nn.Linear(hidden_sizes[1], n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.dueling:
            v = self.fc_value(x)
            a = self.fc_adv(x)
            q = v + (a - a.mean(dim=1, keepdim=True))
            return q
        else:
            return self.fc_out(x)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_sizes=[256,256]):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_sizes[1], n_actions)
        self.critic = nn.Linear(hidden_sizes[1], 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)
