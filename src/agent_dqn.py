"""
DQN agent wrapper (action selection, training step).
"""
import numpy as np
import torch

class DQNAgent:
    def __init__(self, policy_net, target_net, optimizer, device, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.policy = policy_net
        self.target = target_net
        self.opt = optimizer
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.policy.fc_adv.out_features if hasattr(self.policy,'fc_adv') else self.policy.fc_out.out_features)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy(state_t)
            return int(q.argmax().cpu().numpy())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_step(self, batch, gamma):
        import torch.nn.functional as F
        s, a, r, s2, d = batch
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        s2 = torch.tensor(s2, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

        q_values = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(s2).max(1)[0]
            q_target = r + gamma * q_next * (1 - d)
        loss = F.mse_loss(q_values, q_target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
