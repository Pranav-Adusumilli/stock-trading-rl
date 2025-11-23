# src/replay_buffer.py

import numpy as np

class ReplayBuffer:
    """
    Replay buffer with fixed-size storage.
    Stores:
      - state
      - action
      - reward
      - next_state
      - done
    """

    def __init__(self, max_size, state_dim):
        self.max_size = max_size
        self.state_dim = state_dim

        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size,), dtype=np.int64)
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size,), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        """Randomly sample a batch of transitions."""
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = {
            "state": self.states[idxs],
            "action": self.actions[idxs],
            "reward": self.rewards[idxs],
            "next_state": self.next_states[idxs],
            "done": self.dones[idxs],
        }
        return batch

    def __len__(self):
        return self.size
