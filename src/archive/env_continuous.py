"""
Continuous-action trading environment with position sizing (fractional position between -1 and 1).
Action: a single float in [-1, 1] where positive means fraction of portfolio to be invested long,
negative means short fraction.
"""
import gym
from gym import spaces
import numpy as np
import pandas as pd

class ContinuousTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, window_size=20, initial_balance=100000, transaction_cost=0.001):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.prices = self.df['Close'].values
        self.features = df.select_dtypes(include=[np.number]).values
        self.window = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_dim = self.features.shape[1]*self.window + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.idx = self.window
        self.cash = self.initial_balance
        self.position = 0.0
        self.shares = 0.0
        self.net_worth = self.initial_balance
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        win = self.features[self.idx-self.window:self.idx]
        flattened = win.flatten()
        return np.concatenate([flattened, [self.position, self.cash / self.initial_balance]]).astype(np.float32)

    def step(self, action):
        target_pos = float(np.clip(action, -1.0, 1.0))
        price = float(self.prices[self.idx])
        prev_net = self.net_worth
        target_exposure = target_pos * self.net_worth
        current_exposure = self.shares * price
        delta = target_exposure - current_exposure
        if abs(delta) > 1e-6:
            trade_value = abs(delta)
            cost = trade_value * self.transaction_cost
            if delta > 0:
                max_affordable = max(0.0, self.cash - cost)
                actual_trade = min(trade_value, max_affordable)
                shares_bought = actual_trade / price
                self.shares += shares_bought
                self.cash -= actual_trade + cost
            else:
                shares_to_sell = min(self.shares, abs(delta) / price)
                proceeds = shares_to_sell * price
                cost = proceeds * self.transaction_cost
                self.shares -= shares_to_sell
                self.cash += proceeds - cost
        self.idx += 1
        self.net_worth = float(self.cash + self.shares * float(self.prices[self.idx-1]))
        self.position = 0 if self.net_worth==0 else (self.shares * price) / (self.net_worth + 1e-9)
        reward = self.net_worth - prev_net
        self.done = (self.idx >= len(self.prices)-1)
        info = {'net_worth': self.net_worth, 'idx': self.idx, 'position': self.position}
        return self._get_obs(), float(reward), self.done, info

    def render(self, mode='human'):
        print(f"Step {self.idx} | Cash {self.cash:.2f} | Shares {self.shares:.4f} | Net {self.net_worth:.2f} | Pos {self.position:.3f}")
