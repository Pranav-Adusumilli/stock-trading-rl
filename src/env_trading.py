# src/env_trading.py
import numpy as np
import gym

class SimpleTradingEnv(gym.Env):
    """
    Discrete stock trading environment.
    Actions:
      0 = HOLD
      1 = BUY
      2 = SELL
    """

    def __init__(
        self,
        df,
        window_size=20,
        initial_balance=100000,
        transaction_cost=0.001,     # ADDED
        reward_type="net_worth",
        reward_risk_lambda=0.0
    ):
        super().__init__()

        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reward_type = reward_type
        self.reward_risk_lambda = reward_risk_lambda

        # Ensure Close exists
        if "Close" not in df.columns:
            raise KeyError("DataFrame must contain 'Close' column.")

        self.prices = df["Close"].values
        self.features = df.drop(["Close"], axis=1).values

        # Observation: flattened window of indicators
        obs_shape = (window_size * self.features.shape[1],)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        # Action space
        self.action_space = gym.spaces.Discrete(3)

        self.reset()

    def reset(self):
        self.balance = float(self.initial_balance)
        self.shares = 0
        self.net_worth = float(self.initial_balance)
        self.max_net_worth = float(self.initial_balance)
        self.current_step = self.window_size
        return self._get_observation()

    def _get_observation(self):
        start = self.current_step - self.window_size
        window = self.features[start:self.current_step]
        return window.flatten().astype(np.float32)

    def step(self, action):
        price = float(self.prices[self.current_step])
        prev_networth = self.net_worth
        fee = self.transaction_cost

        # --- BUY ---
        if action == 1:
            max_shares = int(self.balance / price)
            if max_shares > 0:
                cost = max_shares * price
                cost_with_fee = cost * (1 + fee)
                if cost_with_fee <= self.balance:
                    self.balance -= cost_with_fee
                    self.shares += max_shares

        # --- SELL ---
        elif action == 2:
            if self.shares > 0:
                revenue = self.shares * price
                revenue_after_fee = revenue * (1 - fee)
                self.balance += revenue_after_fee
                self.shares = 0

        # update net worth
        self.net_worth = self.balance + self.shares * price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # reward
        delta = self.net_worth - prev_networth
        if self.reward_type == "net_worth":
            reward = delta
        else:
            reward = delta - self.reward_risk_lambda * abs(delta)

        # step update
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        info = {
            "net_worth": self.net_worth,
            "price": price,
            "shares": self.shares
        }

        return self._get_observation(), reward, done, info
