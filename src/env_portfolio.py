# src/env_portfolio.py
"""
Portfolio multi-asset trading environment with optional sentiment reward shaping.

Features:
- Multi-asset continuous allocation actions (weights that sum to 1)
- Optional sentiment features included in observation (if present in dataframes)
- reward_type: "net_worth" (default) or "pnl" (daily pnl)
- reward_risk_lambda: penalty multiplier applied to recent volatility/drawdown
- sentiment_reward_lambda: scales sentiment-weighted reward boost/penalty
- Tracks net worth and drawdown in `info`
- Compatible with Gym and Gymnasium reset/step signatures (returns obs or (obs, info) when needed).
"""

from typing import Dict, List
import numpy as np
import pandas as pd
import gym
from gym import spaces


class PortfolioEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df_map: Dict[str, pd.DataFrame],
        window_size: int = 40,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        reward_type: str = "net_worth",
        reward_risk_lambda: float = 0.0,
        include_sentiment: bool = True,
        sentiment_reward_lambda: float = 0.0,
        max_steps: int = None,
    ):
        """
        df_map: dict ticker -> pd.DataFrame (index consecutive)
                Dataframes should contain columns: Close, Open, High, Low, Volume,
                and optionally 'sentiment' or 'sentiment_score' (float).
        window_size: number of lookback timesteps used for observations
        reward_type: 'net_worth' or 'pnl'
        reward_risk_lambda: multiplier for volatility/drawdown penalty
        include_sentiment: if True and df has sentiment columns, include them in obs
        sentiment_reward_lambda: scale to convert sentiment signal into reward shaping
        """
        super().__init__()

        # store params
        self.dfs = {k: v.reset_index(drop=True).copy() for k, v in df_map.items()}
        self.tickers = list(self.dfs.keys())
        self.n_assets = len(self.tickers)
        self.window_size = int(window_size)
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.reward_type = reward_type
        self.reward_risk_lambda = float(reward_risk_lambda)
        self.include_sentiment = bool(include_sentiment)
        self.sentiment_reward_lambda = float(sentiment_reward_lambda)

        # max steps default
        lengths = [len(df) for df in self.dfs.values()] if len(self.dfs) > 0 else [0]
        self.episode_length = max(0, min(lengths) - self.window_size)
        if max_steps is None:
            self.max_steps = max(1, self.episode_length)
        else:
            self.max_steps = int(max_steps)

        # detect sentiment column per asset
        self.sentiment_cols_by_asset = {}
        for t, df in self.dfs.items():
            s_col = None
            if "sentiment" in df.columns:
                s_col = "sentiment"
            elif "sentiment_score" in df.columns:
                s_col = "sentiment_score"
            self.sentiment_cols_by_asset[t] = s_col

        # Precompute derived columns and ensure numeric types
        for t, df in self.dfs.items():
            df["Close"] = pd.to_numeric(df.get("Close", pd.Series(dtype=float)), errors="coerce")
            df["Open"] = pd.to_numeric(df.get("Open", df["Close"]), errors="coerce")
            df["High"] = pd.to_numeric(df.get("High", df["Close"]), errors="coerce")
            df["Low"] = pd.to_numeric(df.get("Low", df["Close"]), errors="coerce")
            df["Volume"] = pd.to_numeric(df.get("Volume", 0), errors="coerce")
            df["ret"] = df["Close"].pct_change().fillna(0.0)
            df["sma_10"] = df["Close"].rolling(10, min_periods=1).mean().bfill().ffill()
            df["sma_40"] = df["Close"].rolling(40, min_periods=1).mean().bfill().ffill()
            if self.sentiment_cols_by_asset[t] is not None:
                df[self.sentiment_cols_by_asset[t]] = pd.to_numeric(df[self.sentiment_cols_by_asset[t]], errors="coerce").fillna(0.0)
            self.dfs[t] = df

        # Observation design:
        # per asset vector: window_size recent returns + sma10_norm + sma40_norm + last_close_norm + last_vol_norm (+ sentiment scalar)
        per_asset_vector_len = self.window_size + 4
        if any(self.sentiment_cols_by_asset.values()) and self.include_sentiment:
            per_asset_vector_len += 1
        self.obs_per_asset = per_asset_vector_len
        self.obs_dim = self.obs_per_asset * self.n_assets + 2  # + cash fraction + gross exposure

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

        # internal state
        self._reset_internal()

    def _reset_internal(self):
        self.current_step = 0
        self.cash = float(self.initial_balance)
        self.positions = np.zeros(self.n_assets, dtype=np.float32)
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.net_worth = float(self.initial_balance)
        self.net_worth_history = []
        self._last_trade_prices = np.zeros(self.n_assets, dtype=np.float32)
        self.done = False

    def reset(self, **kwargs):
        """
        Reset environment. If kwargs contains 'df_map', the dataframes will be replaced
        (useful for random-start slicing outside the env).
        Returns obs (Gym style).
        """
        if "df_map" in kwargs:
            self.dfs = {k: v.reset_index(drop=True).copy() for k, v in kwargs["df_map"].items()}
            # re-detect sentiment columns in case df_map differs
            self.sentiment_cols_by_asset = {}
            for t, df in self.dfs.items():
                s_col = None
                if "sentiment" in df.columns:
                    s_col = "sentiment"
                elif "sentiment_score" in df.columns:
                    s_col = "sentiment_score"
                self.sentiment_cols_by_asset[t] = s_col

        self._reset_internal()
        # initialize last trade prices using the last index of the initial window
        for i, t in enumerate(self.tickers):
            df = self.dfs[t]
            idx0 = max(0, self.window_size - 1)
            self._last_trade_prices[i] = float(df.at[idx0, "Close"])
        self.net_worth = float(self.initial_balance)
        self.net_worth_history = [self.net_worth]
        obs = self._build_obs(self.current_step)
        return obs

    def _normalize_volume(self, v):
        return float(np.log1p(float(v)) / 100.0)

    def _build_obs(self, step_idx):
        vecs = []
        for t in self.tickers:
            df = self.dfs[t]
            idx = step_idx + (self.window_size - 1)
            start = max(0, idx - (self.window_size - 1))
            window_rets = df["ret"].iloc[start: idx + 1].to_numpy(dtype=np.float32)
            if len(window_rets) < self.window_size:
                pad = np.zeros(self.window_size - len(window_rets), dtype=np.float32)
                window_rets = np.concatenate([pad, window_rets])
            sma10 = float(df.at[idx, "sma_10"])
            sma40 = float(df.at[idx, "sma_40"])
            last_close = float(df.at[idx, "Close"])
            last_vol = float(df.at[idx, "Volume"])
            sma10_n = (sma10 / (last_close + 1e-9)) - 1.0
            sma40_n = (sma40 / (last_close + 1e-9)) - 1.0
            last_close_n = np.log1p(last_close) / 1000.0
            last_vol_n = self._normalize_volume(last_vol)
            part = np.concatenate([window_rets.astype(np.float32), np.array([sma10_n, sma40_n, last_close_n, last_vol_n], dtype=np.float32)])
            if self.include_sentiment and self.sentiment_cols_by_asset[t] is not None:
                sent_val = float(df.at[idx, self.sentiment_cols_by_asset[t]])
                part = np.concatenate([part, np.array([sent_val], dtype=np.float32)])
            vecs.append(part)
        obs_vec = np.concatenate(vecs).astype(np.float32)
        cash_frac = np.array([self.cash / (self.initial_balance + 1e-9)], dtype=np.float32)
        gross_exposure = np.array([np.sum(np.abs(self.weights))], dtype=np.float32)
        obs = np.concatenate([obs_vec, cash_frac, gross_exposure]).astype(np.float32)
        return obs

    def _compute_sentiment_signal(self, idx):
        """
        Compute a scalar sentiment signal at given index by averaging per-asset sentiment values.
        Returns array of per-asset sentiments and a scalar weighted sentiment (if weights available).
        """
        s_vals = np.zeros(self.n_assets, dtype=np.float32)
        for i, t in enumerate(self.tickers):
            s_col = self.sentiment_cols_by_asset.get(t, None)
            if s_col is not None:
                df = self.dfs[t]
                # bound index
                idx0 = min(max(0, idx), len(df) - 1)
                s_vals[i] = float(df.at[idx0, s_col])
            else:
                s_vals[i] = 0.0
        # scalar aggregated sentiment (simple mean)
        agg = float(np.nanmean(s_vals)) if s_vals.size > 0 else 0.0
        return s_vals, agg

    def step(self, action):
        """
        action: array of length n_assets (>=0). We'll normalize to weights that sum to 1.
        Returns (obs, reward, done, info)
        """
        if self.done:
            return self._build_obs(self.current_step), 0.0, True, {}

        a = np.array(action, dtype=np.float32).copy()
        a = np.clip(a, 0.0, None)
        if a.sum() <= 0:
            a = np.ones_like(a, dtype=np.float32)
        w = a / (a.sum() + 1e-9)

        idx = self.current_step + (self.window_size - 1)
        prices = np.array([float(self.dfs[t].at[min(idx, len(self.dfs[t]) - 1), "Close"]) for t in self.tickers], dtype=np.float32)

        total_value = self.cash + (self.positions * prices).sum()
        target_allocation = w * total_value
        current_value = (self.positions * prices)
        trade_values = target_allocation - current_value
        trade_shares = trade_values / (prices + 1e-9)

        traded_value_abs = np.abs(trade_shares * prices).sum()
        commission = traded_value_abs * float(self.transaction_cost)

        self.positions = self.positions + trade_shares
        self.cash = self.cash - (trade_shares * prices).sum() - commission
        self._last_trade_prices = prices.copy()
        self.weights = w.copy()

        # advance step
        self.current_step += 1
        idx_new = min(self.current_step + (self.window_size - 1), min([len(df) for df in self.dfs.values()]) - 1)
        prices_new = np.array([float(self.dfs[t].at[idx_new, "Close"]) for t in self.tickers], dtype=np.float32)
        pv = (self.positions * prices_new).sum()
        self.net_worth = float(self.cash + pv)
        self.net_worth_history.append(self.net_worth)

        # base reward
        if self.reward_type == "pnl":
            prev = self.net_worth_history[-2] if len(self.net_worth_history) >= 2 else self.initial_balance
            reward = self.net_worth - prev
        else:
            prev = self.net_worth_history[-2] if len(self.net_worth_history) >= 2 else self.initial_balance
            reward = (self.net_worth - prev) / (prev + 1e-9)

        # risk penalty (volatility or drawdown)
        if len(self.net_worth_history) >= 2:
            eq = np.array(self.net_worth_history, dtype=np.float64)
            rets = np.diff(eq) / (eq[:-1] + 1e-9)
            vol = float(np.std(rets[-min(len(rets), 20):]) if rets.size > 0 else 0.0)
        else:
            vol = 0.0
        peak = float(np.max(self.net_worth_history))
        drawdown = max(0.0, peak - self.net_worth)
        risk_penalty = float(self.reward_risk_lambda) * max(vol, drawdown / (self.initial_balance + 1e-9))
        reward = float(reward) - risk_penalty

        # sentiment reward shaping
        sentiment_scalar = 0.0
        per_asset_sentiments = np.zeros(self.n_assets, dtype=np.float32)
        if self.include_sentiment and any(self.sentiment_cols_by_asset.values()):
            # compute per-asset and aggregate sentiment at idx_new (valuation index)
            per_asset_sentiments, sentiment_scalar = self._compute_sentiment_signal(idx_new)
            # sentiment alignment: dot(weights, per_asset_sentiments)
            weighted_sentiment = float(np.dot(self.weights, per_asset_sentiments))
            # shape reward: if allocation aligned with sentiment, boost reward
            sentiment_bonus = float(self.sentiment_reward_lambda) * weighted_sentiment
            reward = float(reward) + sentiment_bonus
        else:
            weighted_sentiment = 0.0

        # done criteria
        done = False
        if self.current_step >= self.max_steps:
            done = True
        if (self.current_step + (self.window_size - 1)) >= min([len(df) for df in self.dfs.values()]):
            done = True

        info = {
            "net_worth": self.net_worth,
            "cash": float(self.cash),
            "positions": self.positions.copy(),
            "weights": self.weights.copy(),
            "step_index": self.current_step,
            "drawdown": float(drawdown),
            "volatility": float(vol),
            "sentiment_scalar": float(sentiment_scalar),
            "weighted_sentiment": float(weighted_sentiment),
        }

        obs = self._build_obs(self.current_step if not done else max(0, self.current_step - 1))
        return obs, float(reward), bool(done), info

    def render(self, mode="human"):
        print(f"Step {self.current_step} NetWorth {self.net_worth:.2f} Cash {self.cash:.2f} Weights {self.weights}")

    def close(self):
        return None
