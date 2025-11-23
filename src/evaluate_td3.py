# src/evaluate_td3.py
"""
Evaluate a trained TD3 actor on the portfolio environment.

Usage:
  python -m src.evaluate_td3 --actor models_td3/actor_latest.pth --config config_multi_td3.yaml --episodes 1 --deterministic

Notes:
 - Reads config_multi_td3.yaml for data/env settings (including sentiment_reward_lambda)
 - Handles torch checkpoint being either a state_dict (OrderedDict) or a full model object.
"""

import os
import argparse
import time
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data_loader import fetch_multi_asset
from src.env_portfolio import PortfolioEnv
from src.utils import load_config  # in case you keep a helper
# If you don't have load_config, fallback to yaml.safe_load
def _load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# simple metrics
def sharpe_ratio(returns, annual_factor=252.0):
    r = np.array(returns)
    if r.size == 0: return 0.0
    mean = r.mean() * annual_factor
    std = r.std() * np.sqrt(annual_factor)
    return float(mean / (std + 1e-9))

def sortino_ratio(returns, annual_factor=252.0):
    r = np.array(returns)
    if r.size == 0: return 0.0
    mean = r.mean() * annual_factor
    neg = r[r < 0]
    downside = np.std(neg) * np.sqrt(annual_factor) if neg.size>0 else 1e-9
    return float(mean / (downside + 1e-9))

def max_drawdown(equity_curve):
    eq = np.array(equity_curve, dtype=float)
    if eq.size == 0: return 0.0
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return float(np.max(dd))

def try_load_actor(actor_path, state_dim, action_dim, hidden_sizes):
    """
    Tries to load a torch checkpoint. If it's a state_dict, we will attempt to construct
    a minimal MLP actor compatible with state_dim->action_dim and load the dict.
    If it's a full model (has .eval), return that object.
    """
    loaded = torch.load(actor_path, map_location="cpu")
    # if it's a module object (has eval), return it
    if hasattr(loaded, "eval") and hasattr(loaded, "state_dict"):
        loaded.eval()
        return loaded

    # otherwise assume it's a state_dict mapping names -> tensors
    from types import SimpleNamespace
    # define a simple MLP actor that matches training architecture (replace if you have a custom actor)
    import torch.nn as nn
    class SimpleActor(nn.Module):
        def __init__(self, inp, outp, hidden):
            super().__init__()
            hs = [inp] + list(hidden) + [outp]
            layers = []
            for i in range(len(hs)-2):
                layers.append(nn.Linear(hs[i], hs[i+1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hs[-2], hs[-1]))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return torch.sigmoid(self.net(x))  # sigmoid => outputs 0..1 for weights
    actor = SimpleActor(state_dim, action_dim, hidden_sizes)
    try:
        actor.load_state_dict(loaded)
        actor.eval()
        return actor
    except Exception as e:
        print("Failed to load state_dict into default SimpleActor:", e)
        # as a last resort, return loaded object (may fail later)
        return loaded

def evaluate(actor_path, config_path, episodes=1, deterministic=False, save_dir="experiments"):
    cfg = _load_config(config_path)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tickers = cfg["data"].get("tickers", [cfg["data"].get("ticker", "AAPL")])
    df_map = fetch_multi_asset(tickers, cfg["data"]["start_date"], cfg["data"]["end_date"], cfg["data"].get("cache_dir","data/"), use_sentiment=cfg["env"].get("include_sentiment", True))

    # create env with sentiment lambda passed from config
    env = PortfolioEnv(
        df_map,
        window_size=cfg["env"].get("window_size", 40),
        initial_balance=cfg["env"].get("initial_balance", 100000),
        transaction_cost=cfg["env"].get("transaction_cost", 0.001),
        reward_type=cfg["env"].get("reward_type", "net_worth"),
        reward_risk_lambda=cfg["env"].get("reward_risk_lambda", 0.0),
        include_sentiment=cfg["env"].get("include_sentiment", True),
        sentiment_reward_lambda=cfg["env"].get("sentiment_reward_lambda", 0.0),
        max_steps=cfg["training"].get("max_steps", None),
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_sizes = cfg["model"].get("hidden_sizes", [256,256])

    print("Env observation dimension:", state_dim)
    print("Loading actor:", actor_path)
    actor = try_load_actor(actor_path, state_dim, action_dim, hidden_sizes)
    actor.to(device)
    actor.eval()

    all_results = []

    for ep in range(1, episodes+1):
        obs = env.reset()
        done = False
        eq = [env.net_worth]
        action_history = []
        weights_history = []
        rewards = []
        steps = 0

        while not done:
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                a = actor(s)
            # ensure vector shape (action_dim,)
            if isinstance(a, torch.Tensor):
                a = a.squeeze().cpu().numpy()
            else:
                a = np.array(a).squeeze()
            # deterministic/clamp
            a = np.clip(a, 0.0, 1.0)
            # step
            next_obs, reward, done, info = env.step(a)
            rewards.append(float(reward))
            action_history.append(a)
            weights_history.append(info.get("weights", np.zeros(action_dim)))
            obs = next_obs
            eq.append(info.get("net_worth", env.net_worth))
            steps += 1
            # safety escape
            if steps > (cfg["training"].get("max_steps", 5000) + 5):
                break

        # metrics
        eq = np.array(eq)
        daily_returns = np.diff(eq) / (eq[:-1] + 1e-9)
        sr = sharpe_ratio(daily_returns)
        so = sortino_ratio(daily_returns)
        mdd = max_drawdown(eq)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = {
            "episode": ep,
            "final_net_worth": float(eq[-1]),
            "equity_curve": eq.tolist(),
            "daily_returns": daily_returns.tolist(),
            "sharpe": sr,
            "sortino": so,
            "max_drawdown": mdd,
            "actions": np.vstack(action_history).tolist() if len(action_history)>0 else [],
            "weights": np.vstack(weights_history).tolist() if len(weights_history)>0 else [],
        }
        all_results.append(out)

        # plotting
        plt.figure(figsize=(12,4))
        plt.plot(eq, label="Net Worth")
        plt.title(f"Evaluation Episode {ep} Net Worth {eq[-1]:.2f}")
        plt.xlabel("Step")
        plt.ylabel("Net Worth")
        plt.grid(True)
        plt.legend()
        plot_path = os.path.join(save_dir, f"eval_{now}_ep{ep}_networth.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        # action heatmap (actions per asset over time)
        if len(out["actions"])>0:
            acts = np.array(out["actions"])
            plt.figure(figsize=(10,4))
            plt.imshow(acts.T, aspect='auto', interpolation='nearest')
            plt.colorbar(label="action (0..1)")
            plt.yticks(range(action_dim), tickers)
            plt.title(f"Actions Heatmap ep{ep}")
            plt.xlabel("Step")
            plt.ylabel("Asset")
            hm_path = os.path.join(save_dir, f"eval_{now}_ep{ep}_actions_heatmap.png")
            plt.savefig(hm_path, bbox_inches="tight")
            plt.close()

        # arrow plot for asset 0: mark buy/sell arrows on close price
        try:
            asset0 = tickers[0]
            df0 = df_map[asset0]
            # pick the window of indices corresponding to the run
            # we'll plot closes from window_size-1 .. window_size-1+len(actions)
            idx0 = env.window_size - 1
            closes = df0["Close"].iloc[idx0: idx0 + len(out["actions"])].to_numpy()
            fig, ax = plt.subplots(1,1,figsize=(12,4))
            ax.plot(closes, label=f"{asset0} Close")
            # add arrows: action > 0.6 -> buy (green up), <0.4 -> sell (red down)
            acts0 = np.array(out["actions"])[:,0] if len(out["actions"])>0 else np.array([])
            for i, a in enumerate(acts0):
                if a > 0.6:
                    ax.annotate("", xy=(i, closes[i]), xytext=(i, closes[i]- (0.01*closes[i])),
                                arrowprops=dict(arrowstyle="->", color="g"))
                elif a < 0.4:
                    ax.annotate("", xy=(i, closes[i]), xytext=(i, closes[i]+ (0.01*closes[i])),
                                arrowprops=dict(arrowstyle="->", color="r"))
            ax.set_title(f"{asset0} closes with action arrows (ep{ep})")
            ax.grid(True)
            arrow_path = os.path.join(save_dir, f"eval_{now}_ep{ep}_arrows.png")
            fig.savefig(arrow_path, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print("Arrow plot failed:", e)

        # save JSON summary
        import json
        summary_path = os.path.join(save_dir, f"eval_{now}_ep{ep}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"Episode {ep} finished | Final Net Worth: {eq[-1]:.2f} | Sharpe: {sr:.3f} | Sortino: {so:.3f} | MDD: {mdd:.2f}")
        print(f"Saved outputs to {save_dir}")

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=str, required=True)
    parser.add_argument("--config", type=str, default="config_multi_td3.yaml")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    evaluate(args.actor, args.config, episodes=args.episodes, deterministic=args.deterministic)
