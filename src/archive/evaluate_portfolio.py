# src/evaluate_portfolio.py
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data_loader import fetch_multi_asset
from src.env_portfolio import PortfolioEnv
from src.models_td3 import Actor
from src.utils import load_config

def evaluate_actor(actor_path, config_path):
    cfg = load_config(config_path)
    tickers = cfg["data"]["tickers"]
    df_map = fetch_multi_asset(tickers, cfg["data"]["start_date"], cfg["data"]["end_date"], cfg["data"]["cache_dir"])

    env = PortfolioEnv(df_map,
                       window_size=cfg["env"]["window_size"],
                       initial_balance=cfg["env"]["initial_balance"],
                       transaction_cost=cfg["env"]["transaction_cost"],
                       reward_risk_lambda=cfg["env"]["reward_risk_lambda"],
                       normalize_actions=cfg["env"].get("normalize_actions", True))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim, cfg["model"]["actor_hidden"])
    sd = torch.load(actor_path, map_location="cpu")
    actor.load_state_dict(sd)
    actor.eval()

    state = env.reset()
    net_worths = []
    weights_history = []
    prices = []

    done = False
    while not done:
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = actor(s_t).cpu().numpy().flatten()
        state, r, done, info = env.step(action)
        net_worths.append(info.get("net_worth", 0.0))
        weights_history.append(info.get("weights", np.zeros(action_dim)))
        prices.append(info.get("price", np.zeros(action_dim)))

    net_worths = np.array(net_worths)
    weights_history = np.vstack(weights_history)

    # metrics
    returns = np.diff(net_worths) / (net_worths[:-1] + 1e-12)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-9) if len(returns) > 1 else 0.0
    sortino = np.mean(returns) / (np.std(returns[returns < 0]) + 1e-9) if np.any(returns < 0) else float("inf")
    peak = np.maximum.accumulate(net_worths)
    maxdd = np.max(peak - net_worths) if len(net_worths) > 0 else 0.0

    print("Final net worth:", net_worths[-1])
    print("Sharpe:", sharpe, "Sortino:", sortino, "MaxDD:", maxdd)

    # save csv
    pd.DataFrame({"net_worth": net_worths}).to_csv("experiments/td3_equity_eval.csv", index=False)

    # plot equity and average allocation over time
    plt.figure(figsize=(12,6))
    plt.plot(net_worths, label="Net Worth")
    plt.title("TD3 Portfolio Equity Curve")
    plt.legend()
    plt.grid()
    plt.savefig("experiments/td3_equity_eval.png", bbox_inches="tight")
    plt.show()

    # save weights heatmap
    plt.figure(figsize=(12,6))
    plt.imshow(weights_history.T, aspect="auto", interpolation="nearest")
    plt.yticks(range(action_dim), tickers)
    plt.xlabel("Timestep")
    plt.title("Allocations over time (rows = tickers)")
    plt.colorbar(label="Weight")
    plt.savefig("experiments/td3_allocations.png", bbox_inches="tight")
    plt.show()

    return sharpe, sortino, maxdd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=str, required=True)
    parser.add_argument("--config", type=str, default="config_multi_td3.yaml")
    args = parser.parse_args()
    evaluate_actor(args.actor, args.config)
