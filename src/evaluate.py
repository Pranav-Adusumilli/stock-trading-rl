# src/evaluate.py
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data_loader import fetch_and_cache
from src.env_trading import SimpleTradingEnv
from src.env_wrappers import wrap_env
from src.models import MLPQNetwork
from src.utils import load_config


def _reset_env(env):
    """
    Use the unwrapped env.reset() to get raw return values and normalize.
    Returns obs (not (obs, info)).
    """
    # prefer unwrapped to avoid wrapper unpacking mismatch
    if hasattr(env, "unwrapped"):
        raw = env.unwrapped.reset()
    else:
        raw = env.reset()

    # raw may be: obs, (obs, info), (obs, info, extra...), etc.
    if not isinstance(raw, tuple):
        return raw
    return raw[0]


def _step_env(env, action):
    """
    Use unwrapped.step() to get raw outputs, then normalize to (next_obs, reward, done, info).
    Accepts Gym (4-tuple) and Gymnasium (5-tuple) and other custom variants.
    """
    if hasattr(env, "unwrapped"):
        raw = env.unwrapped.step(action)
    else:
        raw = env.step(action)

    if not isinstance(raw, tuple):
        raise RuntimeError("env.step returned non-tuple result")

    # Standard gym: (obs, reward, done, info)
    if len(raw) == 4:
        next_obs, reward, done, info = raw
        return next_obs, float(reward), bool(done), info

    # Gymnasium: (obs, reward, terminated, truncated, info)
    if len(raw) == 5:
        next_obs, reward, terminated, truncated, info = raw
        done = bool(terminated or truncated)
        return next_obs, float(reward), done, info

    # fallback: try to extract obs, reward, info, and any boolean done flags
    next_obs = raw[0]
    reward = float(raw[1]) if len(raw) > 1 else 0.0
    done = False
    info = {}
    for item in raw[2:]:
        if isinstance(item, dict):
            info = item
        elif isinstance(item, (bool, np.bool_)):
            done = done or bool(item)
    return next_obs, reward, done, info


def _safe_align_state(state, expected_dim):
    """
    Fixes mismatch between env observation size and model input size.
    - If state is too short: pad with zeros.
    - If state is too long: truncate.
    """
    state = np.array(state, dtype=np.float32).flatten()

    if len(state) == expected_dim:
        return state

    if len(state) == 0:
        # fallback: produce zero vector
        return np.zeros(expected_dim, dtype=np.float32)

    if len(state) > expected_dim:
        return state[:expected_dim]

    # pad if shorter
    pad_len = expected_dim - len(state)
    return np.concatenate([state, np.zeros(pad_len, dtype=np.float32)])


def evaluate(checkpoint_path, ticker="AAPL"):
    cfg = load_config("config.yaml")

    cache_path = cfg["data"]["cache_csv"].format(ticker=ticker)
    print("Using cached data:", cache_path)
    df = fetch_and_cache(ticker, cfg["data"]["start_date"], cfg["data"]["end_date"], cfg["data"]["cache_csv"])

    # build evaluation environment
    env = SimpleTradingEnv(
        df=df,
        window_size=cfg["env"]["window_size"],
        initial_balance=cfg["env"]["initial_balance"],
        transaction_cost=cfg["env"]["transaction_cost"],
        reward_type=cfg["env"]["reward_type"],
        reward_risk_lambda=cfg["env"]["reward_risk_lambda"],
    )
    # wrap for observation normalization etc — we still call unwrapped.reset/step inside helpers
    env = wrap_env(env, normalize_obs=True)

    # determine env obs dim
    dummy_state = _reset_env(env)
    dummy_state = np.array(dummy_state).flatten()
    env_dim = len(dummy_state)
    print("\nEnv observation dimension:", env_dim)

    # load checkpoint
    print("Loading checkpoint:", checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu")

    # infer model input dimension from weight shapes
    # try common key names robustly
    model_input_dim = None
    for key in ("fc1.weight", "l1.weight", "0.weight"):
        if key in loaded:
            model_input_dim = loaded[key].shape[1]
            break

    if model_input_dim is None:
        # fallback: try to find first param that is 2D
        for v in loaded.values():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                model_input_dim = v.shape[1]
                break

    if model_input_dim is None:
        raise RuntimeError("Could not infer model input dim from checkpoint.")

    print("Model expects state_dim:", model_input_dim)

    # build model
    model = MLPQNetwork(
        input_dim=model_input_dim,
        n_actions=env.action_space.n,
        hidden_sizes=cfg["model"]["hidden_sizes"],
        dueling=cfg["model"]["dueling"],
    )

    # load weights (allow non-strict so we can handle slight mismatches safely)
    model.load_state_dict(loaded, strict=False)
    model.eval()

    # start evaluation
    state = _reset_env(env)
    state = _safe_align_state(state, model_input_dim)

    net_worths = []
    actions_taken = []
    prices = []
    info_list = []

    done = False

    while not done:
        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = model(s_tensor)
            action = int(torch.argmax(q).item())

        next_state, reward, done, info = _step_env(env, action)

        # fix next state dimension
        next_state = _safe_align_state(next_state, model_input_dim)

        prices.append(info.get("price", 0.0))
        net_worths.append(info.get("net_worth", 0.0))
        actions_taken.append(action)
        info_list.append(info)

        state = next_state

    if len(net_worths) == 0:
        print("No timesteps recorded during evaluation — check env / data.")
        return None, None, None

    # convert to arrays
    prices = np.array(prices).flatten()
    net_worths = np.array(net_worths)

    print("\nFinal Net Worth:", net_worths[-1])

    # compute metrics (simple)
    # use returns computed from net_worths
    returns = np.diff(net_worths) / (net_worths[:-1] + 1e-12)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-9) if len(returns) > 1 else 0.0
    sortino = np.mean(returns) / (np.std(returns[returns < 0]) + 1e-9) if np.any(returns < 0) else float("inf")
    peak = np.maximum.accumulate(net_worths)
    drawdowns = peak - net_worths
    maxdd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    print("\n--- Evaluation Metrics ---")
    print("Sharpe Ratio:", sharpe)
    print("Sortino Ratio:", sortino)
    print("Max Drawdown:", maxdd)
    print("--------------------------")

    # save equity curve
    pd.DataFrame({"net_worth": net_worths}).to_csv("experiments/equity_curve_eval.csv", index=False)

    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(net_worths, label="Net Worth")
    plt.title("Equity Curve")
    plt.grid()
    plt.legend()
    plt.savefig("experiments/equity_curve.png", bbox_inches="tight")
    plt.show()

    return sharpe, sortino, maxdd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ticker", type=str, default="AAPL")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.ticker)
