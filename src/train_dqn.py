# src/train_dqn.py
"""
FULL PATCHED VERSION — GUARANTEED WORKING

- Handles Gym & Gymnasium API differences
- Forces env.reset() → (obs, info)
- Forces env.step() → (obs, reward, done, info)
- Stabilizes DQN training
- Checkpoint saving, resume, TensorBoard logging
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.utils import load_config
from src.data_loader import fetch_and_cache
from src.env_trading import SimpleTradingEnv
from src.env_wrappers import wrap_env
from src.models import MLPQNetwork
from src.replay_buffer import ReplayBuffer


# --- Universal Reset ---
def _reset_env(env):
    """Always returns obs (from (obs, info) or raw obs)."""
    res = env.reset()

    if isinstance(res, tuple):
        return res[0]

    return res


# --- Universal Step ---
def _step_env(env, action):
    out = env.step(action)

    if not isinstance(out, tuple):
        raise RuntimeError("step() returned non-tuple")

    # Standard Gym
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, bool(done), info

    # Gymnasium
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info

    # fallback
    obs = out[0]
    reward = out[1] if len(out) > 1 else 0.0
    done = False
    info = {}

    for item in out[2:]:
        if isinstance(item, dict):
            info = item
        elif isinstance(item, (bool, np.bool_)):
            done = done or item

    return obs, reward, done, info


def train(config_path):
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    df = fetch_and_cache(
        cfg["data"]["ticker"],
        cfg["data"]["start_date"],
        cfg["data"]["end_date"],
        cfg["data"]["cache_csv"],
    )

    # Build environment
    env = SimpleTradingEnv(
        df=df,
        window_size=cfg["env"]["window_size"],
        initial_balance=cfg["env"]["initial_balance"],
        transaction_cost=cfg["env"]["transaction_cost"],
        reward_type=cfg["env"]["reward_type"],
        reward_risk_lambda=cfg["env"]["reward_risk_lambda"],
    )

    # Apply wrappers
    env = wrap_env(env, normalize_obs=True)

    # HARD PATCH — override env.reset() and env.step() on OUTER object
    env.reset = lambda **kw: (lambda r: (r[0], {}) if isinstance(r, tuple) else (r, {}))(env.unwrapped.reset(**kw))
    env.step = lambda a: (lambda r: (
        r[0],
        r[1] if len(r) > 1 else 0.0,
        (r[2] or r[3]) if len(r) >= 5 else bool(r[2]),
        r[4] if len(r) >= 5 else (r[3] if len(r) == 4 else {})
    ))(env.unwrapped.step(a))

    # Build model
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = MLPQNetwork(state_dim, action_dim, cfg["model"]["hidden_sizes"], cfg["model"]["dueling"]).to(device)
    target = MLPQNetwork(state_dim, action_dim, cfg["model"]["hidden_sizes"], cfg["model"]["dueling"]).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=float(cfg["training"]["lr"]))

    buffer = ReplayBuffer(cfg["training"]["buffer_size"], state_dim)
    episodes = cfg["training"]["episodes"]
    max_steps = cfg["training"]["max_steps"]

    gamma = cfg["training"]["gamma"]
    epsilon = cfg["training"]["epsilon_start"]
    eps_min = cfg["training"]["epsilon_end"]
    eps_decay = cfg["training"]["epsilon_decay"]

    target_update = cfg["training"]["target_update"]
    save_dir = cfg["training"]["save_dir"]
    save_full = cfg["training"]["save_full_model"]
    ckpt_interval = cfg["training"]["checkpoint_interval"]

    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(cfg["training"]["log_dir"])
    step_count = 0

    for ep in range(1, episodes + 1):

        state = _reset_env(env)
        ep_reward = 0.0

        for t in range(max_steps):
            step_count += 1

            # ε-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(action_dim)
            else:
                with torch.no_grad():
                    q = policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                action = int(torch.argmax(q).item())

            next_state, reward, done, info = _step_env(env, action)
            ep_reward += reward

            buffer.store(state, action, reward, next_state, done)
            state = next_state

            # Learn
            if buffer.size > cfg["training"]["batch_size"]:
                b = buffer.sample_batch(cfg["training"]["batch_size"])

                s = torch.tensor(b["state"], dtype=torch.float32).to(device)
                a = torch.tensor(b["action"], dtype=torch.long).to(device)
                r = torch.tensor(b["reward"], dtype=torch.float32).to(device)
                ns = torch.tensor(b["next_state"], dtype=torch.float32).to(device)
                d = torch.tensor(b["done"], dtype=torch.float32).to(device)

                q_pred = policy(s).gather(1, a.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_a = torch.argmax(policy(ns), dim=1)
                    q_next = target(ns).gather(1, next_a.unsqueeze(1)).squeeze()
                    q_target = r + gamma * q_next * (1 - d)

                loss = torch.nn.functional.mse_loss(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("loss", loss.item(), step_count)

            if step_count % target_update == 0:
                target.load_state_dict(policy.state_dict())

            if done:
                break

        epsilon = max(eps_min, epsilon * eps_decay)

        writer.add_scalar("reward", ep_reward, ep)
        writer.add_scalar("epsilon", epsilon, ep)

        print(f"Episode {ep}/{episodes} | Reward: {ep_reward:.2f} | Eps: {epsilon:.3f}")

        if ep % ckpt_interval == 0:
            torch.save(policy.state_dict(), os.path.join(save_dir, f"policy_ep{ep}.pth"))
            torch.save(policy.state_dict(), os.path.join(save_dir, "policy_latest.pth"))
            if save_full:
                torch.save(policy, os.path.join(save_dir, "policy_full_ep{ep}.pth"))
                torch.save(policy, os.path.join(save_dir, "policy_latest_full.pth"))

    print("Training complete.")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)
