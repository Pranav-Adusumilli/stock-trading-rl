# src/train_td3.py
"""
TD3 training script (clean, optimized).

Usage:
    python -m src.train_td3 --config config_multi_td3.yaml

Features:
- Multi-asset PortfolioEnv integration
- Optional sentiment-weighted reward (from data_loader / sentiment_fetcher)
- Actor, 2 Critics, target networks, soft updates
- Warmup random steps, action noise, target smoothing
- Checkpoint saving and resume support
- Optional TensorBoard logging (if tensorboard installed)
"""

import os
import time
import argparse
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import load_config
from src.data_loader import fetch_multi_asset
from src.env_portfolio import PortfolioEnv

# Models and replay buffer (expected in repo)
from src.models_td3 import Actor, Critic

# try to use user's ReplayBuffer if present; fallback implemented below
try:
    from src.replay_buffer import ReplayBuffer as UserReplayBuffer  # type: ignore
except Exception:
    UserReplayBuffer = None


# fallback simple replay buffer (numpy)
class SimpleReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size,), dtype=np.float32)
        self.done = np.zeros((self.max_size,), dtype=np.float32)

    def store(self, s, a, r, ns, d):
        self.state[self.ptr] = s
        self.next_state[self.ptr] = ns
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.done[self.ptr] = 1.0 if d else 0.0
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return dict(
            state=self.state[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            next_state=self.next_state[idx],
            done=self.done[idx],
        )


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)


def save_ckpt(save_dir, step, actor, critic1, critic2):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(actor.state_dict(), os.path.join(save_dir, f"actor_{step}.pth"))
    torch.save(critic1.state_dict(), os.path.join(save_dir, f"critic1_{step}.pth"))
    torch.save(critic2.state_dict(), os.path.join(save_dir, f"critic2_{step}.pth"))
    torch.save(actor.state_dict(), os.path.join(save_dir, "actor_latest.pth"))
    torch.save(critic1.state_dict(), os.path.join(save_dir, "critic1_latest.pth"))
    torch.save(critic2.state_dict(), os.path.join(save_dir, "critic2_latest.pth"))


def train(config_path: str):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Fetch data (multi-asset) - fetch_multi_asset returns dict ticker->df
    tickers = cfg["data"]["tickers"]
    df_map = fetch_multi_asset(tickers, cfg["data"]["start_date"], cfg["data"]["end_date"],
                               cfg["data"].get("cache_dir", "data/"),
                               use_sentiment=cfg["data"].get("use_sentiment", False))

    # Build environment
    env_cfg = cfg["env"]
    env = PortfolioEnv(df_map,
                       window_size=env_cfg.get("window_size", 40),
                       initial_balance=env_cfg.get("initial_balance", 100000),
                       transaction_cost=env_cfg.get("transaction_cost", 0.001),
                       reward_risk_lambda=env_cfg.get("reward_risk_lambda", 0.0),
                       sentiment_reward_lambda=env_cfg.get("sentiment_reward_lambda", 0.0),
                       sentiment_reward_alpha=env_cfg.get("sentiment_reward_alpha", 1.0),
                       normalize_actions=env_cfg.get("normalize_actions", True)
                       )

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    print(f"Env created. state_dim={state_dim} action_dim={action_dim} assets={len(tickers)}")

    # Models
    actor = Actor(state_dim, action_dim, cfg["model"].get("actor_hidden", [256, 256])).to(device)
    critic1 = Critic(state_dim, action_dim, cfg["model"].get("critic_hidden", [256, 256])).to(device)
    critic2 = Critic(state_dim, action_dim, cfg["model"].get("critic_hidden", [256, 256])).to(device)

    # targets
    actor_t = Actor(state_dim, action_dim, cfg["model"].get("actor_hidden", [256, 256])).to(device)
    critic1_t = Critic(state_dim, action_dim, cfg["model"].get("critic_hidden", [256, 256])).to(device)
    critic2_t = Critic(state_dim, action_dim, cfg["model"].get("critic_hidden", [256, 256])).to(device)

    actor_t.load_state_dict(actor.state_dict())
    critic1_t.load_state_dict(critic1.state_dict())
    critic2_t.load_state_dict(critic2.state_dict())

    # Optimizers
    actor_opt = optim.Adam(actor.parameters(), lr=float(cfg["training"].get("actor_lr", 3e-4)))
    critic1_opt = optim.Adam(critic1.parameters(), lr=float(cfg["training"].get("critic_lr", 3e-4)))
    critic2_opt = optim.Adam(critic2.parameters(), lr=float(cfg["training"].get("critic_lr", 3e-4)))

    # Replay buffer
    buffer_size = int(cfg["training"].get("buffer_size", 300000))
    batch_size = int(cfg["training"].get("batch_size", 128))
    if UserReplayBuffer is not None:
        # try common constructor signatures, fallback to simple buffer
        try:
            replay = UserReplayBuffer(buffer_size, state_dim, action_dim)
        except TypeError:
            try:
                replay = UserReplayBuffer(state_dim, action_dim, buffer_size)
            except Exception:
                replay = SimpleReplayBuffer(buffer_size, state_dim, action_dim)
    else:
        replay = SimpleReplayBuffer(buffer_size, state_dim, action_dim)

    # hyperparams
    max_steps = int(cfg["training"].get("max_steps", 300000))
    random_steps = int(cfg["training"].get("random_steps", 5000))
    gamma = float(cfg["training"].get("gamma", 0.99))
    tau = float(cfg["training"].get("tau", 0.005))
    policy_delay = int(cfg["model"].get("policy_delay", 2))
    action_noise_std = float(cfg["training"].get("action_noise_std", 0.1))
    target_noise_std = float(cfg["training"].get("target_noise_std", 0.2))
    target_noise_clip = float(cfg["training"].get("target_noise_clip", 0.5))

    save_dir = cfg["training"].get("save_dir", "models_td3/")
    os.makedirs(save_dir, exist_ok=True)
    save_interval = int(cfg["training"].get("save_interval", 5000))
    log_dir = cfg["training"].get("log_dir", "logs_td3/")

    # optional tensorboard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
        tensorboard_ok = True
    except Exception:
        writer = None
        tensorboard_ok = False
        print("TensorBoard unavailable; continuing without it.")

    # seed
    seed = int(cfg["training"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # main loop
    step = 0
    episode = 0
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = np.asarray(state, dtype=np.float32)
    ep_reward = 0.0
    ep_len = 0

    t0 = time.time()
    while step < max_steps:
        # select action
        if step < random_steps:
            action = np.random.uniform(-1.0, 1.0, size=(action_dim,))
        else:
            actor.eval()
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                raw = actor(s_t).cpu().numpy().flatten()
            noise = np.random.normal(0, action_noise_std, size=raw.shape)
            action = np.clip(raw + noise, -1.0, 1.0)

        # environment step
        next_state, reward, done, info = env.step(action)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        next_state = np.asarray(next_state, dtype=np.float32)

        replay.store(state, action, float(reward), next_state, bool(done))
        state = next_state
        ep_reward += float(reward)
        ep_len += 1
        step += 1

        # learn
        if getattr(replay, "size", getattr(replay, "size", None)) is None:
            # compatibility: some custom buffers use .size attribute, others use ._size
            size_ok = getattr(replay, "size", getattr(replay, "_size", None))
        else:
            size_ok = getattr(replay, "size", getattr(replay, "_size", None))
        # we handle via replay.size property or fallback to ptr/len heuristic
        try:
            buffer_current_size = replay.size
        except Exception:
            try:
                buffer_current_size = replay._size
            except Exception:
                # fallback: use ptr as a proxy
                buffer_current_size = getattr(replay, "ptr", 0)

        if buffer_current_size >= batch_size and step >= random_steps:
            batch = replay.sample_batch(batch_size) if hasattr(replay, "sample_batch") else replay.sample(batch_size)
            s_b = torch.tensor(batch["state"], dtype=torch.float32, device=device)
            a_b = torch.tensor(batch["action"], dtype=torch.float32, device=device)
            r_b = torch.tensor(batch["reward"], dtype=torch.float32, device=device).unsqueeze(1)
            ns_b = torch.tensor(batch["next_state"], dtype=torch.float32, device=device)
            d_b = torch.tensor(batch["done"], dtype=torch.float32, device=device).unsqueeze(1)

            # target actions with smoothing
            with torch.no_grad():
                next_a = actor_t(ns_b)
                noise = (torch.randn_like(next_a) * target_noise_std).clamp(-target_noise_clip, target_noise_clip)
                next_a = (next_a + noise).clamp(-1, 1)

                q1_next = critic1_t(ns_b, next_a)
                q2_next = critic2_t(ns_b, next_a)
                q_next = torch.min(q1_next, q2_next)
                q_target = r_b + (1.0 - d_b) * gamma * q_next

            q1 = critic1(s_b, a_b)
            q2 = critic2(s_b, a_b)
            critic1_loss = nn.MSELoss()(q1, q_target)
            critic2_loss = nn.MSELoss()(q2, q_target)

            critic1_opt.zero_grad(); critic1_loss.backward(); critic1_opt.step()
            critic2_opt.zero_grad(); critic2_loss.backward(); critic2_opt.step()

            # delayed policy update
            if step % policy_delay == 0:
                actor_opt.zero_grad()
                actor_loss = -critic1(s_b, actor(s_b)).mean()
                actor_loss.backward()
                actor_opt.step()

                soft_update(actor_t, actor, tau)
                soft_update(critic1_t, critic1, tau)
                soft_update(critic2_t, critic2, tau)

            if tensorboard_ok and writer is not None:
                writer.add_scalar("train/critic1_loss", critic1_loss.item(), step)
                writer.add_scalar("train/critic2_loss", critic2_loss.item(), step)
                if step % (policy_delay * 10) == 0:
                    writer.add_scalar("train/actor_loss", actor_loss.item(), step)

        # episode end
        if done:
            episode += 1
            elapsed = time.time() - t0
            print(f"Episode {episode} | Step {step}/{max_steps} | EpReward {ep_reward:.2f} | EpLen {ep_len} | Time {elapsed:.1f}s")
            if tensorboard_ok and writer is not None:
                writer.add_scalar("train/episode_reward", ep_reward, episode)
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            state = np.asarray(state, dtype=np.float32)
            ep_reward = 0.0
            ep_len = 0

        # periodic save
        if step % save_interval == 0:
            save_ckpt(save_dir, step, actor, critic1, critic2)
            print("Saved checkpoint at step", step)

    # final save & cleanup
    save_ckpt(save_dir, step, actor, critic1, critic2)
    if tensorboard_ok and writer is not None:
        writer.flush(); writer.close()
    print("Training finished. Models saved to", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_multi_td3.yaml", help="Path to config YAML")
    args = parser.parse_args()
    train(args.config)
