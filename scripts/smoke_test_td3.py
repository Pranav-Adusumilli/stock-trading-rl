# smoke_test_td3.py
"""
Quick smoke test for the TD3 stack (data loader, env, models, small training loop).

- Runs a short 2000-step loop (random warmup + a few learning updates)
- Saves a small actor checkpoint to models_td3/smoke_actor.pth
- Prints basic stats to confirm shapes and gradients flow

Usage:
    python smoke_test_td3.py --config config_multi_td3.yaml
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim

from src.utils import load_config
from src.data_loader import fetch_multi_asset
from src.env_portfolio import PortfolioEnv
from src.models_td3 import Actor, Critic

# simple numpy replay buffer
class SimpleBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.ns = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.a = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.r = np.zeros((self.max_size,), dtype=np.float32)
        self.d = np.zeros((self.max_size,), dtype=np.float32)

    def store(self, s, a, r, ns, d):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.ns[self.ptr] = ns
        self.d[self.ptr] = 1.0 if d else 0.0
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return dict(
            state=self.s[idx],
            action=self.a[idx],
            reward=self.r[idx],
            next_state=self.ns[idx],
            done=self.d[idx],
        )

def to_torch(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)

def main(config_path, smoke_steps=2000, warmup=500, batch_size=64):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tickers = cfg["data"]["tickers"]
    df_map = fetch_multi_asset(tickers, cfg["data"]["start_date"], cfg["data"]["end_date"], cfg["data"].get("cache_dir","data/"), use_sentiment=cfg["data"].get("use_sentiment", True))
    env = PortfolioEnv(df_map,
                       window_size=cfg["env"]["window_size"],
                       initial_balance=cfg["env"]["initial_balance"],
                       transaction_cost=cfg["env"]["transaction_cost"],
                       reward_risk_lambda=cfg["env"]["reward_risk_lambda"],
                       sentiment_reward_lambda=cfg["env"].get("sentiment_reward_lambda", 0.0),
                       sentiment_reward_alpha=cfg["env"].get("sentiment_reward_alpha", 1.0),
                       normalize_actions=cfg["env"].get("normalize_actions", True))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print("State dim:", state_dim, "Action dim:", action_dim)

    actor = Actor(state_dim, action_dim, cfg["model"].get("actor_hidden",[256,256])).to(device)
    critic1 = Critic(state_dim, action_dim, cfg["model"].get("critic_hidden",[256,256])).to(device)
    critic2 = Critic(state_dim, action_dim, cfg["model"].get("critic_hidden",[256,256])).to(device)
    actor_t = Actor(state_dim, action_dim, cfg["model"].get("actor_hidden",[256,256])).to(device)
    critic1_t = Critic(state_dim, action_dim, cfg["model"].get("critic_hidden",[256,256])).to(device)
    critic2_t = Critic(state_dim, action_dim, cfg["model"].get("critic_hidden",[256,256])).to(device)

    actor_t.load_state_dict(actor.state_dict())
    critic1_t.load_state_dict(critic1.state_dict())
    critic2_t.load_state_dict(critic2.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
    critic1_opt = optim.Adam(critic1.parameters(), lr=1e-4)
    critic2_opt = optim.Adam(critic2.parameters(), lr=1e-4)

    buffer = SimpleBuffer(20000, state_dim, action_dim)

    gamma = 0.99
    tau = 0.005
    action_noise = 0.1
    target_noise = 0.2
    target_clip = 0.5
    policy_delay = 2

    # initialize
    s = env.reset()
    if isinstance(s, tuple): s = s[0]
    s = np.asarray(s, dtype=np.float32)
    step = 0
    ep = 0
    ep_r = 0.0

    start_time = time.time()
    while step < smoke_steps:
        if step < warmup:
            a = np.random.uniform(-1,1,size=(action_dim,))
        else:
            with torch.no_grad():
                a = actor(to_torch(s, device).unsqueeze(0)).cpu().numpy().flatten()
            a = a + np.random.normal(0, action_noise, size=a.shape)
            a = np.clip(a, -1, 1)

        ns, r, done, info = env.step(a)
        if isinstance(ns, tuple): ns = ns[0]
        ns = np.asarray(ns, dtype=np.float32)
        buffer.store(s, a, float(r), ns, done)

        s = ns
        ep_r += float(r)
        step += 1

        if done or step % 500 == 0:
            ep += 1
            print(f"[SMOKE] Step {step} Episode {ep} EpReward {ep_r:.2f} done={done}")
            s = env.reset()
            if isinstance(s, tuple): s = s[0]
            s = np.asarray(s, dtype=np.float32)
            ep_r = 0.0

        # learning
        if buffer.size >= batch_size and step >= warmup:
            batch = buffer.sample(batch_size)
            s_b = to_torch(batch["state"], device)
            a_b = to_torch(batch["action"], device)
            r_b = to_torch(batch["reward"], device).unsqueeze(1)
            ns_b = to_torch(batch["next_state"], device)
            d_b = to_torch(batch["done"], device).unsqueeze(1)

            # target actions
            with torch.no_grad():
                na = actor_t(ns_b)
                noise = (torch.randn_like(na) * target_noise).clamp(-target_clip, target_clip)
                na = (na + noise).clamp(-1,1)
                q1_t = critic1_t(ns_b, na)
                q2_t = critic2_t(ns_b, na)
                q_t = torch.min(q1_t, q2_t)
                q_target = r_b + (1.0 - d_b) * gamma * q_t

            q1 = critic1(s_b, a_b)
            q2 = critic2(s_b, a_b)
            loss1 = torch.nn.functional.mse_loss(q1, q_target)
            loss2 = torch.nn.functional.mse_loss(q2, q_target)

            critic1_opt.zero_grad(); loss1.backward(); critic1_opt.step()
            critic2_opt.zero_grad(); loss2.backward(); critic2_opt.step()

            if step % policy_delay == 0:
                actor_opt.zero_grad()
                actor_loss = -critic1(s_b, actor(s_b)).mean()
                actor_loss.backward()
                actor_opt.step()

                # soft update targets
                for tp, p in zip(actor_t.parameters(), actor.parameters()):
                    tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)
                for tp, p in zip(critic1_t.parameters(), critic1.parameters()):
                    tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)
                for tp, p in zip(critic2_t.parameters(), critic2.parameters()):
                    tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

    elapsed = time.time() - start_time
    print("SMOKE TEST COMPLETE | Steps:", step, "Time(s):", round(elapsed,2))

    # save a small checkpoint
    os.makedirs("models_td3", exist_ok=True)
    torch.save(actor.state_dict(), "models_td3/smoke_actor.pth")
    print("Saved models_td3/smoke_actor.pth")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_multi_td3.yaml")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    main(args.config, smoke_steps=args.steps)
