# src/env_wrappers.py
"""
Environment wrappers and compatibility layer.
Fully robust for Gym, Gymnasium, and custom wrapper chains.
"""

import gym
import numpy as np


class ResetCompatibilityWrapper(gym.Wrapper):
    """
    Ensures env.reset() returns (obs, info)
    even if underlying env returns more values.
    """

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)

        # res may be obs, (obs, info), (obs, info, extra...), etc.
        if not isinstance(res, tuple):
            return res, {}  # convert obs → (obs, {})

        obs = res[0]
        info = res[1] if len(res) > 1 and isinstance(res[1], dict) else {}
        return obs, info


class StepCompatibilityWrapper(gym.Wrapper):
    """
    Normalizes env.step(action) to return:
       (obs, reward, done, info)
    Handles Gym, Gymnasium, and custom wrapper outputs.
    """

    def step(self, action):
        out = self.env.step(action)

        if not isinstance(out, tuple):
            raise RuntimeError("env.step returned non-tuple result")

        # Standard Gym: (obs, reward, done, info)
        if len(out) == 4:
            obs, rew, done, info = out
            return obs, rew, bool(done), info

        # Gymnasium: (obs, reward, terminated, truncated, info)
        if len(out) == 5:
            obs, rew, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, rew, done, info

        # fallback: unknown format
        obs = out[0]
        rew = out[1] if len(out) > 1 else 0.0
        done = False
        info = {}

        for item in out[2:]:
            if isinstance(item, dict):
                info = item
            elif isinstance(item, (bool, np.bool_)):
                done = done or bool(item)

        return obs, rew, done, info


class ObsNormalizeWrapper(gym.ObservationWrapper):
    """
    Simple running-mean observation normalizer.
    """

    def __init__(self, env, eps=1e-8):
        super().__init__(env)
        self.eps = eps
        self.mean = None
        self.var = None
        self.count = 0

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32)

        if self.mean is None:
            self.mean = np.zeros_like(obs)
            self.var = np.ones_like(obs)

        self.count += 1
        alpha = 1.0 / self.count

        new_mean = (1 - alpha) * self.mean + alpha * obs
        new_var = (1 - alpha) * self.var + alpha * (obs - new_mean) ** 2

        self.mean = new_mean
        self.var = new_var

        return (obs - self.mean) / (np.sqrt(self.var) + self.eps)


def wrap_env(env, normalize_obs=False):
    """
    Builds the environment wrapper chain safely in correct order.
    """
    env = ResetCompatibilityWrapper(env)
    env = StepCompatibilityWrapper(env)

    if normalize_obs:
        env = ObsNormalizeWrapper(env)

    return env
