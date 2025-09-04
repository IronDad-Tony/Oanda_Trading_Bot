import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MiniDictTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, symbols: int = 5, feat: int = 8, seq: int = 16, ctx: int = 5, seed: int = 0):
        super().__init__()
        self.S = symbols
        self.F = feat
        self.T = seq
        self.C = ctx
        rng = np.random.RandomState(seed)
        self.trend = rng.normal(0, 0.01, size=(self.S,))
        self.step_idx = 0
        self.max_steps = 256

        self.observation_space = spaces.Dict({
            "market_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.S, self.F), dtype=np.float32),
            "features_from_dataset": spaces.Box(low=-np.inf, high=np.inf, shape=(self.S, self.T, self.F), dtype=np.float32),
            "context_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.S, self.C), dtype=np.float32),
            "symbol_id": spaces.Box(low=0, high=10000, shape=(self.S,), dtype=np.int32),
            "padding_mask": spaces.MultiBinary(self.S),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.S,), dtype=np.float32)

        self._state_seq = np.zeros((self.S, self.T, self.F), dtype=np.float32)
        self._last = np.zeros((self.S, self.F), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self._state_seq.fill(0.0)
        self._last.fill(0.0)
        return self._obs(), {}

    def step(self, action):
        self.step_idx += 1
        noise = np.random.normal(0, 0.01, size=(self.S, self.F)).astype(np.float32)
        drift = (self.trend[:, None] * np.linspace(0.9, 1.1, self.F)).astype(np.float32)
        self._last = 0.9 * self._last + noise + drift
        self._state_seq = np.roll(self._state_seq, shift=-1, axis=1)
        self._state_seq[:, -1, :] = self._last
        # Reward: align action sign with last feature slope
        slope = (self._state_seq[:, -1, 0] - self._state_seq[:, -2, 0]).astype(np.float32)
        reward = float(np.tanh(np.dot(action, slope)))
        terminated = False
        truncated = self.step_idx >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        ctx = np.zeros((self.S, self.C), dtype=np.float32)
        ctx[:, 0] = np.clip(self._state_seq[:, -1, 0], -1, 1)
        return {
            "market_features": self._state_seq[:, -1, :].astype(np.float32),
            "features_from_dataset": self._state_seq.astype(np.float32),
            "context_features": ctx,
            "symbol_id": np.arange(self.S, dtype=np.int32),
            "padding_mask": np.ones((self.S,), dtype=np.int8),
        }

