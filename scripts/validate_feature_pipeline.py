import os
import sys
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium.spaces import Dict as GymDict

# Allow running from repo root
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(repo_root, "..", "src"))

from oanda_trading_bot.training_system.common.config import MAX_SYMBOLS_ALLOWED, TIMESTEPS
from oanda_trading_bot.training_system.agent.sac_policy import CustomSACPolicy
from oanda_trading_bot.training_system.agent.enhanced_feature_extractor import EnhancedTransformerFeatureExtractor


def main():
    th.manual_seed(0)
    np.random.seed(0)

    batch_size = 3
    num_symbols = MAX_SYMBOLS_ALLOWED
    timesteps = max(TIMESTEPS, 32)
    f_raw = 9
    ctx_dim = 5

    # Observation space matching env
    observation_space = GymDict({
        "market_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_symbols, f_raw), dtype=np.float32),
        "features_from_dataset": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_symbols, timesteps, f_raw), dtype=np.float32),
        "context_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_symbols, ctx_dim), dtype=np.float32),
        "symbol_id": gym.spaces.Box(low=0, high=num_symbols, shape=(num_symbols,), dtype=np.int32),
        "padding_mask": gym.spaces.MultiBinary(num_symbols)
    })

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_symbols,), dtype=np.float32)

    # Build policy with enhanced transformer extractor + ESS
    policy = CustomSACPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=EnhancedTransformerFeatureExtractor,
        features_extractor_kwargs={
            "model_config": {
                'hidden_dim': 128,
                'num_layers': 3,
                'num_heads': 4,
                'dropout_rate': 0.1,
                'output_dim': 64,
                'use_symbol_embedding': True,
            }
        },
        use_ess_layer=True,
        ess_config={
            "num_strategies": 8,
            "dropout_rate": 0.1,
            "initial_temperature": 1.0,
            "use_gumbel_softmax": True,
            "adaptive_learning_rate": 1e-3,
            "performance_ema_alpha": 0.1,
        }
    )

    # Synthetic observation batch
    obs = {
        "market_features": th.randn(batch_size, num_symbols, f_raw),
        "features_from_dataset": th.randn(batch_size, num_symbols, timesteps, f_raw),
        "context_features": th.randn(batch_size, num_symbols, ctx_dim),
        "symbol_id": th.tensor(np.arange(num_symbols)[None, :].repeat(batch_size, axis=0), dtype=th.int64),
        "padding_mask": th.zeros(batch_size, num_symbols, dtype=th.bool),
    }
    if num_symbols > 2:
        obs["padding_mask"][:, 2:] = True  # inactive slots

    # Run actor distribution params (will use ESS path)
    mean, log_std, _ = policy.actor.get_action_dist_params(obs)  # type: ignore

    print(f"mean.shape={tuple(mean.shape)} log_std.shape={tuple(log_std.shape)}")
    assert mean.shape == (batch_size, num_symbols)
    assert log_std.shape == (batch_size, num_symbols)

    print("Feature pipeline validation passed.")


if __name__ == "__main__":
    main()

