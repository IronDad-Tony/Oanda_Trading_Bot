import os
import sys
import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Dict as GymDict

# Allow running from repo root
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(repo_root, "..", "src"))

from oanda_trading_bot.training_system.common.config import MAX_SYMBOLS_ALLOWED, TIMESTEPS
from oanda_trading_bot.training_system.agent.transformer_feature_extractor import TransformerFeatureExtractor
from oanda_trading_bot.training_system.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    batch_size = 2
    num_symbols = MAX_SYMBOLS_ALLOWED
    timesteps = max(TIMESTEPS, 32)
    feature_dim = 9

    # Build observation space compatible with our env and extractor
    observation_space = GymDict({
        "features_from_dataset": gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_symbols, timesteps, feature_dim), dtype=np.float32
        ),
        "padding_mask": gym.spaces.MultiBinary(num_symbols),
    })

    # Create feature extractor (time-aware transformer)
    extractor = TransformerFeatureExtractor(
        observation_space=observation_space,
        features_key="features_from_dataset",
        mask_key="padding_mask",
        transformer_kwargs={}
    )

    # Build synthetic batch
    obs = {
        "features_from_dataset": torch.randn(batch_size, num_symbols, timesteps, feature_dim),
        "padding_mask": torch.zeros(batch_size, num_symbols, dtype=torch.int8),
    }
    # Only first 2 symbols active; rest masked as dummy
    if num_symbols > 2:
        obs["padding_mask"][:, 2:] = 1  # 1 means dummy in env; extractor flips to True for masking

    # Run transformer extractor
    with torch.no_grad():
        flat_feats = extractor(obs)  # [B, S*D]

    print(f"TransformerFeatureExtractor output shape: {tuple(flat_feats.shape)}")
    assert flat_feats.shape[0] == batch_size
    assert flat_feats.shape[1] > 0 and flat_feats.shape[1] % num_symbols == 0

    # Setup EnhancedStrategySuperposition (advisor layer)
    ess = EnhancedStrategySuperposition(
        input_dim=flat_feats.shape[1],
        num_strategies=8,
        strategy_config_file_path=None,
        use_gumbel_softmax=True,
        dropout_rate=0.1,
        adaptive_learning_rate=0.001,
        performance_ema_alpha=0.1,
    )

    # Strategy inputs expect [B, S, T, F]
    asset_features_batch = obs["features_from_dataset"]

    # Run ESS forward pass
    with torch.no_grad():
        actions = ess(
            asset_features_batch=asset_features_batch,
            market_state_features=flat_feats,
            current_positions_batch=None,
            timestamps=None,
            external_weights=None,
        )

    print(f"ESS output actions shape: {tuple(actions.shape)}")
    assert actions.shape == (batch_size, num_symbols, 1)

    # Masked symbols should produce some output; env will ignore them in execution
    # Verify first two symbols differ from zeros on average (not a strict test, just sanity)
    active_mean = actions[:, :2, :].abs().mean().item()
    print(f"Mean abs action (first 2 symbols): {active_mean:.6f}")

    print("Validation completed successfully.")


if __name__ == "__main__":
    main()

