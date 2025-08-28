#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium.spaces import Dict as GymDict


def main():
    # Import within main to ensure project paths are correct
    from oanda_trading_bot.training_system.common.config import MAX_SYMBOLS_ALLOWED
    from oanda_trading_bot.training_system.agent.enhanced_feature_extractor import EnhancedTransformerFeatureExtractor
    from oanda_trading_bot.training_system.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
    from configs.training.enhanced_model_config import ModelConfig as EnhancedModelConfig

    B = 2
    N = MAX_SYMBOLS_ALLOWED
    F_market = 12
    F_context = 5

    # Build observation space like env
    obs_space = GymDict({
        "market_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(N, F_market), dtype=np.float32),
        "context_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(N, F_context), dtype=np.float32),
        # symbol_id high equals universe size; we mock universe size as N for padding id demonstration
        "symbol_id": gym.spaces.Box(low=0, high=N, shape=(N,), dtype=np.int32),
    })

    # Create dummy batch observations (B,N,*)
    rng = np.random.default_rng(42)
    market_features = th.tensor(rng.normal(size=(B, N, F_market)).astype(np.float32))
    context_features = th.tensor(rng.normal(size=(B, N, F_context)).astype(np.float32))
    # Active first 2 symbols; rest padded
    symbol_id = th.full((B, N), fill_value=N, dtype=th.long)
    for b in range(B):
        symbol_id[b, 0] = 3  # arbitrary valid ids
        symbol_id[b, 1] = 7

    obs = {
        "market_features": market_features,
        "context_features": context_features,
        "symbol_id": symbol_id,
    }

    # 1) Validate EnhancedTransformerFeatureExtractor masking and pooling
    fe = EnhancedTransformerFeatureExtractor(observation_space=obs_space, model_config=EnhancedModelConfig)
    feat = fe(obs)
    assert feat.shape[0] == B and feat.ndim == 2, f"Unexpected feature shape: {feat.shape}"
    print("[OK] EnhancedTransformerFeatureExtractor output shape:", tuple(feat.shape))

    # 2) Validate ESS consumption shapes (classical strategies input)
    # Build minimal ESS with no explicit strategies (registry will load defaults if any)
    ess = EnhancedStrategySuperposition(
        input_dim=feat.shape[1],
        num_strategies=2,
        strategy_config_file_path=None,
        use_gumbel_softmax=True,
        dropout_rate=0.1,
    )

    # Provide raw asset features [B,N,F_raw] → upcast to [B,N,1,F_raw] in SAC; we pass directly as [B,N,1,F_raw]
    raw_asset_features = market_features.unsqueeze(2)
    mean_actions = ess(
        asset_features_batch=raw_asset_features,
        market_state_features=feat,
        current_positions_batch=None,
        timestamps=None,
    )
    assert mean_actions.shape == (B, N, 1), f"Unexpected ESS output shape: {tuple(mean_actions.shape)}"
    print("[OK] ESS output shape:", tuple(mean_actions.shape))

    print("Validation passed.")


if __name__ == "__main__":
    # Ensure project root and src on path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_dir = os.path.join(project_root, 'src')
    for p in [project_root, src_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)
    main()
