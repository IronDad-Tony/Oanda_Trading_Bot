import pytest
import torch
import torch.nn as nn # Added for MockActor
import pandas as pd
from gymnasium import spaces
import numpy as np
import os
import sys
import logging # Added for mock_logger_integration

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agent.enhanced_feature_extractor import EnhancedTransformerFeatureExtractor
from src.agent.strategies.statistical_arbitrage_strategies import (
    StatisticalArbitrageStrategy,
    MeanReversionStrategy,
    CointegrationStrategy
)
from src.agent.strategies.base_strategy import StrategyConfig
from src.common.config import DEVICE

# --- Test Constants ---
TEST_BATCH_SIZE_INTEGRATION = 2
TEST_SEQ_LEN_INTEGRATION = 60
# From environment/dataset
DATASET_NUM_FEATURES = 9 # e.g., Open, High, Low, Close, Vol, Signal1, Signal2, Signal3, Signal4
# For strategies
STRATEGY_NUM_FEATURES_PER_ASSET = 3 # e.g., HLC, or selected from DATASET_NUM_FEATURES

# From common.config (using values observed in files if direct import is complex for isolated test setup)
MAX_SYMBOLS_ALLOWED_INTEGRATION = 20 # Default seen in enhanced_feature_extractor.py
TRANSFORMER_OUTPUT_DIM_PER_SYMBOL_INTEGRATION = 128 # Default seen

# Asset names for the StatisticalArbitrageStrategy
STAT_ARB_ASSET_NAMES = ["SYM_A", "SYM_B", "SYM_C"] # Manage 3 assets
NUM_STAT_ARB_ASSETS = len(STAT_ARB_ASSET_NAMES)
MOCK_ACTION_DIM_PER_ASSET = 1 # e.g., a single continuous action per asset


@pytest.fixture
def mock_logger_integration(): # Defined a simple logger for this test file
    logger = logging.getLogger("IntegrationTestLogger")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

@pytest.fixture
def observation_space_integration():
    return spaces.Dict({
        "features_from_dataset": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(MAX_SYMBOLS_ALLOWED_INTEGRATION, TEST_SEQ_LEN_INTEGRATION, DATASET_NUM_FEATURES),
            dtype=np.float32
        ),
        "current_positions_nominal_ratio_ac": spaces.Box(
            low=-1.0, high=1.0, shape=(MAX_SYMBOLS_ALLOWED_INTEGRATION,), dtype=np.float32
        ),
        "unrealized_pnl_ratio_ac": spaces.Box(
            low=-1.0, high=1.0, shape=(MAX_SYMBOLS_ALLOWED_INTEGRATION,), dtype=np.float32
        ),
        "margin_level": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
        "padding_mask": spaces.Box(low=0, high=1, shape=(MAX_SYMBOLS_ALLOWED_INTEGRATION,), dtype=np.bool_)
    })

@pytest.fixture
def mock_observations_integration(observation_space_integration):
    obs = {
        key: torch.from_numpy(val.sample()).unsqueeze(0).repeat(TEST_BATCH_SIZE_INTEGRATION, *((1,) * len(val.shape)))
        for key, val in observation_space_integration.spaces.items()
    }
    # Ensure correct types and device
    obs["features_from_dataset"] = obs["features_from_dataset"].float().to(DEVICE)
    obs["current_positions_nominal_ratio_ac"] = obs["current_positions_nominal_ratio_ac"].float().to(DEVICE)
    obs["unrealized_pnl_ratio_ac"] = obs["unrealized_pnl_ratio_ac"].float().to(DEVICE)
    obs["margin_level"] = obs["margin_level"].float().to(DEVICE)
    obs["padding_mask"] = obs["padding_mask"].bool().to(DEVICE) # Ensure boolean
    return obs

@pytest.fixture
def statistical_arbitrage_strategy_config():
    mrs_params = MeanReversionStrategy.default_config().default_params
    mrs_params.update({
        "window": 20, "close_idx": 0, "num_features_per_asset": STRATEGY_NUM_FEATURES_PER_ASSET
    })

    cs_params = CointegrationStrategy.default_config().default_params
    cs_params.update({
        "window": 20, "asset1_close_idx": 0, "asset2_close_idx": 0,
        "num_features_per_asset": STRATEGY_NUM_FEATURES_PER_ASSET
    })
    
    stat_arb_default_params = {
        "num_features_per_asset": STRATEGY_NUM_FEATURES_PER_ASSET,
        "asset_names": STAT_ARB_ASSET_NAMES,
        "strategy_configs": [
            {
                "class_name": "MeanReversionStrategy",
                "params": mrs_params,
                "applicable_assets": [STAT_ARB_ASSET_NAMES[0]]
            },
            {
                "class_name": "CointegrationStrategy",
                "params": cs_params,
                "applicable_assets": [[STAT_ARB_ASSET_NAMES[1], STAT_ARB_ASSET_NAMES[2]]]
            }
        ],
        "combination_logic": "sum",
        "output_multiplier": 1.0
    }
    
    config = StrategyConfig(
        name="IntegrationTestStatArb",
        description="Config for StatArb in integration test.",
        default_params=stat_arb_default_params,
        applicable_assets=STAT_ARB_ASSET_NAMES
    )
    return config

# --- Mock Actor Network ---
class MockActor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh() # SAC actions are often in [-1, 1]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(state))
        x = self.tanh(self.fc2(x))
        return x

def test_sac_statistical_arbitrage_integration(
    mock_observations_integration,
    observation_space_integration,
    statistical_arbitrage_strategy_config,
    mock_logger_integration
):
    # 1. Instantiate EnhancedTransformerFeatureExtractor
    # It expects enhanced_transformer_output_dim_per_symbol in constructor
    feature_extractor = EnhancedTransformerFeatureExtractor(
        observation_space_integration,
        enhanced_transformer_output_dim_per_symbol=TRANSFORMER_OUTPUT_DIM_PER_SYMBOL_INTEGRATION
    )
    feature_extractor.to(DEVICE)
    feature_extractor.eval() # Set to eval mode

    # 2. Get base features from the extractor
    with torch.no_grad():
        base_extracted_features = feature_extractor(mock_observations_integration)

    expected_base_features_dim = (MAX_SYMBOLS_ALLOWED_INTEGRATION * TRANSFORMER_OUTPUT_DIM_PER_SYMBOL_INTEGRATION) + \
                                 (MAX_SYMBOLS_ALLOWED_INTEGRATION * 3) + 1 # pos, pnl, padding_mask_float + margin
    assert base_extracted_features.shape == (TEST_BATCH_SIZE_INTEGRATION, expected_base_features_dim)

    # 3. Instantiate StatisticalArbitrageStrategy
    stat_arb_strategy = StatisticalArbitrageStrategy(
        statistical_arbitrage_strategy_config,
        statistical_arbitrage_strategy_config.default_params, # instance_params
        mock_logger_integration
    )
    stat_arb_strategy.to(DEVICE)
    stat_arb_strategy.eval()

    # 4. Prepare input for StatisticalArbitrageStrategy
    # Mocking the selection of relevant assets and features from the larger observation tensor.
    # For this test, assume the first NUM_STAT_ARB_ASSETS in MAX_SYMBOLS_ALLOWED_INTEGRATION
    # correspond to STAT_ARB_ASSET_NAMES.
    # And the first STRATEGY_NUM_FEATURES_PER_ASSET in DATASET_NUM_FEATURES are used.

    # (batch_size, MAX_SYMBOLS_ALLOWED, seq_len, DATASET_NUM_FEATURES)
    all_asset_features_from_dataset = mock_observations_integration["features_from_dataset"]

    # Select features for the assets managed by StatArb
    # -> (batch_size, NUM_STAT_ARB_ASSETS, seq_len, DATASET_NUM_FEATURES)
    stat_arb_asset_features_raw = all_asset_features_from_dataset[:, :NUM_STAT_ARB_ASSETS, :, :]
    
    # Select the specific features needed by the strategy
    # -> (batch_size, NUM_STAT_ARB_ASSETS, seq_len, STRATEGY_NUM_FEATURES_PER_ASSET)
    stat_arb_asset_features_selected = stat_arb_asset_features_raw[:, :, :, :STRATEGY_NUM_FEATURES_PER_ASSET]

    # Reshape for StatArb: (batch_size, seq_len, NUM_STAT_ARB_ASSETS * STRATEGY_NUM_FEATURES_PER_ASSET)
    # Permute: (batch_size, seq_len, NUM_STAT_ARB_ASSETS, STRATEGY_NUM_FEATURES_PER_ASSET)
    stat_arb_input_features = stat_arb_asset_features_selected.permute(0, 2, 1, 3).reshape(
        TEST_BATCH_SIZE_INTEGRATION,
        TEST_SEQ_LEN_INTEGRATION,
        NUM_STAT_ARB_ASSETS * STRATEGY_NUM_FEATURES_PER_ASSET
    )
    assert stat_arb_input_features.shape == (
        TEST_BATCH_SIZE_INTEGRATION,
        TEST_SEQ_LEN_INTEGRATION,
        NUM_STAT_ARB_ASSETS * STRATEGY_NUM_FEATURES_PER_ASSET
    )

    # Prepare current_positions for StatArb
    # (batch_size, MAX_SYMBOLS_ALLOWED_INTEGRATION)
    all_current_positions = mock_observations_integration["current_positions_nominal_ratio_ac"]
    # -> (batch_size, NUM_STAT_ARB_ASSETS)
    stat_arb_current_positions = all_current_positions[:, :NUM_STAT_ARB_ASSETS]
    
    assert stat_arb_current_positions.shape == (TEST_BATCH_SIZE_INTEGRATION, NUM_STAT_ARB_ASSETS)

    # 5. Run StatisticalArbitrageStrategy.forward()
    with torch.no_grad():
        strategy_signals = stat_arb_strategy.forward(stat_arb_input_features, stat_arb_current_positions)

    assert strategy_signals.shape == (TEST_BATCH_SIZE_INTEGRATION, NUM_STAT_ARB_ASSETS)

    # 6. Mock Concatenation of base features and strategy signals
    flat_strategy_signals = strategy_signals # Shape is (batch, NUM_STAT_ARB_ASSETS)

    combined_features = torch.cat([base_extracted_features, flat_strategy_signals], dim=1)

    expected_combined_dim = expected_base_features_dim + NUM_STAT_ARB_ASSETS
    assert combined_features.shape == (TEST_BATCH_SIZE_INTEGRATION, expected_combined_dim)
    mock_logger_integration.info(f"Combined features created with shape: {combined_features.shape}")

    # 7. Pass combined features to a Mock Actor (simulating SAC policy's actor network)
    # The SAC agent's action space is typically related to MAX_SYMBOLS_ALLOWED_INTEGRATION
    # For this test, let's assume the actor outputs an action for each of the MAX_SYMBOLS_ALLOWED_INTEGRATION
    mock_actor_output_dim = MAX_SYMBOLS_ALLOWED_INTEGRATION * MOCK_ACTION_DIM_PER_ASSET
    
    mock_actor = MockActor(input_dim=expected_combined_dim, action_dim=mock_actor_output_dim)
    mock_actor.to(DEVICE)
    mock_actor.eval()

    with torch.no_grad():
        actions = mock_actor(combined_features)

    assert actions.shape == (TEST_BATCH_SIZE_INTEGRATION, mock_actor_output_dim)
    mock_logger_integration.info(f"Mock actor produced actions with shape: {actions.shape}")

    mock_logger_integration.info(f"Integration test passed. Combined features shape: {combined_features.shape}, Actor output shape: {actions.shape}")

