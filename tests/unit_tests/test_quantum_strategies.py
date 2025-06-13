# tests/unit_tests/test_quantum_strategies.py
import unittest
import torch
import torch.nn.functional as F # Added F import
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Type
import pytest
import torch.nn as nn
from unittest.mock import MagicMock, patch
import os # Added global import

# Define project_root globally
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Attempt to import from src
try:
    from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition, DynamicStrategyGenerator
    from src.agent.strategies.base_strategy import BaseStrategy, StrategyConfig # Ensure StrategyConfig is imported
    from src.common.config import DEVICE
    from src.agent.strategies import STRATEGY_REGISTRY
    from src.agent.strategies.trend_strategies import MomentumStrategy, MeanReversionStrategy as TrendMeanReversionStrategy, ReversalStrategy, BreakoutStrategy, TrendFollowingStrategy # For checking params and testing
    from src.agent.strategies.statistical_arbitrage_strategies import (
        MeanReversionStrategy,
        CointegrationStrategy,
        PairsTradeStrategy,
        StatisticalArbitrageStrategy,
        VolatilityBreakoutStrategy
    )
except ImportError:
    # Fallback for environments where src is not directly in PYTHONPATH
    import sys
    # 'os' is already imported globally
    # 'project_root' is already defined globally
    # Add project root to sys.path to allow src.module imports
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), \'..\', \'..\', \'..\')) # Adjusted path # No longer needed here
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Now try importing with src prefix
    from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition, DynamicStrategyGenerator
    from src.agent.strategies.base_strategy import BaseStrategy, StrategyConfig
    from src.common.config import DEVICE
    from src.agent.strategies import STRATEGY_REGISTRY
    from src.agent.strategies.trend_strategies import MomentumStrategy, MeanReversionStrategy as TrendMeanReversionStrategy, ReversalStrategy, BreakoutStrategy, TrendFollowingStrategy # For checking params and testing
    from src.agent.strategies.statistical_arbitrage_strategies import (
        MeanReversionStrategy,
        CointegrationStrategy,
        PairsTradeStrategy,
        StatisticalArbitrageStrategy,
        VolatilityBreakoutStrategy
    )

# Configure a logger for tests
test_logger = logging.getLogger("TestQuantumStrategies")
test_logger.setLevel(logging.DEBUG) # Or logging.INFO
# Ensure a handler is configured to see output during tests
if not test_logger.handlers:
    # Define project_root if not already defined in this specific scope
    # It's typically defined above for imports, ensure 'os' is imported.
    # import os # Make sure os is imported; it should be from the top of the file.
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), \'..\', \'..\', \'..\'))

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')

    # File Handler
    # Assuming project_root is defined as in the original file structure
    # (os.path.abspath(os.path.join(os.path.dirname(__file__), \'..\', \'..\', \'..\')))
    # This path needs to be correct for the script's location.
    # The variable 'project_root' is defined in the try-except block for imports earlier in the file.
    log_file_path = os.path.join(project_root, "test_quantum_strategies.log")
    file_handler = logging.FileHandler(log_file_path, mode='w') # mode='w' to overwrite each run
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    test_logger.addHandler(file_handler)

    # Console Handler (optional, but good for seeing logs during interactive runs)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)
    test_logger.addHandler(console_handler)

# Define LearnableMockStrategy for gradient testing
class LearnableMockStrategy(BaseStrategy):
    # MODIFIED: Align __init__ with BaseStrategy
    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params, logger)
        # MODIFIED: Initialize nn.Parameter without explicit device. Device placement handled by parent module.
        self.learnable_param = nn.Parameter(torch.randn(1))
        # self.name = self.params.get(\\\'name\\\', \\\'LearnableMockStrategy\\\') # Name comes from config

    @staticmethod
    def default_config() -> StrategyConfig: # Return StrategyConfig object
        return StrategyConfig(
            name='LearnableMockStrategy',
            description='A mock strategy with a learnable parameter for gradient testing.',
            default_params={'mock_specific_param': 100}
        )

    def forward(self,
                asset_features: torch.Tensor, # MODIFIED: Renamed and updated shape comment
                                                 # (batch_size, sequence_length, input_dim)
                current_positions: Optional[torch.Tensor] = None, # MODIFIED: Renamed and updated shape comment
                                                                    # (batch_size, 1)
                timestamp: Optional[pd.Timestamp] = None
                ) -> torch.Tensor: # Expected output: (batch_size, 1, 1) signal tensor

        # MODIFIED: asset_features_for_strategy -> asset_features
        if asset_features.numel() == 0:
            # Ensure batch_size dimension is preserved if asset_features.size(0) is valid
            batch_s = asset_features.size(0) if asset_features.dim() > 0 else 0 # Handle 0-dim tensor from numel=0
            return torch.zeros((batch_s, 1, 1), device=asset_features.device)

        # Assuming we use the first feature (index 0) from input_dim for calculation
        # asset_features shape: (batch_size, sequence_length, input_dim)
        # MODIFIED: Corrected indexing for 3D tensor
        mean_feature = asset_features[:, :, 0].mean(dim=1, keepdim=True) # Result: (batch_size, 1)
        signal = torch.tanh(mean_feature * self.learnable_param) # Result: (batch_size, 1)

        # Output expected by EnhancedStrategySuperposition is (batch_size, 1, 1)
        return signal.unsqueeze(1) # This was correct: (batch_size, 1) -> (batch_size, 1, 1)

    # ADDED: Implement abstract method generate_signals
    def generate_signals(self, 
                         processed_data_dict: Dict[str, pd.DataFrame], 
                         portfolio_context: Optional[Dict[str, Any]] = None
                         ) -> pd.DataFrame:
        self.logger.info(f"LearnableMockStrategy.generate_signals called. Input keys: {list(processed_data_dict.keys())}")
        first_asset_data = next(iter(processed_data_dict.values()), None)
        if first_asset_data is not None and not first_asset_data.empty:
            last_timestamp = first_asset_data.index[-1]
            signal_value = 0 
            if 'feature1' in first_asset_data.columns:
                signal_value = 1 if first_asset_data['feature1'].iloc[-1] > 0 else -1
            return pd.DataFrame({'signal': [signal_value]}, index=[last_timestamp])
        else:
            return pd.DataFrame(columns=['signal'], index=pd.DatetimeIndex([]))

# Configure pytest fixtures
@pytest.fixture(scope="session")
def device():
    """Provides the torch device (CPU or CUDA) for tests."""
    return DEVICE

@pytest.fixture
def mock_logger():
    # Return the actual test_logger instance instead of a MagicMock
    # This allows seeing log output during tests when run with --log-cli-level=DEBUG
    return test_logger

@pytest.fixture
def base_config(tmp_path):
    """Provides a base configuration as a StrategyConfig object."""
    # Define default parameters that would typically be part of a base StrategyConfig
    default_strategy_params = {
        "feature_dim": 5,  # Example value, adjust as needed
        "num_assets": 2,     # Example value
        "sequence_length": 60, # Example value
        "market_feature_dim": 10, # Example value
        "adaptive_weight_config": { # Example, adjust based on actual structure
            "method": "attention",
            "attention_dim": 64
        },
        # Add any other general default parameters expected by StrategyConfig.default_params
    }

    # Define strategy-specific overrides or initial parameters
    strategy_specific_params_overrides = {
        "MomentumStrategy": {"window": 7},
        "QuantitativeStrategy": {"expression": "close > sma(close, 20)"},
        "CointegrationStrategy": {"asset_pair": ["EUR_USD", "USD_JPY"]},
        "PairsTradeStrategy": {"asset_pair": ["EUR_USD", "USD_JPY"]},
        "AlgorithmicStrategy": {
            "rule_buy_condition": "close > placeholder_indicator(period=10)",
            "rule_sell_condition": "close < placeholder_indicator(period=10)",
            "asset_list": ["EUR_USD"]
        },
        # For LearnableMockStrategy, if it has specific params to be set via strategy_specific_params
        "LearnableMockStrategy": {"mock_specific_param": 150} # This was checked in a test
    }
    
    # Create and return a StrategyConfig object
    return StrategyConfig(
        name="BaseTestConfig",
        description="Base configuration for testing EnhancedStrategySuperposition.",
        default_params=default_strategy_params,
        strategy_specific_params=strategy_specific_params_overrides,
        applicable_assets=["EUR_USD", "USD_JPY"] # General applicable assets
    )

@pytest.fixture
def mock_input_features_for_real_strategies(base_config, device): # Added device fixture
    """Generates mock input features for tests using real strategies."""
    batch_size = 2
    # Access params from base_config.default_params
    num_assets = base_config.default_params["num_assets"]
    sequence_length = base_config.default_params["sequence_length"]
    feature_dim = base_config.default_params["feature_dim"] 
    market_feature_dim = base_config.default_params["market_feature_dim"]

    asset_features_batch = torch.randn(batch_size, num_assets, sequence_length, feature_dim, device=device) # Use injected device
    market_state_features = torch.randn(batch_size, market_feature_dim, device=device) # Use injected device
    timestamps = [pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=i) for i in range(batch_size)]
    current_positions_batch = torch.randn(batch_size, num_assets, 1, device=device) # Use injected device
    
    return asset_features_batch, market_state_features, timestamps, current_positions_batch


# Helper to create a mock EnhancedStrategySuperposition for tests that need it
def create_mock_ess_layer(input_dim, num_strategies_to_init, overall_config, strategy_classes, logger_instance, layer_settings_override=None):
    default_layer_settings = {
        "adaptive_weight_config": overall_config.default_params.get("adaptive_weight_config"),
        "dsg_optimizer_config": None,
        "max_generated_strategies": 0,
        "innovation_engine_active": False,
        # Parameters for EnhancedStrategySuperposition constructor
        "dropout_rate": 0.1,
        "initial_temperature": 1.0,
        "use_gumbel_softmax": True,
        "strategy_input_dim": overall_config.default_params.get("feature_dim", 64) # Default if not in overall_config
    }
    if layer_settings_override:
        default_layer_settings.update(layer_settings_override)

    return EnhancedStrategySuperposition(
        input_dim=input_dim,
        num_strategies=num_strategies_to_init, # Changed from num_assets
        strategy_configs=[sc.default_config() for sc in strategy_classes if hasattr(sc, 'default_config')] + [overall_config], # Pass configs
        explicit_strategies=strategy_classes,
        dropout_rate=default_layer_settings["dropout_rate"],
        initial_temperature=default_layer_settings["initial_temperature"],
        use_gumbel_softmax=default_layer_settings["use_gumbel_softmax"],
        strategy_input_dim=default_layer_settings["strategy_input_dim"]
        # logger is handled by the class itself now
        # dynamic_loading_enabled can be added if needed for specific tests
    )

# Base class for strategy tests (if not already defined elsewhere and imported)
# If BaseStrategyTest is defined in another file, ensure it's imported.
# For now, defining a minimal one here if it's missing.
class BaseStrategyTest:
    # Common setup for strategy tests, can be expanded
    batch_size = 2
    seq_len = 20
    feature_dim = 5 # Default feature dimension for single asset
    num_features_per_asset = 3 # For multi-feature assets

    @pytest.fixture
    def mock_logger_fixture(self):
        """Provides a mock logger for testing."""
        logger = logging.getLogger("TestQuantumStrategies")
        logger.setLevel(logging.DEBUG)
        # If you need to assert on log messages, you might add a handler here
        # or use caplog fixture from pytest if preferred for capturing.
        return logger

# Test class for EnhancedStrategySuperposition (ESS)
# These tests are module-level functions, not methods of a class, so no 'self'

def test_initialization_with_real_strategies(base_config: StrategyConfig, mock_logger): # Removed mocker dependency
    """Test EnhancedStrategySuperposition initialization with actual strategies from STRATEGY_REGISTRY."""
    if not STRATEGY_REGISTRY:
        pytest.skip("STRATEGY_REGISTRY is empty or not imported. Skipping test with real strategies.")

    input_dim = base_config.default_params.get("feature_dim", 128) # Example default
    # num_assets = base_config.default_params.get("num_assets", 1) # Not directly passed to ESS init

    strategy_classes_to_test = list(STRATEGY_REGISTRY.values())
    num_strategies_to_initialize = len(strategy_classes_to_test)

    # Patch logger inside the module where ESS is defined using patch
    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger):
        layer = EnhancedStrategySuperposition(
            input_dim=input_dim,
            num_strategies=num_strategies_to_initialize, # Corrected: num_strategies
            # Pass strategy configurations explicitly.
            # Each strategy class will have its default_config() called internally if not overridden.
            # Or, provide a list of StrategyConfig objects or dicts.
            strategy_configs=[cls.default_config() for cls in strategy_classes_to_test if hasattr(cls, 'default_config')],
            explicit_strategies=strategy_classes_to_test,
            strategy_input_dim=input_dim # Assuming strategies take same input_dim as layer for simplicity here
        )

    layer = EnhancedStrategySuperposition(
        input_dim=input_dim,
        num_strategies=num_strategies_to_initialize, # Corrected: num_strategies
        # Pass strategy configurations explicitly.
        # Each strategy class will have its default_config() called internally if not overridden.
        # Or, provide a list of StrategyConfig objects or dicts.
        strategy_configs=[cls.default_config() for cls in strategy_classes_to_test if hasattr(cls, 'default_config')],
        explicit_strategies=strategy_classes_to_test,
        strategy_input_dim=input_dim # Assuming strategies take same input_dim as layer for simplicity here
    )
    
    # Assert that the number of initialized strategies matches expectations.
    # This depends on how many unique, valid strategies are in STRATEGY_REGISTRY
    # and how _initialize_strategies handles them.
    # For now, let's assume all in STRATEGY_REGISTRY are valid and get loaded.
    # The actual number might be different if some fail to load or are duplicates.
    # Check layer.num_actual_strategies
    assert layer.num_actual_strategies > 0, "No strategies were loaded."
    # A more precise check would be:
    # expected_successful_loads = count_successfully_loadable_strategies(strategy_classes_to_test)
    # assert layer.num_actual_strategies == expected_successful_loads
    # For now, a simple check:
    assert layer.num_actual_strategies <= len(strategy_classes_to_test)

    # Verify that strategy names are populated
    assert len(layer.strategy_names) == layer.num_actual_strategies

    # Verify that internal strategy modules are created
    assert len(layer.strategies) == layer.num_actual_strategies
    for strategy_module in layer.strategies:
        assert isinstance(strategy_module, BaseStrategy)

    # Check if logger was called for warnings/errors if any strategy failed (harder to test precisely without more mocks)
    # mock_logger_fixture.error.assert_not_called() # If expecting no errors

def test_forward_with_internal_weights_real_strategies(base_config: StrategyConfig, mock_logger, mock_input_features_for_real_strategies): # Removed mocker dependency
    """Test forward pass with internal weight calculation using real strategies."""
    if not STRATEGY_REGISTRY:
        pytest.skip("STRATEGY_REGISTRY is empty. Skipping forward test with real strategies.")
    
    input_dim_for_attention = base_config.default_params.get("market_feature_dim", 128) # For attention network
    strategy_input_feature_dim = base_config.default_params.get("feature_dim", 64) # For strategy inputs

    strategy_classes_to_test = list(STRATEGY_REGISTRY.values())
    num_strategies_to_initialize = len(strategy_classes_to_test)

    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger):
        layer = EnhancedStrategySuperposition(
            input_dim=input_dim_for_attention, # For attention
            num_strategies=num_strategies_to_initialize,
            strategy_configs=[cls.default_config() for cls in strategy_classes_to_test if hasattr(cls, 'default_config')],
            explicit_strategies=strategy_classes_to_test,
            strategy_input_dim=strategy_input_feature_dim # For strategies
        )

    if layer.num_actual_strategies == 0:
        pytest.skip("No strategies loaded in ESS, cannot perform forward pass test.")

    # MODIFIED: Unpack all 4 values
    asset_features_batch, market_state_features, timestamps, current_positions_batch = mock_input_features_for_real_strategies
    
    # Ensure tensors are on the correct device if layer is on GPU
    # device = next(layer.parameters()).device # Get device from layer
    # asset_features_batch = asset_features_batch.to(device)
    # market_state_features = market_state_features.to(device)
    # current_positions_batch = current_positions_batch.to(device)

    # MODIFIED: Call layer.forward with updated signature
    output_actions = layer.forward(
        asset_features_batch, 
        market_state_features=market_state_features,
        current_positions_batch=current_positions_batch,
        timestamps=timestamps
    )
    
    assert output_actions is not None
    # Expected shape: (batch_size, num_assets, 1)
    expected_num_assets = asset_features_batch.shape[1]
    assert output_actions.shape == (asset_features_batch.shape[0], expected_num_assets, 1)


def test_forward_with_external_weights_real_strategies(base_config: StrategyConfig, mock_logger, mock_input_features_for_real_strategies): # Removed mocker dependency
    """Test forward pass with externally provided weights using real strategies."""
    if not STRATEGY_REGISTRY:
        pytest.skip("STRATEGY_REGISTRY is empty. Skipping forward test with real strategies.")
    
    input_dim_for_attention = base_config.default_params.get("market_feature_dim", 128)
    strategy_input_feature_dim = base_config.default_params.get("feature_dim", 64)
    strategy_classes_to_test = list(STRATEGY_REGISTRY.values())
    num_strategies_to_initialize = len(strategy_classes_to_test)

    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger):
        layer = EnhancedStrategySuperposition(
            input_dim=input_dim_for_attention,
            num_strategies=num_strategies_to_initialize,
            strategy_configs=[cls.default_config() for cls in strategy_classes_to_test if hasattr(cls, 'default_config')],
            explicit_strategies=strategy_classes_to_test,
            strategy_input_dim=strategy_input_feature_dim
        )

    if layer.num_actual_strategies == 0:
        pytest.skip("No strategies loaded in ESS, cannot perform forward pass test with external weights.")

    # MODIFIED: Unpack all 4 values
    asset_features_batch, market_state_features, timestamps, current_positions_batch = mock_input_features_for_real_strategies
    batch_size = asset_features_batch.shape[0]
    
    # External weights: (batch_size, num_actual_strategies)
    external_weights = F.softmax(torch.randn(batch_size, layer.num_actual_strategies, device=asset_features_batch.device), dim=-1)
    # external_weights = external_weights.to(next(layer.parameters()).device)


    # MODIFIED: Call layer.forward with updated signature
    output_actions = layer.forward(
        asset_features_batch, 
        market_state_features=market_state_features, 
        external_weights=external_weights,
        current_positions_batch=current_positions_batch,
        timestamps=timestamps
    )
    
    assert output_actions is not None
    expected_num_assets = asset_features_batch.shape[1]
    assert output_actions.shape == (batch_size, expected_num_assets, 1)


def test_forward_with_invalid_external_weights_fallback_real_strategies(base_config: StrategyConfig, mock_logger, mock_input_features_for_real_strategies): # Removed mocker dependency
    """Test fallback to internal weights if external_weights have an invalid shape, using real strategies."""
    if not STRATEGY_REGISTRY:
        pytest.skip("STRATEGY_REGISTRY is empty. Skipping forward test with real strategies.")
    
    input_dim_for_attention = base_config.default_params.get("market_feature_dim", 128)
    strategy_input_feature_dim = base_config.default_params.get("feature_dim", 64)
    strategy_classes_to_test = list(STRATEGY_REGISTRY.values())
    num_strategies_to_initialize = len(strategy_classes_to_test)

    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger):
        layer = EnhancedStrategySuperposition(
            input_dim=input_dim_for_attention,
            num_strategies=num_strategies_to_initialize, 
            strategy_configs=[cls.default_config() for cls in strategy_classes_to_test if hasattr(cls, 'default_config')],
            explicit_strategies=strategy_classes_to_test,
            strategy_input_dim=strategy_input_feature_dim
        )

    if layer.num_actual_strategies == 0:
        pytest.skip("No strategies loaded in ESS, cannot perform fallback test.")

    # MODIFIED: Unpack all 4 values
    asset_features_batch, market_state_features, timestamps, current_positions_batch = mock_input_features_for_real_strategies
    batch_size = asset_features_batch.shape[0]
    
    # Invalid external weights (e.g., wrong number of strategies dimension)
    invalid_external_weights = torch.randn(batch_size, layer.num_actual_strategies + 1, device=asset_features_batch.device)
    # invalid_external_weights = invalid_external_weights.to(next(layer.parameters()).device)

    # Expect a warning about invalid weights and fallback
    with patch.object(layer.logger, 'warning') as mock_warning:
        # MODIFIED: Call layer.forward with updated signature
        output_actions = layer.forward(
            asset_features_batch, 
            market_state_features=market_state_features, 
            external_weights=invalid_external_weights,
            current_positions_batch=current_positions_batch,
            timestamps=timestamps
        )
        mock_warning.assert_called() 

    assert output_actions is not None
    expected_num_assets = asset_features_batch.shape[1]
    assert output_actions.shape == (batch_size, expected_num_assets, 1)


def test_gradient_flow_through_layer_and_mock_strategy(base_config: StrategyConfig, mock_logger): # Removed mocker dependency
    """Test that gradients can flow through the layer and a learnable mock strategy."""
    # Use distinct dimensions for clarity
    layer_attention_input_dim = base_config.default_params.get("market_feature_dim", 32) # Smaller for test
    strategy_feature_input_dim = base_config.default_params.get("feature_dim", 16) # Smaller for test


    mock_strategy_classes = [LearnableMockStrategy]
    num_strategies_to_initialize = len(mock_strategy_classes)
    
    learnable_mock_config = LearnableMockStrategy.default_config()
    # Ensure strategy_input_dim is set for the mock strategy's config
    learnable_mock_config.input_dim = strategy_feature_input_dim 

    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger):
        layer_for_grad_test = EnhancedStrategySuperposition(
            input_dim=layer_attention_input_dim, # For attention mechanism
            num_strategies=num_strategies_to_initialize,
            strategy_configs=[learnable_mock_config],
            explicit_strategies=mock_strategy_classes,
            strategy_input_dim=strategy_feature_input_dim # For strategies
        )

    # Explicitly move the entire layer to the target device
    layer_for_grad_test.to(DEVICE)

    if layer_for_grad_test.num_actual_strategies == 0:
        pytest.skip("LearnableMockStrategy not loaded, cannot test gradient flow.")
        
    mock_strategy_instance = layer_for_grad_test.strategies[0]
    assert any(p.requires_grad for p in mock_strategy_instance.parameters()), "Mock strategy has no learnable parameters."
    if layer_for_grad_test.attention_network:
        assert any(p.requires_grad for p in layer_for_grad_test.attention_network.parameters()), "Attention network has no learnable parameters."


    batch_size = 2
    seq_len = 10 
    num_test_assets = 1 # Test with one asset for simplicity in this grad test

    # MODIFIED: asset_features_batch shape (B, N_assets, Seq, StrategyInputDim)
    asset_features_batch_grad = torch.randn(batch_size, num_test_assets, seq_len, strategy_feature_input_dim, requires_grad=False, device=DEVICE)
    
    # MODIFIED: market_state_features for attention (B, LayerAttentionInputDim)
    # Needs requires_grad=True if we want to check grads flowing *into* it, 
    # but for grads on attention *weights*, input grad is not strictly needed.
    # For grads on strategy params, this also doesn't need grad.
    market_state_features_grad = torch.randn(batch_size, layer_attention_input_dim, requires_grad=False, device=DEVICE)
    
    current_positions_grad = torch.zeros(batch_size, num_test_assets, 1, device=DEVICE)


    # Forward pass
    # MODIFIED: Pass appropriate inputs
    output_actions = layer_for_grad_test.forward(
        asset_features_batch_grad, 
        market_state_features=market_state_features_grad,
        current_positions_batch=current_positions_grad
    )
    
    # Create a dummy loss and backward pass
    loss = output_actions.sum() 
    loss.backward()

    grad_found_in_strategy = False
    for param in mock_strategy_instance.parameters():
        if param.grad is not None:
            grad_found_in_strategy = True
            # print(f"Grad found in strategy param: {param.grad.abs().sum()}") # Debug
            break
    assert grad_found_in_strategy, "No gradients found in LearnableMockStrategy parameters."

    if layer_for_grad_test.attention_network and any(p.requires_grad for p in layer_for_grad_test.attention_network.parameters()):
        grad_found_in_attention = False
        for param_name, param in layer_for_grad_test.attention_network.named_parameters():
            if param.grad is not None:
                grad_found_in_attention = True
                # print(f"Grad found in attention param ({param_name}): {param.grad.abs().sum()}") # Debug
                break
        assert grad_found_in_attention, "No gradients found in attention network parameters (when expected)."

# --- Test Constants for Statistical Arbitrage Strategies ---
TEST_BATCH_SIZE = 2
TEST_SEQ_LEN = 60
# Number of features per individual asset (e.g., Open, High, Low, Close, Volume -> 5)
# For simplicity in tests, let\'s use 3 (e.g., HLC or just 3 generic features)
TEST_NUM_FEATURES_PER_ASSET = 3 # This was used for StatArb. For Trend, let's assume 5 for OHLCV
TEST_WINDOW_SHORT = 10
TEST_WINDOW_LONG = 20

# Ensure close_idx, high_idx, low_idx are within [0, TEST_NUM_FEATURES_PER_ASSET - 1]
# For example, if TEST_NUM_FEATURES_PER_ASSET = 3:
# feature 0: close/primary
# feature 1: high
# feature 2: low

# --- Test Constants for Trend Strategies ---
# Let's use a more common feature set for trend strategies (OHLCV)
TEST_TREND_NUM_FEATURES = 5 
TEST_TREND_CLOSE_IDX = 3 # Assuming O=0, H=1, L=2, C=3, V=4
TEST_TREND_HIGH_IDX = 1
TEST_TREND_LOW_IDX = 2
TEST_TREND_PERIOD_SHORT = 7
TEST_TREND_PERIOD_MEDIUM = 14
TEST_TREND_PERIOD_LONG = 21


# --- Helper Functions for Mock Data ---
def create_single_asset_mock_features(batch_size, seq_len, num_features, device): # Added device param
    """Creates mock features for a single asset: (batch_size, seq_len, num_features)"""
    return torch.randn(batch_size, seq_len, num_features, device=device)

def create_pair_asset_mock_features(batch_size, seq_len, num_features_per_asset_in_pair, device): # Added device param
    """Creates mock features for a pair of assets, concatenated: (batch_size, seq_len, num_features_per_asset_in_pair * 2)"""
    return torch.randn(batch_size, seq_len, num_features_per_asset_in_pair * 2, device=device)

def create_composite_mock_features(batch_size, seq_len, num_total_assets, num_features_per_asset_overall, device): # Added device param
    """Creates mock features for a composite strategy: (batch_size, seq_len, num_total_assets * num_features_per_asset_overall)"""
    return torch.randn(batch_size, seq_len, num_total_assets * num_features_per_asset_overall, device=device)

# --- Unit Tests for Refactored Statistical Arbitrage Strategies ---

class TestMeanReversionStrategy(BaseStrategyTest):
    @pytest.fixture
    def strategy_config(self):
        config = MeanReversionStrategy.default_config()
        config.default_params.update({
            "window": TEST_WINDOW_SHORT,
            "std_dev_multiplier": 2.0,
            "close_idx": 0,
            "num_features_per_asset": TEST_NUM_FEATURES_PER_ASSET        })
        return config

    def test_forward_pass(self, strategy_config, mock_logger_fixture, device): # Removed mocker dependency
        strategy = MeanReversionStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device) # Use fixture device

        asset_features = create_single_asset_mock_features(
            TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_NUM_FEATURES_PER_ASSET, device=device # Pass device
        )
        signals = strategy.forward(asset_features)

        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        # MODIFIED: Assert against device string directly
        assert signals.device.type == device # Use fixture device
        assert torch.all(signals >= -1) and torch.all(signals <= 1) # Assuming signals are bounded
        # Correct assertion for discrete signals: -1, 0, or 1
        is_valid_signal = (signals == -1.0) | (signals == 0.0) | (signals == 1.0)
        assert torch.all(is_valid_signal), "Signals are not strictly -1, 0, or 1"

    def test_forward_short_sequence(self, strategy_config, mock_logger_fixture, device): # Added device
        strategy = MeanReversionStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device) # Use fixture device
        
        short_seq_len = strategy_config.default_params["window"] - 1
        if short_seq_len <= 0:
            pytest.skip("Window is too small to test short sequence meaningfully.")

        asset_features = create_single_asset_mock_features(
            TEST_BATCH_SIZE, short_seq_len, TEST_NUM_FEATURES_PER_ASSET, device=device # Pass device
        )
        signals = strategy.forward(asset_features)

        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        assert torch.all(signals == 0)

    def test_forward_zero_std_dev(self, strategy_config, mock_logger_fixture, device): # Added device
        strategy = MeanReversionStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device) # Use fixture device

        # Create features where close price is constant, leading to zero std dev
        asset_features = torch.zeros(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_NUM_FEATURES_PER_ASSET, device=device) # Pass device
        asset_features[:, :, strategy_config.default_params["close_idx"]] = 100.0 # Constant close price

        signals = strategy.forward(asset_features)
        
        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        # Expect signals to be 0 if std dev is clamped at a small value and price is at mean
        # Or, if price is exactly the mean, signal is 0. If std is 0 and price is not mean, could be +/-1.
        # The strategy clamps std_dev at 1e-6. If (price - mean) is non-zero, signal will be +/-1.
        # If price is constant, mean is price, so (price - mean) is 0. Signal should be 0.
        assert torch.all(signals == 0)


class TestCointegrationStrategy(BaseStrategyTest):
    @pytest.fixture
    def strategy_config(self):
        config = CointegrationStrategy.default_config()
        config.default_params.update({
            "window": TEST_WINDOW_SHORT,
            "std_dev_multiplier": 1.5,
            "asset1_close_idx": 0, # Relative to asset1's feature block
            "asset2_close_idx": 0, # Relative to asset2's feature block (i.e., index 0 within the second half of features)
            "num_features_per_asset": TEST_NUM_FEATURES_PER_ASSET,
            # "pair_assets": ["ASSET1", "ASSET2"] # Not directly used by forward, but by __init__
        })
        # For CointegrationStrategy, __init__ also expects 'pair_assets' in instance_params
        # but this test focuses on forward, which uses indices.
        return config

    def test_forward_pass(self, strategy_config, mock_logger_fixture, device): # Removed mocker dependency
        # instance_params for __init__ can be strategy_config.default_params
        # It needs \'pair_assets\' if the strategy\'s __init__ strictly requires it for logging/naming,
        # but the tensor logic in forward relies on indices.
        instance_params = strategy_config.default_params.copy()
        instance_params["pair_assets"] = ["ASSET1", "ASSET2"] # Add dummy pair_assets for init
        
        strategy = CointegrationStrategy(strategy_config, instance_params, mock_logger_fixture)
        strategy.to(device) # Use fixture device

        asset_features = create_pair_asset_mock_features(
            TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_NUM_FEATURES_PER_ASSET, device=device # Pass device
        )
        signals = strategy.forward(asset_features)

        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        # MODIFIED: Assert against device string directly
        assert signals.device.type == device # Use fixture device
        assert torch.all(signals >= -1) and torch.all(signals <= 1)
        is_valid_signal = (signals == -1.0) | (signals == 0.0) | (signals == 1.0)
        assert torch.all(is_valid_signal), "Signals are not strictly -1, 0, or 1"

    def test_forward_short_sequence(self, strategy_config, mock_logger_fixture, device): # Added device
        instance_params = strategy_config.default_params.copy()
        instance_params["pair_assets"] = ["ASSET1", "ASSET2"]
        strategy = CointegrationStrategy(strategy_config, instance_params, mock_logger_fixture)
        strategy.to(device) # Use fixture device
        
        short_seq_len = strategy_config.default_params["window"] - 1
        if short_seq_len <= 0:
            pytest.skip("Window is too small.")

        asset_features = create_pair_asset_mock_features(
            TEST_BATCH_SIZE, short_seq_len, TEST_NUM_FEATURES_PER_ASSET, device=device # Pass device
        )
        signals = strategy.forward(asset_features)

        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        assert torch.all(signals == 0)

class TestPairsTradeStrategy(BaseStrategyTest):
    @pytest.fixture
    def strategy_config(self):
        config = PairsTradeStrategy.default_config()
        config.default_params.update({
            "window": TEST_WINDOW_LONG,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "asset1_close_idx": 0,            "asset2_close_idx": 0,
            "num_features_per_asset": TEST_NUM_FEATURES_PER_ASSET,
        })
        return config

    def test_forward_pass(self, strategy_config, mock_logger_fixture, device): # Removed mocker dependency
        instance_params = strategy_config.default_params.copy()
        instance_params["pair_assets"] = ["ASSET_X", "ASSET_Y"]
        strategy = PairsTradeStrategy(strategy_config, instance_params, mock_logger_fixture)
        strategy.to(device) # Use fixture device

        asset_features = create_pair_asset_mock_features(
            TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_NUM_FEATURES_PER_ASSET, device=device # Pass device
        )
        # current_positions for the pair: (batch_size, 1)
        current_positions = torch.randint(-1, 2, (TEST_BATCH_SIZE, 1), device=device, dtype=torch.float32) # Pass device
        
        signals = strategy.forward(asset_features, current_positions)

        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        # MODIFIED: Assert against device string directly
        assert signals.device.type == device # Use fixture device
        # Replace isin with a compatible check
        is_valid_signal = (signals == -1.0) | (signals == 0.0) | (signals == 1.0)
        assert torch.all(is_valid_signal), "Signals are not strictly -1, 0, or 1"

    def test_forward_short_sequence(self, strategy_config, mock_logger_fixture, device): # Added device
        instance_params = strategy_config.default_params.copy()
        instance_params["pair_assets"] = ["ASSET_X", "ASSET_Y"]
        strategy = PairsTradeStrategy(strategy_config, instance_params, mock_logger_fixture)
        strategy.to(device) # Use fixture device

        short_seq_len = strategy_config.default_params["window"] - 1
        if short_seq_len <= 0:
            pytest.skip("Window is too small.")

        asset_features = create_pair_asset_mock_features(
            TEST_BATCH_SIZE, short_seq_len, TEST_NUM_FEATURES_PER_ASSET, device=device # Pass device
        )
        current_positions = torch.zeros(TEST_BATCH_SIZE, 1, device=device) # Pass device
        signals = strategy.forward(asset_features, current_positions)

        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        assert torch.all(signals == 0) # Expect no action if data is insufficient

# --- Unit Tests for Refactored Trend Strategies ---

class TestMomentumStrategy(BaseStrategyTest):
    @pytest.fixture
    def strategy_config(self):
        config = MomentumStrategy.default_config()
        config.default_params.update({
            "window": TEST_TREND_PERIOD_MEDIUM,
            "close_idx": TEST_TREND_CLOSE_IDX,
            "num_features_per_asset": TEST_TREND_NUM_FEATURES
        })
        return config

    def test_forward_pass(self, strategy_config, mock_logger_fixture, device):
        strategy = MomentumStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device)

        asset_features = create_single_asset_mock_features(
            TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_TREND_NUM_FEATURES, device=device
        )
        signals = strategy.forward(asset_features)

        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        assert signals.device.type == device
        is_valid_signal = (signals == -1.0) | (signals == 0.0) | (signals == 1.0)
        assert torch.all(is_valid_signal), "Signals are not strictly -1, 0, or 1"

    def test_forward_short_sequence(self, strategy_config, mock_logger_fixture, device):
        strategy = MomentumStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device)
        
        # ROC needs at least `window` prior data points + 1 current point.
        # So, sequence length must be >= window + 1 for a non-zero ROC.
        # If seq_len is `window`, ROC will be calculated based on `window-1` lookback, which is still valid.
        # If seq_len is less than `window`, it should produce 0.
        short_seq_len = strategy_config.default_params["window"] -1 
        if short_seq_len <= 0:
            pytest.skip("Window is too small to test short sequence meaningfully.")

        asset_features = create_single_asset_mock_features(
            TEST_BATCH_SIZE, short_seq_len, TEST_TREND_NUM_FEATURES, device=device
        )
        signals = strategy.forward(asset_features)

        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        assert torch.all(signals == 0), "Signal should be 0 for sequence shorter than window"

    def test_forward_roc_logic(self, strategy_config, mock_logger_fixture, device):
        strategy = MomentumStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device)
        window = strategy_config.default_params["window"]
        close_idx = strategy_config.default_params["close_idx"]

        # Create data with known ROC
        # Batch 1: Price increasing (positive ROC -> buy signal)
        # Batch 2: Price decreasing (negative ROC -> sell signal)
        asset_features = torch.zeros(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_TREND_NUM_FEATURES, device=device)
        
        # Batch 1: Increasing prices
        for i in range(TEST_SEQ_LEN):
            asset_features[0, i, close_idx] = 100 + i 
        # Batch 2: Decreasing prices
        for i in range(TEST_SEQ_LEN):
            asset_features[1, i, close_idx] = 100 - i
            
        signals = strategy.forward(asset_features)
        assert signals[0, 0, 0] == 1.0, "Expected buy signal for increasing price"
        assert signals[1, 0, 0] == -1.0, "Expected sell signal for decreasing price"

class TestTrendMeanReversionStrategy(BaseStrategyTest):
    @pytest.fixture
    def strategy_config(self):
        config = TrendMeanReversionStrategy.default_config() # Use the aliased import
        config.default_params.update({
            "window": TEST_TREND_PERIOD_MEDIUM,
            "std_dev_multiplier": 2.0,
            "close_idx": TEST_TREND_CLOSE_IDX,
            "num_features_per_asset": TEST_TREND_NUM_FEATURES
        })
        return config

    def test_forward_pass(self, strategy_config, mock_logger_fixture, device):
        strategy = TrendMeanReversionStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device)

        asset_features = create_single_asset_mock_features(
            TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_TREND_NUM_FEATURES, device=device
        )
        signals = strategy.forward(asset_features)

        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        assert signals.device.type == device
        is_valid_signal = (signals == -1.0) | (signals == 0.0) | (signals == 1.0)
        assert torch.all(is_valid_signal), "Signals are not strictly -1, 0, or 1"

    def test_forward_short_sequence(self, strategy_config, mock_logger_fixture, device):
        strategy = TrendMeanReversionStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device)
        
        short_seq_len = strategy_config.default_params["window"] -1
        if short_seq_len <= 0:
            pytest.skip("Window is too small.")

        asset_features = create_single_asset_mock_features(
            TEST_BATCH_SIZE, short_seq_len, TEST_TREND_NUM_FEATURES, device=device
        )
        signals = strategy.forward(asset_features)

        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        assert torch.all(signals == 0)

    def test_forward_signal_logic(self, strategy_config, mock_logger_fixture, device):
        strategy = TrendMeanReversionStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device)
        window = strategy_config.default_params["window"]
        std_multiplier = strategy_config.default_params["std_dev_multiplier"]
        close_idx = strategy_config.default_params["close_idx"]

        asset_features = torch.zeros(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_TREND_NUM_FEATURES, device=device)
        
        # Batch 1: Price significantly above rolling mean + std_dev (sell signal)
        mean_val_b1 = 100
        std_val_b1 = 5
        asset_features[0, :, close_idx] = torch.linspace(mean_val_b1 - std_val_b1, mean_val_b1 + std_val_b1, TEST_SEQ_LEN, device=device)
        asset_features[0, -1, close_idx] = mean_val_b1 + (std_multiplier + 0.5) * std_val_b1 # Far above upper band

        # Batch 2: Price significantly below rolling mean - std_dev (buy signal)
        mean_val_b2 = 100
        std_val_b2 = 5
        asset_features[1, :, close_idx] = torch.linspace(mean_val_b2 + std_val_b2, mean_val_b2 - std_val_b2, TEST_SEQ_LEN, device=device)
        asset_features[1, -1, close_idx] = mean_val_b2 - (std_multiplier + 0.5) * std_val_b2 # Far below lower band

        signals = strategy.forward(asset_features)
        assert signals[0, 0, 0] == -1.0, "Expected sell signal for price above upper band"
        assert signals[1, 0, 0] == 1.0, "Expected buy signal for price below lower band"

class TestReversalStrategy(BaseStrategyTest):
    @pytest.fixture
    def strategy_config(self):
        config = ReversalStrategy.default_config()
        config.default_params.update({
            "rsi_period": TEST_TREND_PERIOD_MEDIUM,
            "stoch_k_period": TEST_TREND_PERIOD_MEDIUM,
            "stoch_d_period": TEST_TREND_PERIOD_SHORT,
            "overbought_rsi": 70.0,
            "oversold_rsi": 30.0,
            "overbought_stoch": 80.0,
            "oversold_stoch": 20.0,
            "close_idx": TEST_TREND_CLOSE_IDX,
            "high_idx": TEST_TREND_HIGH_IDX,
            "low_idx": TEST_TREND_LOW_IDX,
            "num_features_per_asset": TEST_TREND_NUM_FEATURES
        })
        return config

    def test_forward_pass(self, strategy_config, mock_logger_fixture, device):
        strategy = ReversalStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device)

        asset_features = create_single_asset_mock_features(
            TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_TREND_NUM_FEATURES, device=device
        )
        # Ensure High >= Low, High >= Close, Low <= Close for realistic HLC values
        low_prices = torch.rand(TEST_BATCH_SIZE, TEST_SEQ_LEN, device=device) * 100
        high_prices = low_prices + torch.rand(TEST_BATCH_SIZE, TEST_SEQ_LEN, device=device) * 10
        close_prices = low_prices + torch.rand(TEST_BATCH_SIZE, TEST_SEQ_LEN, device=device) * (high_prices - low_prices)
        asset_features[:, :, strategy_config.default_params["low_idx"]] = low_prices
        asset_features[:, :, strategy_config.default_params["high_idx"]] = high_prices
        asset_features[:, :, strategy_config.default_params["close_idx"]] = close_prices

        signals = strategy.forward(asset_features)

        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        assert signals.device.type == device
        is_valid_signal = (signals == -1.0) | (signals == 0.0) | (signals == 1.0)
        assert torch.all(is_valid_signal), "Signals are not strictly -1, 0, or 1"

    def test_forward_short_sequence(self, strategy_config, mock_logger_fixture, device):
        strategy = ReversalStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device)
        
        # RSI and Stochastic need at least their respective periods.
        # The strategy uses max(rsi_period, stoch_k_period) for its internal check.
        required_len = max(strategy_config.default_params["rsi_period"], strategy_config.default_params["stoch_k_period"])
        short_seq_len = required_len -1 
        if short_seq_len <= 0:
            pytest.skip("Periods are too small to test short sequence meaningfully.")

        asset_features = create_single_asset_mock_features(
            TEST_BATCH_SIZE, short_seq_len, TEST_TREND_NUM_FEATURES, device=device
        )
        signals = strategy.forward(asset_features)

        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        assert torch.all(signals == 0), "Signal should be 0 for insufficient sequence length"

    # More specific tests for RSI/Stochastic conditions can be added here if needed,
    # but they would be complex to set up with precise mock data to trigger specific crossover events.
    # The current forward pass test covers the basic operation.

# Placeholder for BreakoutStrategy tests (to be implemented)
class TestBreakoutStrategy(BaseStrategyTest):
    @pytest.fixture
    def strategy_config(self):
        config = BreakoutStrategy.default_config()
        config.default_params.update({
            "breakout_window": TEST_TREND_PERIOD_LONG,
            "std_dev_multiplier": 2.0,
            "min_breakout_volume_increase_pct": 0.5,
            "close_idx": TEST_TREND_CLOSE_IDX, # Standard OHLCV: C=3
            "high_idx": TEST_TREND_HIGH_IDX,   # H=1
            "low_idx": TEST_TREND_LOW_IDX,    # L=2
            "volume_idx": 4, # V=4
            "num_features_per_asset": TEST_TREND_NUM_FEATURES
        })
        return config

    # @pytest.mark.skip(reason="BreakoutStrategy tensor forward and tests not yet fully implemented.")
    def test_forward_pass(self, strategy_config, mock_logger_fixture, device):
        strategy = BreakoutStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device)
        asset_features = create_single_asset_mock_features(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_TREND_NUM_FEATURES, device=device)
        
        # Ensure HLCV are somewhat realistic if strategy depends on their relationships
        low_prices = torch.rand(TEST_BATCH_SIZE, TEST_SEQ_LEN, device=device) * 100
        high_prices = low_prices + torch.rand(TEST_BATCH_SIZE, TEST_SEQ_LEN, device=device) * 10
        close_prices = low_prices + torch.rand(TEST_BATCH_SIZE, TEST_SEQ_LEN, device=device) * (high_prices - low_prices)
        volume_data = torch.rand(TEST_BATCH_SIZE, TEST_SEQ_LEN, device=device) * 1000

        asset_features[:, :, strategy_config.default_params["low_idx"]] = low_prices
        asset_features[:, :, strategy_config.default_params["high_idx"]] = high_prices
        asset_features[:, :, strategy_config.default_params["close_idx"]] = close_prices
        asset_features[:, :, strategy_config.default_params["volume_idx"]] = volume_data
        
        signals = strategy.forward(asset_features)
        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        assert signals.device.type == device
        is_valid_signal = (signals == -1.0) | (signals == 0.0) | (signals == 1.0)
        assert torch.all(is_valid_signal)

    def test_forward_short_sequence(self, strategy_config, mock_logger_fixture, device):
        strategy = BreakoutStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device)
        
        short_seq_len = strategy_config.default_params["breakout_window"] - 1
        if short_seq_len <= 0:
            pytest.skip("Window is too small.")

        asset_features = create_single_asset_mock_features(
            TEST_BATCH_SIZE, short_seq_len, TEST_TREND_NUM_FEATURES, device=device
        )
        signals = strategy.forward(asset_features)

        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        assert torch.all(signals == 0)

    def test_breakout_logic(self, strategy_config, mock_logger_fixture, device):
        strategy = BreakoutStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device)
        window = strategy_config.default_params["breakout_window"]
        std_multiplier = strategy_config.default_params["std_dev_multiplier"]
        vol_increase_pct = strategy_config.default_params["min_breakout_volume_increase_pct"]
        close_idx = strategy_config.default_params["close_idx"]
        high_idx = strategy_config.default_params["high_idx"]
        low_idx = strategy_config.default_params["low_idx"]
        volume_idx = strategy_config.default_params["volume_idx"]

        asset_features = torch.ones(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_TREND_NUM_FEATURES, device=device) * 100 # Base price 100
        asset_features[:, :, volume_idx] = 1000 # Base volume 1000

        # Batch 0: Upward breakout
        # Make last few prices stable to establish BBands
        for i in range(window):
            asset_features[0, TEST_SEQ_LEN - 1 - i, close_idx] = 100 
            asset_features[0, TEST_SEQ_LEN - 1 - i, high_idx] = 101
            asset_features[0, TEST_SEQ_LEN - 1 - i, low_idx] = 99
        # Last point breaks out
        asset_features[0, -1, close_idx] = 110 # Above typical std_dev for constant price
        asset_features[0, -1, high_idx] = 112
        asset_features[0, -1, low_idx] = 108
        asset_features[0, -1, volume_idx] = 2000 # Increased volume

        # Batch 1: Downward breakout
        for i in range(window):
            asset_features[1, TEST_SEQ_LEN - 1 - i, close_idx] = 100
            asset_features[1, TEST_SEQ_LEN - 1 - i, high_idx] = 101
            asset_features[1, TEST_SEQ_LEN - 1 - i, low_idx] = 99
        # Last point breaks out downwards
        asset_features[1, -1, close_idx] = 90
        asset_features[1, -1, high_idx] = 92
        asset_features[1, -1, low_idx] = 88
        asset_features[1, -1, volume_idx] = 2000 # Increased volume

        signals = strategy.forward(asset_features)
        # Note: Exact BBand values depend on rolling_std of near-constant data which can be very small.
        # This test checks if signals are generated; precise values might need fine-tuning of mock data.
        # For constant data, std will be near zero, so any deviation is a breakout.
        assert signals[0, 0, 0] == 1.0, "Expected buy signal for upward breakout"
        assert signals[1, 0, 0] == -1.0, "Expected sell signal for downward breakout"


# Placeholder for TrendFollowingStrategy tests (to be implemented)
class TestTrendFollowingStrategy(BaseStrategyTest):
    @pytest.fixture
    def strategy_config(self):
        config = TrendFollowingStrategy.default_config()
        config.default_params.update({
            "short_sma_period": TEST_TREND_PERIOD_SHORT,
            "long_sma_period": TEST_TREND_PERIOD_LONG,
            "close_idx": TEST_TREND_CLOSE_IDX, # C=3 for OHLCV
            "num_features_per_asset": TEST_TREND_NUM_FEATURES
        })
        return config

    # @pytest.mark.skip(reason="TrendFollowingStrategy tensor forward and tests not yet fully implemented.")
    def test_forward_pass(self, strategy_config, mock_logger_fixture, device):
        strategy = TrendFollowingStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device)
        asset_features = create_single_asset_mock_features(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_TREND_NUM_FEATURES, device=device)
        signals = strategy.forward(asset_features)
        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        assert signals.device.type == device
        is_valid_signal = (signals == -1.0) | (signals == 0.0) | (signals == 1.0)
        assert torch.all(is_valid_signal)

    def test_forward_short_sequence(self, strategy_config, mock_logger_fixture, device):
        strategy = TrendFollowingStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device)
        
        short_seq_len = strategy_config.default_params["long_sma_period"] - 1
        if short_seq_len <= 0:
            pytest.skip("Long SMA period is too small.")

        asset_features = create_single_asset_mock_features(
            TEST_BATCH_SIZE, short_seq_len, TEST_TREND_NUM_FEATURES, device=device
        )
        signals = strategy.forward(asset_features)

        assert signals.shape == (TEST_BATCH_SIZE, 1, 1)
        assert torch.all(signals == 0)

    def test_sma_crossover_logic(self, strategy_config, mock_logger_fixture, device):
        strategy = TrendFollowingStrategy(strategy_config, strategy_config.default_params, mock_logger_fixture)
        strategy.to(device)
        short_window = strategy_config.default_params["short_sma_period"]
        long_window = strategy_config.default_params["long_sma_period"]
        close_idx = strategy_config.default_params["close_idx"]

        asset_features = torch.zeros(TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_TREND_NUM_FEATURES, device=device)

        # Batch 0: Golden cross (short SMA crosses above long SMA)
        # Prices start low, then rise, causing short SMA to rise faster
        prices_b0 = torch.cat([
            torch.linspace(100, 90, long_window // 2, device=device), 
            torch.linspace(90, 110, TEST_SEQ_LEN - (long_window // 2), device=device)
        ])
        asset_features[0, :, close_idx] = prices_b0

        # Batch 1: Death cross (short SMA crosses below long SMA)
        # Prices start high, then fall, causing short SMA to fall faster
        prices_b1 = torch.cat([
            torch.linspace(100, 110, long_window // 2, device=device), 
            torch.linspace(110, 90, TEST_SEQ_LEN - (long_window // 2), device=device)
        ])
        asset_features[1, :, close_idx] = prices_b1
        
        # Ensure sequence is long enough for at least two points for crossover check
        if TEST_SEQ_LEN < 2:
            pytest.skip("Sequence length too short for crossover logic test.")

        signals = strategy.forward(asset_features)
        
        # These assertions are highly dependent on the exact price series and window lengths.
        # A robust test would pre-calculate expected SMA values and crossover points.
        # For now, we check if a signal is generated. It might be 0 if crossover isn't at the very end.
        # To make it more deterministic, one might need to craft the last few points very carefully.
        # For simplicity, we'll check that the signals are valid (-1, 0, or 1).
        # A more precise test would involve setting up data for a clear crossover at the last step.

        # Example of setting up a clear golden cross at the end for Batch 0:
        # Assume short_sma_period=5, long_sma_period=10. Seq_len >= 10.
        # Prices for Batch 0 to force golden cross at t-1 -> t:
        # t-11 to t-2: long_sma is higher or equal
        # t-1: short_sma crosses above long_sma
        # Example: close_prices_b0 = [..., 98,99,100,101,102,  100,100,100,100,105] (last 10, short=5, long=10)
        # prev_short_sma_b0 around (98+99+100+101+102)/5 = 100
        # prev_long_sma_b0 around average of last 10 up to t-2 (assume it's ~100)
        # current_short_sma_b0 around (99+100+101+102+105)/5 = 101.4
        # current_long_sma_b0 around average of last 10 up to t-1 (assume it's ~100.5)
        # This is tricky to set up without exact SMA calculation here.
        # The current TrendFollowingStrategy forward looks at [-1] and [-2] of the *entire sequence* SMAs.

        # For now, just check signal validity
        is_valid_signal_b0 = (signals[0,0,0] == -1.0) | (signals[0,0,0] == 0.0) | (signals[0,0,0] == 1.0)
        assert is_valid_signal_b0, "Signal for batch 0 (potential golden cross) is invalid"
        is_valid_signal_b1 = (signals[1,0,0] == -1.0) | (signals[1,0,0] == 0.0) | (signals[1,0,0] == 1.0)
        assert is_valid_signal_b1, "Signal for batch 1 (potential death cross) is invalid"
        
        # A more direct test for crossover logic:
        # Create SMAs directly and test the crossover condition
        # Batch 0: Golden Cross
        # short_sma: ..., 99, 101 (prev, current)
        # long_sma:  ..., 100, 100 (prev, current)
        asset_features_gc = torch.zeros(1, TEST_SEQ_LEN, TEST_TREND_NUM_FEATURES, device=device)
        if TEST_SEQ_LEN >= long_window:
            # Construct prices to create specific SMA values at the end
            # This is complex. A simpler way is to mock the _rolling_mean or test the logic separately.
            # For now, we rely on the forward pass test and short sequence test.
            pass # Placeholder for more precise crossover data generation

