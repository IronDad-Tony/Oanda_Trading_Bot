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
    batch_size = asset_features_batch.shape[0] # ADDED: Define batch_size
    assert output_actions.shape == (batch_size, expected_num_assets, 1)


# ADDED TEST
@pytest.mark.parametrize("use_gumbel_softmax_param", [True, False]) # Test both Gumbel and standard Softmax for internal weights
def test_strategy_combination_weights_and_execution_order(
    base_config: StrategyConfig, 
    mock_logger, 
    device,
    use_gumbel_softmax_param: bool
):
    """
    Tests the strategy combination mechanism in EnhancedStrategySuperposition,
    focusing on weight distribution (both internal and external) and execution order.
    """
    layer_attention_input_dim = base_config.default_params.get("market_feature_dim", 32)
    # CORRECTED: Use a consistent feature dimension for the test
    strategy_feature_input_dim = 5 # Match the dimension used in mock_input_features_for_real_strategies fixture if applicable, or define clearly
    base_config.default_params["feature_dim"] = strategy_feature_input_dim # Ensure base_config reflects this for consistency if used by other parts

    batch_size = 2
    num_test_assets = 2
    seq_len = 10

    # Create mock strategies
    mock_strategy_configs = []
    mock_strategy_classes = []
    mock_strategy_instances = []

    for i in range(3): # Create 3 mock strategies
        strat_id_numeric = i + 1
        # Define a unique mock strategy class for each instance to allow specific mocking
        class MockStrat(BaseStrategy):
            # Need to capture i for default_config name and forward method
            # This is tricky with class definitions in a loop. A closure or functools.partial might be needed
            # For simplicity, we'll make the name and output dependent on an instance variable set in __init__
            
            def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
                super().__init__(config, params, logger)
                # Extract numeric id from name, assuming name is like "MockStrat1", "MockStrat2"
                self.numeric_id = int(self.config.name.replace("MockStrat", ""))
                self.forward_call_args = []

            @staticmethod
            def default_config(): 
                # This static method cannot use `i` directly from the loop.
                # The name will be set when creating the config object for the layer.
                # The default_params can set the input_dim.
                return StrategyConfig(
                    name="PLACEHOLDER_NAME", # This will be overridden
                    default_params={'input_dim': strategy_feature_input_dim} # Ensure this matches asset_features_batch
                )

            def forward(self, asset_features, current_positions=None, timestamp=None):
                self.forward_call_args.append({
                    'asset_features_shape': asset_features.shape,
                    'current_positions_shape': current_positions.shape if current_positions is not None else None,
                    'timestamp': timestamp
                })
                # Return a unique, predictable output for this strategy
                # Output shape: (batch_size, 1, 1)
                return torch.full((asset_features.shape[0], 1, 1), float(self.numeric_id), device=asset_features.device)

        # Update the class name to be unique for registration if STRATEGY_REGISTRY is used by ESS init
        MockStrat.__name__ = f"MockStrat{strat_id_numeric}"
        
        # Create the config with the correct name that the layer will see
        config = StrategyConfig(
            name=f"MockStrat{strat_id_numeric}", 
            default_params={'input_dim': strategy_feature_input_dim},
            description=f"Mock strategy {strat_id_numeric}"
        )
        # Ensure input_dim is correctly set in the config object passed to the layer
        config.input_dim = strategy_feature_input_dim 
        mock_strategy_configs.append(config)
        mock_strategy_classes.append(MockStrat)

    # Initialize EnhancedStrategySuperposition
    # Patch STRATEGY_REGISTRY temporarily for this test to include our mocks by name
    # This ensures that _initialize_strategies can find them if it looks up by config name.
    temp_strategy_registry = STRATEGY_REGISTRY.copy()
    for idx, strat_class in enumerate(mock_strategy_classes):
        temp_strategy_registry[mock_strategy_configs[idx].name] = strat_class

    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger), \
         patch('src.agent.enhanced_quantum_strategy_layer.STRATEGY_REGISTRY', temp_strategy_registry):        
        layer = EnhancedStrategySuperposition(
            input_dim=layer_attention_input_dim,
            num_strategies=len(mock_strategy_classes),
            strategy_configs=mock_strategy_configs, 
            explicit_strategies=None, # Rely on strategy_configs and patched STRATEGY_REGISTRY
            strategy_input_dim=strategy_feature_input_dim,
            use_gumbel_softmax=use_gumbel_softmax_param 
        )
    layer.to(device)
    layer.eval() # Set layer to evaluation mode

    assert layer.num_actual_strategies == len(mock_strategy_classes), "Incorrect number of strategies loaded"
    
    mock_strategy_instances = list(layer.strategies) # Get the instantiated strategies
    assert len(mock_strategy_instances) == len(mock_strategy_classes), "Not all mock strategies were found in the layer"

    # Prepare inputs
    asset_features_batch = torch.randn(batch_size, num_test_assets, seq_len, strategy_feature_input_dim, device=device)
    market_state_features = torch.randn(batch_size, layer_attention_input_dim, device=device) 
    current_positions_batch = torch.randn(batch_size, num_test_assets, 1, device=device)
    timestamps = [pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=k) for k in range(batch_size)]

    # --- Test Case 1: External Weights ---
    mock_logger.info("Testing with External Weights")
    external_weights_input = torch.tensor([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]], device=device, dtype=torch.float) # Renamed to avoid confusion
    assert external_weights_input.shape == (batch_size, layer.num_actual_strategies)

    for mock_strat_instance in mock_strategy_instances:
        mock_strat_instance.forward_call_args = []

    output_actions_ext = layer.forward(
        asset_features_batch,
        market_state_features=market_state_features, 
        current_positions_batch=current_positions_batch,
        timestamps=timestamps,
        external_weights=external_weights_input # Pass the raw input weights
    )

    assert output_actions_ext.shape == (batch_size, num_test_assets, 1)

    for mock_strat_instance in mock_strategy_instances:
        assert len(mock_strat_instance.forward_call_args) == num_test_assets
        for k in range(num_test_assets):
            call_arg = mock_strat_instance.forward_call_args[k]
            assert call_arg['asset_features_shape'] == (batch_size, seq_len, strategy_feature_input_dim)
            assert call_arg['current_positions_shape'] == (batch_size, 1)
            # Timestamps are passed per batch item, but strategies currently take the first one.
            # The layer passes timestamps[0] to each strategy if timestamps is not None.
            # If timestamps has batch_size elements, this check is fine.
            assert call_arg['timestamp'] == timestamps[0] 

    # Calculate expected output mirroring the layer's logic for external weights
    processed_external_weights = F.softmax(external_weights_input / layer.temperature.item(), dim=1)
    # If dropout is applied in the layer even in eval mode (which it shouldn't for standard nn.Dropout),
    # this expectation would be harder to match without mocking dropout or knowing its exact behavior.
    # Assuming nn.Dropout in eval mode is an identity operation.

    expected_output_ext = torch.zeros(batch_size, num_test_assets, 1, device=device)
    for b in range(batch_size):
        for a in range(num_test_assets):
            weighted_sum = 0.0
            for s_idx, mock_strat_instance in enumerate(mock_strategy_instances):
                # Each mock strategy returns its numeric_id as a float
                # The forward method of MockStrat returns torch.full((asset_features.shape[0], 1, 1), float(self.numeric_id), ...)
                # So, for a given batch item, the raw output is float(mock_strat_instance.numeric_id)
                strategy_raw_output = float(mock_strat_instance.numeric_id) 
                weighted_sum += processed_external_weights[b, s_idx] * strategy_raw_output
            expected_output_ext[b, a, 0] = weighted_sum
    
    assert torch.allclose(output_actions_ext, expected_output_ext, atol=1e-5), (
        f"External weights combination failed. Expected:\\n{expected_output_ext}\\n"
        f"Got:\\n{output_actions_ext}\\n"
        f"Processed weights used for expectation:\\n{processed_external_weights}"
    )

    # --- Test Case 2: Internal Weights (Attention-based) ---
    mock_logger.info(f"Testing with Internal Weights (Gumbel Softmax: {use_gumbel_softmax_param})")
    for mock_strat_instance in mock_strategy_instances:
        mock_strat_instance.forward_call_args = []
    
    internal_mock_logits = torch.tensor([[1.0, 5.0, 2.0], [5.0, 1.0, 2.0]], device=device, dtype=torch.float)
    # CORRECTED: Patch the 'forward' method of the attention_network instance
    with patch.object(layer.attention_network, 'forward', MagicMock(return_value=internal_mock_logits)) as mock_attention_forward:
        output_actions_internal = layer.forward(
            asset_features_batch,
            market_state_features=market_state_features, 
            current_positions_batch=current_positions_batch,
            timestamps=timestamps,
            external_weights=None 
        )
        mock_attention_forward.assert_called_once_with(market_state_features)

    assert output_actions_internal.shape == (batch_size, num_test_assets, 1)

    for mock_strat_instance in mock_strategy_instances:
        assert len(mock_strat_instance.forward_call_args) == num_test_assets 

    current_temp = layer.temperature.item()
    expected_internal_weights = F.softmax(internal_mock_logits / current_temp, dim=1)
    layer.eval() 

    expected_output_int = torch.zeros(batch_size, num_test_assets, 1, device=device)
    for b in range(batch_size):
        for a in range(num_test_assets):
            weighted_sum = 0.0
            for s_idx, mock_strat_instance in enumerate(mock_strategy_instances):
                strategy_raw_output = float(mock_strat_instance.numeric_id) # Use numeric_id
                weighted_sum += expected_internal_weights[b, s_idx] * strategy_raw_output
            expected_output_int[b, a, 0] = weighted_sum
            
    assert torch.allclose(output_actions_internal, expected_output_int, atol=1e-4), (
        f"Internal weights combination failed (Gumbel: {use_gumbel_softmax_param}). Expected:\\n{expected_output_int}\\n"
        f"Got:\\n{output_actions_internal}\\n"
        f"Processed weights used for expectation:\\n{expected_internal_weights}"
    )

    mock_logger.info(f"Test passed with Gumbel Softmax: {use_gumbel_softmax_param}")

def test_dynamic_strategy_generator_generates_strategy(mock_logger): # Removed mocker dependency
    """Test that DynamicStrategyGenerator can generate a new strategy instance."""

