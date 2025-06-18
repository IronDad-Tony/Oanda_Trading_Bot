
import torch
import pandas as pd # Ensure pandas is imported if used by strategies
import numpy as np # Ensure numpy is imported if used by strategies
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[2] # Adjusted for strategies subdirectory
sys.path.append(str(PROJECT_ROOT))

# --- Configure Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Attempt to Import Core Modules ---
try:
    from src.agent.strategies.base_strategy import BaseStrategy, StrategyConfig
    from src.agent.strategies.trend_strategies import MomentumStrategy, BreakoutStrategy, TrendFollowingStrategy, ReversalStrategy, MeanReversionStrategy
    # Import other strategy types as you create their debug scripts
    logger.info("✓ Trend strategy modules imported successfully.")
except ImportError as e:
    logger.error(f"❌ Failed to import trend strategy modules: {e}")
    sys.exit(1)

# --- Dummy Data and Config Generation ---
def get_dummy_strategy_config(strategy_name: str, params: Optional[Dict[str, Any]] = None, asset_index: int = 0) -> StrategyConfig:
    """Creates a dummy StrategyConfig."""
    base_params = {
        "strategy_id": f"debug_{strategy_name.lower()}_{asset_index}",
        "asset_name": f"EUR_USD_{asset_index}", # Example asset name
        "asset_index": asset_index, # Index for multi-asset scenarios
        "timeframe": "H1",
        "max_concurrent_trades": 1,
        "max_position_percentage": 0.1,
        "stop_loss_type": "percentage",
        "stop_loss_value": 0.02,
        "take_profit_type": "percentage",
        "take_profit_value": 0.04,
        "trade_amount_type": "fixed_units",
        "trade_amount_value": 1000,
        "trailing_stop_enabled": False,
        "trailing_stop_type": "percentage",
        "trailing_stop_value": 0.01,
        "min_trade_duration": "1H", # Example: 1 hour
        "max_trade_duration": "24H", # Example: 24 hours
        "trade_cool_down_period": "30M" # Example: 30 minutes
    }
    if params:
        base_params.update(params)
    return StrategyConfig(**base_params)

def get_dummy_market_data(seq_len: int = 100, num_features: int = 5) -> torch.Tensor:
    """Generates dummy market data tensor (batch_size=1, seq_len, num_features)."""
    # Features could be: open, high, low, close, volume
    return torch.rand(1, seq_len, num_features)

def get_dummy_portfolio_context(num_assets: int = 1) -> Dict[str, Any]:
    """Generates a dummy portfolio context."""
    return {
        "timestamp": pd.Timestamp.now(tz='UTC'),
        "current_balance": 10000.0,
        "equity": 10000.0,
        "margin_available": 5000.0,
        "open_positions": [], # List of open position objects/dicts
        "historical_trades": [], # List of past trade objects/dicts
        "market_conditions": {"volatility_index": 0.5, "trend_strength": 0.7},
        "asset_specific_data": {
            f"asset_{i}": {"current_price": 1.12345 + i*0.1, "spread": 0.0002} for i in range(num_assets)
        }
    }

# --- Strategy Test Function ---
def test_strategy(StrategyClass: type, strategy_name: str, default_params: Dict[str, Any], asset_index: int = 0):
    logger.info(f"--- Testing Strategy: {strategy_name} (Asset Index: {asset_index}) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strategy_config = get_dummy_strategy_config(strategy_name, default_params, asset_index)
    
    try:
        strategy_instance = StrategyClass(config=strategy_config, logger=logger.getChild(strategy_name))
        strategy_instance.to(device) # If strategy is nn.Module based
        logger.info(f"✓ {strategy_name} instance created successfully.")

        # Test forward pass (if applicable, primarily for nn.Module based strategies)
        if hasattr(strategy_instance, 'forward') and isinstance(strategy_instance, torch.nn.Module):
            logger.info(f"Testing forward() method for {strategy_name}...")
            # Create dummy input tensor for the strategy's forward method
            # This shape depends on what the strategy expects. 
            # For example, (batch_size, seq_len, num_features_for_strategy)
            # BaseStrategy.forward expects (asset_features: torch.Tensor, ...)
            # asset_features shape: (batch_size, sequence_length, num_features)
            dummy_input_features = get_dummy_market_data(seq_len=50, num_features=strategy_instance.config.get('input_features_dim', 5)) # Get num_features from config or default
            dummy_input_features = dummy_input_features.to(device)
            
            # If current_positions are needed by forward:
            # dummy_current_positions = torch.zeros(1, strategy_instance.config.get('num_assets_for_strategy', 1), device=device) 
            dummy_current_positions = None # Assuming not strictly needed for all strategies for a basic test
            
            signal = strategy_instance.forward(asset_features=dummy_input_features, current_positions=dummy_current_positions)
            logger.info(f"✓ {strategy_name}.forward() executed. Signal shape: {signal.shape}, Signal: {signal.cpu().detach().numpy()}")
            # Expected signal shape: (batch_size, num_actions) or (batch_size, num_assets, num_actions_per_asset)

            # Check for gradients if strategy has parameters
            if list(strategy_instance.parameters()):
                logger.info(f"{strategy_name} has trainable parameters. Checking gradient flow...")
                # Simple gradient check: sum output, backward, check grads
                # This requires the output to be a scalar or sum() to be called.
                if signal.numel() > 0:
                    try:
                        # Ensure requires_grad is true for inputs if params exist
                        # For strategies, params are internal. Input features usually don't require grad w.r.t strategy params.
                        # We are checking if strategy params themselves get grads if the signal is part of a larger computation graph.
                        # Here, we just check if params exist and can be part of a backward pass.
                        
                        # Create a dummy loss
                        dummy_loss = signal.sum() 
                        if dummy_loss.requires_grad:
                             # Zero grads first
                            for param in strategy_instance.parameters():
                                if param.grad is not None:
                                    param.grad.zero_()
                            dummy_loss.backward()
                            
                            has_grads = False
                            for name, param in strategy_instance.named_parameters():
                                if param.grad is not None and param.grad.abs().sum() > 0:
                                    logger.info(f"  ✓ Grad found for param: {name}")
                                    has_grads = True
                                elif param.grad is None:
                                    logger.info(f"  Grad is None for param: {name}")
                                else:
                                    logger.info(f"  Grad is zero for param: {name}")
                            if has_grads:
                                logger.info(f"  ✓ Gradient flow confirmed for {strategy_name}.")
                            else:
                                logger.warning(f"  No gradients computed for {strategy_name} parameters. This might be okay if it's not a learned strategy or if inputs didn't require grad.")
                        else:
                            logger.info(f"Signal for {strategy_name} does not require grad. Skipping gradient check for this signal.")

                    except RuntimeError as e:
                        logger.error(f"  ❌ Error during gradient check for {strategy_name}: {e}")
                else:
                    logger.info(f"Signal for {strategy_name} is empty. Skipping gradient check.")
            else:
                logger.info(f"{strategy_name} has no trainable parameters.")

        # Test generate_signals method (if applicable, for more traditional strategies)
        if hasattr(strategy_instance, 'generate_signals'):
            logger.info(f"Testing generate_signals() method for {strategy_name}...")
            # Create dummy processed_data_dict (DataFrame based)
            # This is more for non-NN strategies or those that work with pandas DataFrames
            dummy_df_data = {
                'Open': np.random.rand(100) * 100 + 1000,
                'High': np.random.rand(100) * 100 + 1050,
                'Low': np.random.rand(100) * 100 + 950,
                'Close': np.random.rand(100) * 100 + 1000,
                'Volume': np.random.randint(100, 1000, 100)
            }
            dummy_df = pd.DataFrame(dummy_df_data)
            dummy_processed_data_dict = {strategy_config.asset_name: dummy_df}
            dummy_portfolio_ctx = get_dummy_portfolio_context()

            signals_df = strategy_instance.generate_signals(processed_data_dict=dummy_processed_data_dict, portfolio_context=dummy_portfolio_ctx)
            logger.info(f"✓ {strategy_name}.generate_signals() executed. Signals DataFrame columns: {signals_df.columns}, Head:\n{signals_df.head()}")
            # Expected columns: 'signal', 'confidence', etc.

        logger.info(f"--- Test for {strategy_name} (Asset Index: {asset_index}) completed ---\n")

    except Exception as e:
        logger.error(f"❌ Error during test for {strategy_name} (Asset Index: {asset_index}): {e}", exc_info=True)
        logger.info(f"--- Test for {strategy_name} (Asset Index: {asset_index}) failed ---\n")


if __name__ == "__main__":
    logger.info("===== Starting Trend Strategies Debug Script =====")

    # --- Test MomentumStrategy ---
    # Parameters for MomentumStrategy might include 'window', 'signal_threshold', etc.
    # These should match the __init__ of the strategy or be handled by its config.
    # The `params` in `get_dummy_strategy_config` will be merged into the StrategyConfig.
    # The strategy itself will then use these from its `self.config`.
    test_strategy(MomentumStrategy, "MomentumStrategy", default_params={"momentum_window": 20, "input_features_dim": 5, "rsi_period":14, "roc_period":10})
    test_strategy(MomentumStrategy, "MomentumStrategy", default_params={"momentum_window": 10, "input_features_dim": 5, "rsi_period":7, "roc_period":5}, asset_index=1) # Test for a second asset

    # --- Test BreakoutStrategy ---
    # Params: 'lookback_period', 'std_multiplier', etc.
    test_strategy(BreakoutStrategy, "BreakoutStrategy", default_params={"breakout_window": 50, "std_dev_multiplier": 2.0, "input_features_dim": 5, "volume_increase_factor": 1.5})

    # --- Test TrendFollowingStrategy ---
    # Params: 'ma_short_period', 'ma_long_period', 'indicator_type' ('sma', 'ema')
    test_strategy(TrendFollowingStrategy, "TrendFollowingStrategy", default_params={"short_ma_period": 10, "long_ma_period": 50, "input_features_dim": 5, "trend_indicator": "sma"})

    # --- Test ReversalStrategy ---
    # Params: 'reversal_pattern_window', 'confirmation_indicator_period'
    test_strategy(ReversalStrategy, "ReversalStrategy", default_params={"reversal_window": 15, "confirmation_period": 7, "input_features_dim": 5, "overbought_threshold":70, "oversold_threshold":30})

    # --- Test MeanReversionStrategy ---
    # Params: 'mean_reversion_window', 'std_dev_threshold'
    test_strategy(MeanReversionStrategy, "MeanReversionStrategy", default_params={"mean_reversion_window": 30, "std_dev_entry_threshold": 2.0, "std_dev_exit_threshold": 0.5, "input_features_dim": 5})

    logger.info("===== Trend Strategies Debug Script Completed =====")

