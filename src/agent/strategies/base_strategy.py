# src/agent/strategies/base_strategy.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

@dataclass
class StrategyConfig:
    name: str
    description: str = ""
    risk_level: float = 0.5  # 0.0 to 1.0
    market_regime: str = "all"  # e.g., "trending", "ranging", "all"
    complexity: int = 3  # 1 to 5
    base_performance: float = 0.5  # Expected baseline performance
    # Parameters specific to the strategy type, can be overridden by instance params
    default_params: Dict[str, Any] = field(default_factory=dict)
    # List of assets this strategy is applicable to, if empty, applies to all provided.
    applicable_assets: List[str] = field(default_factory=list)


class BaseStrategy(ABC):
    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None):
        self.config = config
        # Instance params override default_params from config's default_params
        # and then specific params passed at instantiation also override.
        # The order is: StrategyConfig.default_params < params_in_config_object < instance_params
        # However, current StrategyConfig only has one default_params.
        # So, it's config.default_params < instance_params
        effective_params = {**self.config.default_params}
        if params: # params passed during instantiation
            effective_params.update(params)
        self.params = effective_params
        # self.strategy_id = config.name # strategy_id is now self.config.name

    @abstractmethod
    def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        """
        Processes raw market data to generate features or processed data.
        Input: Dict of DataFrames, one per asset. Each DF should have 'open', 'high', 'low', 'close', 'volume'.
               Index should be DatetimeIndex.
        Output: Dict of DataFrames (can be a subset of input keys if strategy is asset-specific),
                with added columns for indicators/features. Index should be preserved.
        """
        pass

    @abstractmethod
    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Generates trading signals based on processed data from self.forward().
        Input: Dict of DataFrames as returned by self.forward().
        Output: A single Pandas DataFrame with at least a 'signal' column and a DatetimeIndex.
                The 'signal' column should contain numerical values (e.g., 1 for buy, -1 for sell, 0 for hold).
                The DataFrame should contain signals for a primary asset or an aggregated signal.
                The last row of this DataFrame is typically used by the superposition layer.
                Ensure the index aligns with the input data's timestamps.
        """
        pass

    @property
    def effective_params(self) -> Dict[str, Any]:
        """Alias for params to maintain backward compatibility."""
        return self.params

    def get_strategy_name(self) -> str:
        return self.config.name

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def update_params(self, new_params: Dict[str, Any]):
        # This will overwrite existing keys and add new ones.
        self.params.update(new_params)
        # print(f"Parameters for {self.get_strategy_name()} updated to: {self.params}")

    @classmethod
    def get_class_name(cls) -> str:
        """Returns the class name, useful for mapping in generators/superposition."""
        return cls.__name__

    def _get_primary_symbol(self, data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Helper to determine the primary symbol to operate on.
        Strategies can override this if they have specific logic.
        """
        if self.config.applicable_assets:
            for asset in self.config.applicable_assets:
                if asset in data_dict:
                    return asset
            # If none of the applicable_assets are in market_data_dict, maybe return None or log warning
            # print(f"Warning: None of the applicable_assets for {self.get_strategy_name()} found in market data.")
            return None 
        
        # Fallback: use portfolio_context hint or first available symbol
        if portfolio_context and portfolio_context.get("primary_symbol"):
            if portfolio_context["primary_symbol"] in data_dict:
                return portfolio_context["primary_symbol"]
        
        return next(iter(data_dict)) if data_dict else None

# Example usage (for illustration, not part of the actual BaseStrategy file normally)
if __name__ == '__main__':
    @dataclass
    class MyStrategyConfig(StrategyConfig):
        # Example of adding more specific config fields if needed,
        # though default_params is often flexible enough.
        my_specific_config_param: int = 100

    class MyExampleStrategy(BaseStrategy):
        def __init__(self, config: MyStrategyConfig, params: Optional[Dict[str, Any]] = None):
            # Example: Set default params directly in the class if not using StrategyConfig.default_params extensively
            # This is just one way; StrategyConfig.default_params is preferred for consistency.
            # if 'my_default_lookback' not in (params or {}):
            #     if params is None: params = {}
            #     params['my_default_lookback'] = 10 
            super().__init__(config, params)

        def forward(self, market_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
            print(f"Executing {self.get_strategy_name()} forward pass with params: {self.get_params()}")
            processed_output = {}
            for asset, df in market_data_dict.items():
                if not self.config.applicable_assets or asset in self.config.applicable_assets:
                    df_copy = df.copy()
                    df_copy['processed_feature'] = df_copy['close'].rolling(window=self.params.get('lookback_period', 5)).mean()
                    processed_output[asset] = df_copy
            return processed_output

        def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
            print(f"Executing {self.get_strategy_name()} generate_signals with params: {self.get_params()}")
            
            primary_symbol = self._get_primary_symbol(processed_data_dict, portfolio_context)
            
            if not primary_symbol or primary_symbol not in processed_data_dict:
                print(f"{self.get_strategy_name()}: No data for primary symbol or primary symbol not found.")
                return pd.DataFrame(columns=['signal']) # Return empty DataFrame with 'signal' column

            data_df = processed_data_dict[primary_symbol]
            signals = pd.DataFrame(index=data_df.index, columns=['signal'], dtype=float)
            
            if 'processed_feature' in data_df.columns and not data_df['processed_feature'].empty:
                # Example: Buy if close > processed_feature, Sell if close < processed_feature
                signals['signal'] = 0 # Default to hold
                signals.loc[data_df['close'] > data_df['processed_feature'], 'signal'] = 1
                signals.loc[data_df['close'] < data_df['processed_feature'], 'signal'] = -1
                signals.fillna(0, inplace=True) # Ensure no NaNs in signal column
            else:
                signals['signal'] = 0 # Hold if no feature

            print(f"Signals for {primary_symbol}:\\n{signals.tail()}")
            return signals[['signal']] # Ensure only 'signal' column and correct index

    # Example Instantiation
    # 1. Define a config object
    example_config = MyStrategyConfig(
        name="MyExampleStrategyAlpha",
        description="An example strategy.",
        risk_level=0.3,
        market_regime="trending",
        complexity=2,
        default_params={'lookback_period': 7, 'another_param': 'value1'},
        applicable_assets=['EUR_USD']
    )

    # 2. Instantiate the strategy with the config and optional overriding params
    # Params passed here will override those in example_config.default_params
    strategy_instance = MyExampleStrategy(config=example_config, params={'lookback_period': 10, 'new_param': 'value2'})
    
    print(f"Strategy Name: {strategy_instance.get_strategy_name()}")
    print(f"Strategy Params: {strategy_instance.get_params()}") # Should show lookback_period:10

    # Create dummy market data
    idx = pd.to_datetime(['2023-01-01 10:00', '2023-01-01 10:01', '2023-01-01 10:02', '2023-01-01 10:03', '2023-01-01 10:04', 
                          '2023-01-01 10:05', '2023-01-01 10:06', '2023-01-01 10:07', '2023-01-01 10:08', '2023-01-01 10:09'])
    dummy_market_data = {
        'EUR_USD': pd.DataFrame({
            'open': [1.0, 1.1, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1],
            'high': [1.1, 1.2, 1.3, 1.2, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2],
            'low': [0.9, 1.0, 1.1, 1.0, 0.9, 1.0, 1.1, 1.2, 1.1, 1.0],
            'close': [1.1, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.05],
            'volume': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
        }, index=idx),
        'USD_JPY': pd.DataFrame({
            'close': [110, 111, 112, 111, 110, 109, 110, 111, 112, 113]
        }, index=idx) # Simplified for example
    }
    dummy_portfolio_context = {'cash': 100000, 'primary_symbol': 'EUR_USD'}

    # Test forward pass
    processed_data = strategy_instance.forward(dummy_market_data, dummy_portfolio_context)
    # print("\\nProcessed Data (EUR_USD):")
    # if 'EUR_USD' in processed_data:
    #     print(processed_data['EUR_USD'].tail())

    # Test generate_signals
    final_signals_df = strategy_instance.generate_signals(processed_data, dummy_portfolio_context)
    # print("\\nFinal Signals DataFrame:")
    # print(final_signals_df.tail())
    
    # Test with different params
    strategy_instance.update_params({'lookback_period': 3})
    print(f"Updated Strategy Params: {strategy_instance.get_params()}")
    processed_data_updated = strategy_instance.forward(dummy_market_data, dummy_portfolio_context)
    final_signals_df_updated = strategy_instance.generate_signals(processed_data_updated, dummy_portfolio_context)
    # print("\\nFinal Signals DataFrame (updated params):")
    # print(final_signals_df_updated.tail())
