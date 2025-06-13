import pandas as pd
import pandas_ta as ta # Import pandas_ta
from enum import Enum
from typing import Dict, Any, Optional 

# NOTE: This module now depends on the pandas-ta library. 
# Ensure it is installed in your environment (e.g., pip install pandas-ta).

class VolatilityLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class TrendStrength(Enum):
    NO_TREND = "No_Trend"
    WEAK_TREND = "Weak_Trend"
    STRONG_TREND = "Strong_Trend"

class MacroRegime(Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    RANGING = "Ranging"
    UNDEFINED = "Undefined" # Added for initial state or insufficient data

class MarketRegimeIdentifier:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the MarketRegimeIdentifier.

        Args:
            config (Dict[str, Any]): Configuration dictionary. Expected keys:
                - "atr_period": int, period for ATR calculation (e.g., 14)
                - "atr_resample_freq": str, pandas resample frequency for ATR (e.g., "1H", "4H")
                - "atr_thresholds": Dict[str, float], thresholds for volatility (e.g., {"low_to_medium": 0.005, "medium_to_high": 0.01})
                - "adx_period": int, period for ADX calculation (e.g., 14)
                - "adx_resample_freq": str, pandas resample frequency for ADX (e.g., "1D")
                - "adx_thresholds": Dict[str, float], thresholds for trend strength (e.g., {"no_to_weak": 20, "weak_to_strong": 25})
                # HMM/GMM related configs can be added later
                # - "hmm_model_path": Optional[str], path to pre-trained HMM model
                # - "hmm_resample_freq": str, pandas resample frequency for HMM input
        """
        self.config = config
        self.hmm_model = None # Placeholder for a loaded HMM model

        # Validate required configurations
        required_keys = [
            "atr_period", "atr_resample_freq", "atr_thresholds",
            "adx_period", "adx_resample_freq", "adx_thresholds"
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        if not isinstance(self.config["atr_thresholds"], dict) or \
           not all(k in self.config["atr_thresholds"] for k in ["low_to_medium", "medium_to_high"]):
            raise ValueError("Invalid atr_thresholds configuration.")

        if not isinstance(self.config["adx_thresholds"], dict) or \
           not all(k in self.config["adx_thresholds"] for k in ["no_to_weak", "weak_to_strong"]):
            raise ValueError("Invalid adx_thresholds configuration.")


    def _resample_ohlcv(self, s5_data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resamples S5 OHLCV data to the specified frequency.
        Assumes s5_data has a DatetimeIndex.
        """
        if not isinstance(s5_data.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex for resampling.")

        resampled_data = pd.DataFrame()
        resampled_data['open'] = s5_data['open'].resample(freq).first()
        resampled_data['high'] = s5_data['high'].resample(freq).max()
        resampled_data['low'] = s5_data['low'].resample(freq).min()
        resampled_data['close'] = s5_data['close'].resample(freq).last()
        resampled_data['volume'] = s5_data['volume'].resample(freq).sum()
        
        # Forward fill to handle non-trading periods if necessary, 
        # but be cautious as this might not be suitable for all indicators.
        # For now, we'll keep NaNs produced by resampling.
        # resampled_data.fillna(method='ffill', inplace=True) 
        return resampled_data.dropna() # Drop rows with NaNs which occur if a period has no S5 data

    def get_volatility_level(self, s5_data: pd.DataFrame) -> VolatilityLevel:
        """
        Calculates the current volatility level based on ATR using pandas-ta.
        """
        resample_freq = self.config["atr_resample_freq"]
        period = self.config["atr_period"]
        thresholds = self.config["atr_thresholds"]

        resampled_data = self._resample_ohlcv(s5_data, resample_freq)
        
        # pandas-ta typically needs at least 'period' data points to calculate ATR.
        # The .dropna() in _resample_ohlcv might reduce rows.
        if len(resampled_data) < period: 
            # print(f"Not enough data for ATR after resampling to {resample_freq}. Have {len(resampled_data)}, need {period}")
            return VolatilityLevel.MEDIUM # Default or Undefined

        # Calculate ATR using pandas-ta
        # Ensure columns are named as expected by pandas-ta (high, low, close)
        atr_series = resampled_data.ta.atr(length=period, high='high', low='low', close='close', append=False)

        atr_value: Optional[float] = None
        if atr_series is not None and not atr_series.empty:
            last_atr = atr_series.iloc[-1]
            if pd.notna(last_atr):
                atr_value = float(last_atr)

        if atr_value is None:
            # print(f"ATR calculation with pandas-ta returned None/NaN for {resample_freq} data.")
            return VolatilityLevel.MEDIUM # Default or Undefined

        if atr_value < thresholds["low_to_medium"]:
            return VolatilityLevel.LOW
        elif atr_value < thresholds["medium_to_high"]:
            return VolatilityLevel.MEDIUM
        else:
            return VolatilityLevel.HIGH

    def get_trend_strength(self, s5_data: pd.DataFrame) -> TrendStrength:
        """
        Calculates the current trend strength based on ADX using pandas-ta.
        """
        resample_freq = self.config["adx_resample_freq"]
        period = self.config["adx_period"]
        thresholds = self.config["adx_thresholds"]

        resampled_data = self._resample_ohlcv(s5_data, resample_freq)

        # ADX calculation often needs more data, e.g., 2 * period or more for reliable values.
        # pandas-ta will return NaNs if data is insufficient.
        if len(resampled_data) < period * 2: # A common heuristic for minimum data for ADX
            # print(f"Not enough data for ADX after resampling to {resample_freq}. Have {len(resampled_data)}, need {period*2}")
            return TrendStrength.NO_TREND # Default or Undefined

        # Calculate ADX using pandas-ta
        # Ensure columns are named as expected (high, low, close)
        adx_df = resampled_data.ta.adx(length=period, high='high', low='low', close='close', append=False)
        
        adx_value: Optional[float] = None
        adx_col_name = f'ADX_{period}'
        if adx_df is not None and not adx_df.empty and adx_col_name in adx_df.columns:
            last_adx = adx_df[adx_col_name].iloc[-1]
            if pd.notna(last_adx):
                adx_value = float(last_adx)
        
        if adx_value is None:
            # print(f"ADX calculation with pandas-ta returned None/NaN for {resample_freq} data.")
            return TrendStrength.NO_TREND # Default or Undefined

        if adx_value < thresholds["no_to_weak"]:
            return TrendStrength.NO_TREND
        elif adx_value < thresholds["weak_to_strong"]:
            return TrendStrength.WEAK_TREND
        else:
            return TrendStrength.STRONG_TREND

    def get_macro_regime(self, s5_data: pd.DataFrame) -> MacroRegime:
        """
        Identifies the macro market regime (e.g., Bullish, Bearish, Ranging).
        Placeholder: Currently returns RANGING.
        This would involve HMM/GMM or other macro trend analysis.
        """
        # resample_freq = self.config.get("hmm_resample_freq", "1D") # Example
        # resampled_data = self._resample_ohlcv(s5_data, resample_freq)
        # if self.hmm_model and len(resampled_data) > some_min_length:
        #     # Preprocess data for HMM
        #     # state = self.hmm_model.predict(processed_data)
        #     # return self._map_hmm_state_to_macro_regime(state[-1])
        #     pass
        return MacroRegime.RANGING # Default placeholder

    def get_current_regime(self, s5_data: pd.DataFrame) -> Dict[str, Enum]:
        """
        Determines the overall current market regime by combining different analyses.

        Args:
            s5_data (pd.DataFrame): DataFrame with 5-second OHLCV data.
                                    Must have a DatetimeIndex.
                                    Columns: ['open', 'high', 'low', 'close', 'volume']

        Returns:
            Dict[str, Enum]: A dictionary containing the identified market regimes.
                e.g., {
                    "macro_regime": MacroRegime.BULLISH,
                    "volatility_level": VolatilityLevel.MEDIUM,
                    "trend_strength": TrendStrength.STRONG_TREND
                }
        """
        if not isinstance(s5_data, pd.DataFrame) or not all(col in s5_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("Invalid s5_data input. Must be a DataFrame with OHLCV columns.")
        if not isinstance(s5_data.index, pd.DatetimeIndex):
            raise ValueError("s5_data must have a DatetimeIndex.")
        if s5_data.empty:
            # print("Warning: s5_data is empty. Returning default regime.")
            return {
                "macro_regime": MacroRegime.UNDEFINED,
                "volatility_level": VolatilityLevel.MEDIUM, # Default
                "trend_strength": TrendStrength.NO_TREND    # Default
            }
            
        volatility = self.get_volatility_level(s5_data.copy()) # Use .copy() to avoid SettingWithCopyWarning on resamples
        trend = self.get_trend_strength(s5_data.copy())
        macro = self.get_macro_regime(s5_data.copy()) # Currently placeholder

        return {
            "macro_regime": macro,
            "volatility_level": volatility,
            "trend_strength": trend
        }

# Example Usage (for testing purposes, normally this would be instantiated and used by the agent)
if __name__ == '__main__':
    # Create sample S5 data (replace with actual data loading)
    # Ensure DatetimeIndex for resampling
    rng = pd.date_range('2023-01-01', periods=24 * 60 * 12 * 2, freq='5S') # 2 days of 5S data
    data = {
        'open': [i/1000 + 20 for i in range(len(rng))],
        'high': [i/1000 + 20.05 for i in range(len(rng))],
        'low': [i/1000 + 19.95 for i in range(len(rng))],
        'close': [i/1000 + 20 for i in range(len(rng))],
        'volume': [100 + i % 50 for i in range(len(rng))]
    }
    sample_s5_df = pd.DataFrame(data, index=rng)

    config = {
        "atr_period": 14,
        "atr_resample_freq": "1H", # Resample to 1-hour for ATR
        "atr_thresholds": {"low_to_medium": 0.02, "medium_to_high": 0.05}, # Example thresholds
        "adx_period": 14,
        "adx_resample_freq": "4H", # Resample to 4-hour for ADX
        "adx_thresholds": {"no_to_weak": 20, "weak_to_strong": 25} # Standard ADX thresholds
    }

    regime_identifier = MarketRegimeIdentifier(config)
    
    # Test with sufficient data
    current_regime = regime_identifier.get_current_regime(sample_s5_df)
    print(f"Current Market Regime (sufficient data): {current_regime}")

    # Test with insufficient data for one of the resamplings (e.g., less than 14 hours for 1H ATR)
    short_rng = pd.date_range('2023-01-01', periods=10 * 60 * 12, freq='5S') # 10 hours of 5S data
    short_s5_df = pd.DataFrame(data, index=short_rng[:len(short_rng)]) # Ensure data matches index length
    
    current_regime_short = regime_identifier.get_current_regime(short_s5_df)
    print(f"Current Market Regime (short data): {current_regime_short}")

    # Test with empty data
    empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'], index=pd.to_datetime([]))
    current_regime_empty = regime_identifier.get_current_regime(empty_df)
    print(f"Current Market Regime (empty data): {current_regime_empty}")
    
    # Test with invalid config (missing key)
    try:
        invalid_config = config.copy()
        del invalid_config["atr_period"]
        MarketRegimeIdentifier(invalid_config)
    except ValueError as e:
        print(f"Caught expected error for invalid config: {e}")

    # Test with invalid threshold config
    try:
        invalid_config_thresh = config.copy()
        invalid_config_thresh["atr_thresholds"] = {"low": 0.1} # Missing keys
        MarketRegimeIdentifier(invalid_config_thresh)
    except ValueError as e:
        print(f"Caught expected error for invalid threshold config: {e}")

    # Test _resample_ohlcv with non-DatetimeIndex
    try:
        non_dt_index_df = pd.DataFrame({'open': [1,2], 'high': [1,2], 'low': [1,2], 'close': [1,2], 'volume': [1,2]})
        regime_identifier._resample_ohlcv(non_dt_index_df, "1H")
    except ValueError as e:
        print(f"Caught expected error for non-DatetimeIndex: {e}")

