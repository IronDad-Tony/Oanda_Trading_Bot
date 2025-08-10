import pytest
import pandas as pd
from src.market_analysis.market_regime_identifier import (
    MarketRegimeIdentifier, 
    VolatilityLevel, 
    TrendStrength,
    MacroRegime
)
from typing import Dict, Any

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    return {
        "atr_period": 14,
        "atr_resample_freq": "1h", # Changed to lowercase 'h'
        "atr_thresholds": {"low_to_medium": 0.002, "medium_to_high": 0.005}, 
        "adx_period": 14,
        "adx_resample_freq": "4h", # Changed to lowercase 'h'
        "adx_thresholds": {"no_to_weak": 20, "weak_to_strong": 25}
    }

@pytest.fixture
def sample_s5_data_sufficient() -> pd.DataFrame:
    # ~2 days of 5S data, should be enough for 1H and 4H resampling for ATR/ADX periods
    periods = 2 * 24 * 60 * 12  # 2 days * 24 hours/day * 60 minutes/hour * 12 (5s periods per minute)
    rng = pd.date_range('2023-01-01', periods=periods, freq='5s') # Changed to lowercase 's'
    data = {
        'open': [20 + (i * 0.00001) for i in range(periods)],
        'high': [20.05 + (i * 0.00001) for i in range(periods)],
        'low': [19.95 + (i * 0.00001) for i in range(periods)],
        'close': [20 + (i * 0.00001) for i in range(periods)],
        'volume': [100 + (i % 50) for i in range(periods)]
    }
    return pd.DataFrame(data, index=rng)

@pytest.fixture
def sample_s5_data_insufficient_atr() -> pd.DataFrame:
    # 10 hours of 5S data, insufficient for ATR period of 14 (1H bars)
    periods = 10 * 60 * 12 
    rng = pd.date_range('2023-01-01', periods=periods, freq='5s') # Changed to lowercase 's'
    data = {
        'open': [20 + (i * 0.00001) for i in range(periods)],
        'high': [20.05 + (i * 0.00001) for i in range(periods)],
        'low': [19.95 + (i * 0.00001) for i in range(periods)],
        'close': [20 + (i * 0.00001) for i in range(periods)],
        'volume': [100 + (i % 50) for i in range(periods)]
    }
    return pd.DataFrame(data, index=rng)

@pytest.fixture
def sample_s5_data_insufficient_adx() -> pd.DataFrame:
    # 2 days of 5S data, but ADX period is 14 (4H bars). 2 days = 12 4H bars. Need 14*2=28 for placeholder ADX.
    # This should be enough for ATR (1H bars), but not for ADX (4H bars * 2 for placeholder).
    periods = 2 * 24 * 60 * 12 
    rng = pd.date_range('2023-01-01', periods=periods, freq='5s') # Changed to lowercase 's'
    data = {
        'open': [20 + (i * 0.00001) for i in range(periods)],
        'high': [20.05 + (i * 0.00001) for i in range(periods)],
        'low': [19.95 + (i * 0.00001) for i in range(periods)],
        'close': [20 + (i * 0.00001) for i in range(periods)],
        'volume': [100 + (i % 50) for i in range(periods)]
    }
    # Modify config for this test to make ADX require more data than available
    # Default ADX period 14 (4H bars). 2 days = 48H = 12 4H bars. Placeholder needs period*2 = 28 bars.
    # So this data is insufficient for the placeholder ADX.
    return pd.DataFrame(data, index=rng)

@pytest.fixture
def empty_s5_data() -> pd.DataFrame:
    return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'], index=pd.to_datetime([]))


def test_market_regime_identifier_init(sample_config):
    identifier = MarketRegimeIdentifier(sample_config)
    assert identifier.config == sample_config

def test_market_regime_identifier_init_invalid_config():
    with pytest.raises(ValueError, match="Missing required configuration key: atr_period"):
        MarketRegimeIdentifier({"adx_period": 14}) # Missing atr_period
    with pytest.raises(ValueError, match="Invalid atr_thresholds configuration."):
        MarketRegimeIdentifier({
            "atr_period": 14, "atr_resample_freq": "1H", "atr_thresholds": {"low": 0.1}, # Invalid keys
            "adx_period": 14, "adx_resample_freq": "4H", "adx_thresholds": {"no_to_weak": 20, "weak_to_strong": 25}
        })
    with pytest.raises(ValueError, match="Invalid adx_thresholds configuration."):
        MarketRegimeIdentifier({
            "atr_period": 14, "atr_resample_freq": "1H", "atr_thresholds": {"low_to_medium": 0.1, "medium_to_high":0.2},
            "adx_period": 14, "adx_resample_freq": "4H", "adx_thresholds": {"no": 20} # Invalid keys
        })


def test_resample_ohlcv(sample_config, sample_s5_data_sufficient):
    identifier = MarketRegimeIdentifier(sample_config)
    resampled_1h = identifier._resample_ohlcv(sample_s5_data_sufficient, "1h") # Changed to lowercase '1h'
    assert not resampled_1h.empty
    assert isinstance(resampled_1h.index, pd.DatetimeIndex)
    assert resampled_1h.index.freqstr in [None, 'h', '1h'] # Adjusted to lowercase 'h' and '1h'
    # Expected number of 1H bars from 2 days of data is 48.
    # It might be slightly less if the first/last partial period doesn't form a full bar or if dropna() removes some.
    assert len(resampled_1h) <= 48 
    assert len(resampled_1h) > 40 # Should be close to 48

    resampled_4h = identifier._resample_ohlcv(sample_s5_data_sufficient, "4h") # Changed to lowercase '4h'
    assert not resampled_4h.empty
    assert isinstance(resampled_4h.index, pd.DatetimeIndex)
    assert len(resampled_4h) <= 12 # 2 days = 12 4H bars
    assert len(resampled_4h) > 10 # Should be close to 12

def test_resample_ohlcv_invalid_input(sample_config):
    identifier = MarketRegimeIdentifier(sample_config)
    with pytest.raises(ValueError, match="Input DataFrame must have a DatetimeIndex"):
        identifier._resample_ohlcv(pd.DataFrame({'open': [1]}), "1H")

# Since _calculate_atr and _calculate_adx are placeholders using TA-Lib or similar is key.
# These tests will be very basic for the placeholder logic.

def test_get_volatility_level_sufficient_data(sample_config, sample_s5_data_sufficient):
    identifier = MarketRegimeIdentifier(sample_config)
    # The placeholder _calculate_atr is very naive. We expect it to return *a* value.
    # The exact VolatilityLevel depends on the naive calculation and thresholds.
    # This test mainly checks that it runs without error and returns a valid Enum member.
    volatility = identifier.get_volatility_level(sample_s5_data_sufficient)
    assert isinstance(volatility, VolatilityLevel)

def test_get_volatility_level_insufficient_data(sample_config, sample_s5_data_insufficient_atr):
    identifier = MarketRegimeIdentifier(sample_config)
    volatility = identifier.get_volatility_level(sample_s5_data_insufficient_atr)
    # Default for insufficient data
    assert volatility == VolatilityLevel.MEDIUM 

def test_get_trend_strength_sufficient_data(sample_config, sample_s5_data_sufficient):
    # For this test, we need enough data for the ADX placeholder (period * 2)
    # sample_s5_data_sufficient (2 days) resampled to 4H gives 12 bars. ADX period 14. Placeholder needs 28.
    # So, this will actually hit the insufficient data path for the *placeholder* ADX.
    # To test the "sufficient" path for the placeholder, we'd need much more data or smaller ADX period/freq.
    # Let's adjust the config for this specific test to make data sufficient for placeholder ADX.
    test_specific_config = sample_config.copy()
    test_specific_config["adx_period"] = 5 # 5 * 2 = 10 bars needed. 2 days @ 4H = 12 bars. Sufficient.
    identifier = MarketRegimeIdentifier(test_specific_config)
    trend = identifier.get_trend_strength(sample_s5_data_sufficient)
    assert isinstance(trend, TrendStrength)

def test_get_trend_strength_insufficient_data(sample_config, sample_s5_data_insufficient_adx):
    # sample_s5_data_insufficient_adx (2 days) with adx_period=14 (4H) -> 12 bars, placeholder needs 28.
    identifier = MarketRegimeIdentifier(sample_config)
    trend = identifier.get_trend_strength(sample_s5_data_insufficient_adx)
    # Default for insufficient data
    assert trend == TrendStrength.NO_TREND

def test_get_macro_regime_placeholder(sample_config, sample_s5_data_sufficient):
    identifier = MarketRegimeIdentifier(sample_config)
    macro = identifier.get_macro_regime(sample_s5_data_sufficient)
    assert macro == MacroRegime.RANGING # Current placeholder behavior

def test_get_current_regime_sufficient_data(sample_config, sample_s5_data_sufficient):
    # Similar to trend strength, adjust ADX config for placeholder if needed
    test_specific_config = sample_config.copy()
    test_specific_config["adx_period"] = 5 # Make ADX calc likely to succeed with placeholder
    identifier = MarketRegimeIdentifier(test_specific_config)
    
    regime = identifier.get_current_regime(sample_s5_data_sufficient)
    assert isinstance(regime["volatility_level"], VolatilityLevel)
    assert isinstance(regime["trend_strength"], TrendStrength)
    assert regime["macro_regime"] == MacroRegime.RANGING # Placeholder

def test_get_current_regime_empty_data(sample_config, empty_s5_data):
    identifier = MarketRegimeIdentifier(sample_config)
    regime = identifier.get_current_regime(empty_s5_data)
    assert regime["macro_regime"] == MacroRegime.UNDEFINED
    assert regime["volatility_level"] == VolatilityLevel.MEDIUM
    assert regime["trend_strength"] == TrendStrength.NO_TREND

def test_get_current_regime_invalid_data_format(sample_config):
    identifier = MarketRegimeIdentifier(sample_config)
    with pytest.raises(ValueError, match="Invalid s5_data input"):
        identifier.get_current_regime(pd.DataFrame({'foo': [1]}))
    
    with pytest.raises(ValueError, match="s5_data must have a DatetimeIndex"):
        identifier.get_current_regime(pd.DataFrame({'open': [1], 'high': [1], 'low': [1], 'close': [1], 'volume': [1]}))

# It's important to note that these tests are for the *structure* and placeholder logic.
# Once real TA calculations (e.g., from pandas_ta) are integrated,
# tests would need to be more sophisticated, possibly using known data points
# or comparing against trusted implementations if exact values are hard to pin down.
