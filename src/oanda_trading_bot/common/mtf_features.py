import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if 'time' in df.columns:
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        df.set_index('time', inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex or a 'time' column")
    return df


def _mid_from_df(df: pd.DataFrame) -> pd.Series:
    # Try mid columns commonly used
    for c in ('mid_c', 'mid_close', 'mid'):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            if s.notna().any():
                return s
    # Fall back to bid/ask close average if available
    if 'bid_close' in df.columns and 'ask_close' in df.columns:
        b = pd.to_numeric(df['bid_close'], errors='coerce')
        a = pd.to_numeric(df['ask_close'], errors='coerce')
        return (a + b) / 2.0
    # Last resort: any close we find
    for c in ('close', 'ask_c', 'bid_c'):
        if c in df.columns:
            return pd.to_numeric(df[c], errors='coerce')
    raise KeyError("Could not infer mid price from dataframe columns")


def _resample_last(s: pd.Series, freq: str) -> pd.Series:
    return s.resample(freq).last()


def _align_to_index(feature_s: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    # Reindex to target and forward-fill to align lower-frequency features to S5 timeline
    return feature_s.reindex(target_index, method='ffill').fillna(method='bfill')


def compute_mtf_features(
    df_raw: pd.DataFrame,
    timeframes: Optional[List[str]] = None,
    rolling_len: int = 20,
) -> pd.DataFrame:
    """
    Build multi-timeframe features from S5 series.

    Returns a DataFrame indexed by time with columns:
      - log_ret_5s
      - For each timeframe in timeframes (e.g., 1T, 15T, 1H):
        ret_<tf>, z_<tf>, sma_ratio_<tf>
    """
    if timeframes is None:
        timeframes = ['1T', '15T', '1H']

    df = _ensure_datetime_index(df_raw)
    mid = _mid_from_df(df)
    out = pd.DataFrame(index=df.index)

    # Base 5s log return
    eps = 1e-9
    mid_safe = mid.clip(lower=eps)
    out['log_ret_5s'] = np.log(mid_safe / mid_safe.shift(1)).fillna(0.0)

    for tf in timeframes:
        # Resampled close series
        res = _resample_last(mid, tf)
        ret_tf = np.log(res.clip(lower=eps) / res.clip(lower=eps).shift(1))
        # Z-score of resampled returns
        mean_tf = ret_tf.rolling(rolling_len, min_periods=1).mean()
        std_tf = ret_tf.rolling(rolling_len, min_periods=1).std().replace(0, np.nan)
        z_tf = (ret_tf - mean_tf) / std_tf
        # SMA ratio of price vs resampled SMA
        sma_tf = res.rolling(rolling_len, min_periods=1).mean()
        ratio_tf = (mid / _align_to_index(sma_tf, mid.index)).replace([np.inf, -np.inf], np.nan)

        # Align to S5 index
        out[f'ret_{tf}'] = _align_to_index(ret_tf, out.index).fillna(0.0)
        out[f'z_{tf}'] = _align_to_index(z_tf, out.index).fillna(0.0)
        out[f'sma_ratio_{tf}'] = ratio_tf.fillna(method='ffill').fillna(method='bfill').fillna(1.0)

    # Basic volume log feature if present
    if 'volume' in df.columns:
        out['volume_log'] = np.log1p(pd.to_numeric(df['volume'], errors='coerce')).fillna(0.0)

    # Finite only
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out

