# live_trading_system/data/data_preprocessor.py
"""
實時數據預處理器

負責將從 Oanda API 獲取的原始價量數據，轉換為與訓練時一致的、
可直接輸入模型的特徵。

此版本整合了來自 `src/feature_engineer/preprocessor.py` 的核心邏輯，
特別是滑動窗口標準化，以確保與訓練過程最大程度的一致性。
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import logging
import pywt  # For Wavelet features
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("LiveTradingSystem")

class LivePreprocessor:
    """
    對即時數據進行預處理，使其符合模型輸入要求。
    """
    def __init__(self, scalers_path: str, config: Dict[str, Any], model_config_path: str):
        """
        初始化預處理器。
        
        Args:
            scalers_path (str): 指向保存了 scaler 參數的 JSON 檔案路徑。
            config (Dict[str, Any]): 即時系統設定檔 (live_config.json)。
            model_config_path (str): 指向模型架構設定檔的路徑 (e.g., enhanced_model_config.json)。
        """
        self.scalers = self._load_scalers(scalers_path)
        self.config = config
        self.model_config = self._load_model_config(model_config_path)
        
        self.freshness_threshold_minutes = self.config.get('data_freshness_threshold_minutes', 10)
        self.model_lookback_window = self.model_config.get('max_sequence_length', 128) # 從模型配置讀取
        
        logger.info(f"預處理器已成功從 {scalers_path} 載入 Scaler 參數。")
        logger.info(f"模型所需的回看窗口長度為 {self.model_lookback_window}。")
        logger.info(f"模型設定檔載入成功，將生成對應特徵。")

    def _load_scalers(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"載入 Scaler 檔案時發生錯誤: {e}", exc_info=True)
            raise

    def _load_model_config(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"載入模型設定檔時發生錯誤: {e}", exc_info=True)
            raise

    def _calculate_log_returns(self, series: pd.Series, epsilon: float = 1e-9) -> pd.Series:
        shifted_series = series.shift(1)
        valid_mask = (series > epsilon) & (shifted_series > epsilon)
        log_ret = pd.Series(np.nan, index=series.index, dtype=np.float64)
        log_ret[valid_mask] = np.log(series[valid_mask] / shifted_series[valid_mask])
        return log_ret.fillna(0.0)

    def _add_fourier_features(self, series: pd.Series, n_harmonics: int = 10) -> pd.DataFrame:
        fourier_features = pd.DataFrame(index=series.index)
        t = np.arange(len(series))
        for i in range(1, n_harmonics + 1):
            fourier_features[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * t / len(t))
            fourier_features[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * t / len(t))
        return fourier_features

    def _add_wavelet_features(self, series: pd.Series, wavelet='db4', level=4) -> pd.DataFrame:
        coeffs = pywt.wavedec(series, wavelet, level=level)
        # Pad coefficients to match original length
        coeffs_padded = [np.pad(c, (0, len(series) - len(c)), 'constant') for c in coeffs]
        wavelet_features = pd.DataFrame(np.vstack(coeffs_padded).T, index=series.index, 
                                        columns=[f'wavelet_coeff_{i}' for i in range(len(coeffs))])
        return wavelet_features

    def _is_data_fresh(self, last_candle_time: pd.Timestamp) -> bool:
        # Implementation from old preprocessor
        if last_candle_time.tzinfo is None:
            last_candle_time = last_candle_time.tz_localize('UTC')
        current_time_utc = datetime.now(timezone.utc)
        time_diff = current_time_utc - last_candle_time
        is_fresh = time_diff <= timedelta(minutes=self.freshness_threshold_minutes)
        if not is_fresh:
            logger.warning(f"數據新鮮度檢查失敗！最新 K 線時間: {last_candle_time}, "
                           f"當前時間: {current_time_utc}, 差距: {time_diff}.")
        return is_fresh

    def transform(self, raw_candles: List[Dict[str, Any]], instrument: str) -> np.ndarray:
        if not raw_candles or len(raw_candles) < self.model_lookback_window:
            logger.warning(f"[{instrument}] 原始蠟燭數據不足 ({len(raw_candles)}條)，需要 {self.model_lookback_window} 條。")
            return np.array([])

        # 1. Convert to DataFrame
        df = pd.DataFrame(raw_candles)
        if 'mid' in df.columns and isinstance(df['mid'].iloc[0], dict):
            df = df.join(pd.json_normalize(df['mid']).add_prefix('mid_'))
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        if not self._is_data_fresh(df.index[-1]):
            return np.array([])

        # 2. Feature Engineering
        processed_df = pd.DataFrame(index=df.index)
        
        # Basic features
        processed_df['log_ret'] = self._calculate_log_returns(df['mid_c'])
        processed_df['volume_log'] = np.log1p(df['volume'])

        # Advanced features based on model config
        if self.model_config.get('use_fourier_features'):
            fourier_df = self._add_fourier_features(df['mid_c'])
            processed_df = pd.concat([processed_df, fourier_df], axis=1)

        if self.model_config.get('use_wavelet_features'):
            wavelet_df = self._add_wavewavelet_features(df['mid_c'])
            processed_df = pd.concat([processed_df, wavelet_df], axis=1)
            
        # Drop rows with NaN from feature calculation
        processed_df.dropna(inplace=True)
        
        # 3. Standardization
        instrument_scalers = self.scalers.get(instrument)
        if not instrument_scalers:
            raise ValueError(f"Scaler for {instrument} not found.")

        expected_feature_order = list(instrument_scalers.keys())
        
        for feature_col in expected_feature_order:
            if feature_col in processed_df.columns:
                scaler_params = instrument_scalers[feature_col]
                mean = scaler_params.get('mean', [0.0])[0]
                scale = scaler_params.get('scale', [1.0])[0]
                if scale < 1e-9: scale = 1.0
                
                processed_df[feature_col] = (processed_df[feature_col] - mean) / scale
                processed_df[feature_col] = np.clip(processed_df[feature_col], -5.0, 5.0)
            else:
                logger.warning(f"[{instrument}] 預期特徵 '{feature_col}' 未生成，將以 0 填充。")
                processed_df[feature_col] = 0.0
        
        final_df = processed_df[expected_feature_order]

        # 4. Ensure correct window size
        if len(final_df) > self.model_lookback_window:
            final_df = final_df.tail(self.model_lookback_window)
        elif len(final_df) < self.model_lookback_window:
            logger.error(f"[{instrument}] 預處理後數據長度 ({len(final_df)}) 小於模型所需 ({self.model_lookback_window})。")
            return np.array([])

        logger.info(f"[{instrument}] 成功轉換數據，最終特徵矩陣形狀: {final_df.shape}。")
        return final_df.values