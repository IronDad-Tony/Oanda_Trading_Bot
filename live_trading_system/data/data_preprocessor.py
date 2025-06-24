# live_trading_system/data/data_preprocessor.py
"""
實時數據預處理器

負責將從 Oanda API 獲取的原始價量數據，轉換為與訓練時一致的、
可直接輸入模型的特徵。
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any
from datetime import datetime, timezone, timedelta
import logging

logger = logging.getLogger("LiveTradingSystem")

class LivePreprocessor:
    """
    對即時數據進行預處理，使其符合模型輸入要求。
    
    此預處理器嚴格複製訓練階段的特徵工程、標準化和數據處理邏輯，
    以確保模型在推論時接收到的數據分佈與訓練時完全一致。
    它使用在訓練階段保存的 scaler 參數 (mean, std) 來對新數據進行標準化。
    """
    def __init__(self, scalers_path: str, config: Dict[str, Any]):
        """
        初始化預處理器。
        
        Args:
            scalers_path (str): 指向保存了 scaler 參數的 JSON 檔案路徑。
            config (Dict[str, Any]): 系統設定檔。
        """
        self.scalers = self._load_scalers(scalers_path)
        self.freshness_threshold_minutes = config.get('data_freshness_threshold_minutes', 10)
        self.model_lookback_window = config.get('model_lookback_window', 128)
        logger.info(f"預處理器已成功從 {scalers_path} 載入 Scaler 參數。")
        logger.info(f"數據新鮮度檢查閾值設定為 {self.freshness_threshold_minutes} 分鐘。")
        logger.info(f"模型所需的回看窗口長度為 {self.model_lookback_window}。")

    def _load_scalers(self, path: str) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        """
        從 JSON 檔案載入每個 symbol 的均值和標準差。
        預期的 JSON 格式: 
        {
            "EUR_USD": {
                "bid_close_log_ret": {"mean": [0.1], "scale": [1.2]}, 
                ...
            },
            "USD_JPY": { ... }
        }
        """
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Scaler 檔案未找到: {path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"無法解析 Scaler 檔案: {path}")
            raise
        except Exception as e:
            logger.error(f"載入 Scaler 檔案時發生未知錯誤: {e}")
            raise

    def _calculate_log_returns(self, series: pd.Series, epsilon: float = 1e-9) -> pd.Series:
        """計算單個價格序列的對數回報率，並處理無效值，與訓練代碼一致。"""
        shifted_series = series.shift(1)
        valid_mask = (series > epsilon) & (shifted_series > epsilon)
        log_ret = pd.Series(np.nan, index=series.index, dtype=np.float64)
        log_ret[valid_mask] = np.log(series[valid_mask] / shifted_series[valid_mask])
        return log_ret.fillna(0.0)

    def _is_data_fresh(self, last_candle_time: pd.Timestamp) -> bool:
        """檢查數據是否新鮮。"""
        if last_candle_time.tzinfo is None:
            last_candle_time = last_candle_time.tz_localize('UTC')
        
        current_time_utc = datetime.now(timezone.utc)
        time_diff = current_time_utc - last_candle_time
        
        is_fresh = time_diff <= timedelta(minutes=self.freshness_threshold_minutes)
        if not is_fresh:
            logger.warning(
                f"數據新鮮度檢查失敗！最新 K 線時間: {last_candle_time}, "
                f"當前時間: {current_time_utc}, 差距: {time_diff}. "
                f"超過 {self.freshness_threshold_minutes} 分鐘的閾值。"
            )
        return is_fresh

    def transform(self, raw_candles: List[Dict[str, Any]], instrument: str) -> np.ndarray:
        """
        將從 Oanda API 獲取的原始蠟燭圖數據列表轉換為模型輸入的 NumPy 陣列。
        此方法現在完全複製訓練時的特徵工程和標準化流程。
        """
        if not raw_candles or len(raw_candles) < 2: # 計算回報率至少需要2個點
            logger.warning(f"[{instrument}] 原始蠟燭數據不足 ({len(raw_candles)}條)，無法進行轉換。")
            return np.array([])

        # 1. 將原始蠟燭圖轉換為 DataFrame
        df = pd.DataFrame(raw_candles)
        if 'mid' in df.columns and isinstance(df['mid'].iloc[0], dict):
            df = df.join(pd.json_normalize(df['mid']).add_prefix('mid_'))
        if 'bid' in df.columns and isinstance(df['bid'].iloc[0], dict):
            df = df.join(pd.json_normalize(df['bid']).add_prefix('bid_'))
        if 'ask' in df.columns and isinstance(df['ask'].iloc[0], dict):
            df = df.join(pd.json_normalize(df['ask']).add_prefix('ask_'))
        
        rename_map = {
            'mid_o': 'mid_open', 'mid_h': 'mid_high', 'mid_l': 'mid_low', 'mid_c': 'mid_close',
            'bid_o': 'bid_open', 'bid_h': 'bid_high', 'bid_l': 'bid_low', 'bid_c': 'bid_close',
            'ask_o': 'ask_open', 'ask_h': 'ask_high', 'ask_l': 'ask_low', 'ask_c': 'ask_close',
        }
        df.rename(columns=rename_map, inplace=True)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        # 數據新鮮度檢查
        if not self._is_data_fresh(df.index[-1]):
            return np.array([])

        # 2. 特徵工程 (完全鏡像 `src/feature_engineer/preprocessor.py`)
        processed_df = pd.DataFrame(index=df.index)
        
        price_feature_cols = []
        # 訓練時主要使用 bid/ask，這裡保持一致
        for col_type in ['bid', 'ask']:
            for ohlc in ['open', 'high', 'low', 'close']:
                price_col = f"{col_type}_{ohlc}"
                if price_col in df.columns:
                    return_col_name = f"{price_col}_log_ret"
                    processed_df[return_col_name] = self._calculate_log_returns(df[price_col])
                    price_feature_cols.append(return_col_name)

        volume_feature_col = None
        if 'volume' in df.columns:
            volume_col_name = "volume_log"
            processed_df[volume_col_name] = np.log1p(df['volume'])
            volume_feature_col = volume_col_name

        features_to_process = price_feature_cols + ([volume_feature_col] if volume_feature_col else [])
        
        if not features_to_process:
            logger.warning(f"[{instrument}] 沒有生成任何特徵。")
            return np.array([])

        # 3. 使用已載入的 Scaler 進行標準化
        instrument_scalers = self.scalers.get(instrument)
        if not instrument_scalers:
            logger.error(f"[{instrument}] 找不到該交易對的 Scaler。無法進行標準化。")
            raise ValueError(f"Scaler for {instrument} not found in scalers file.")

        expected_feature_order = list(instrument_scalers.keys())

        for feature_col in expected_feature_order:
            if feature_col in processed_df.columns:
                scaler_params = instrument_scalers[feature_col]
                mean = scaler_params.get('mean', [0.0])[0] # 從list中獲取
                scale = scaler_params.get('scale', [1.0])[0] # 從list中獲取

                if scale < 1e-9: scale = 1.0 # 防止除以零

                series_to_scale = processed_df[feature_col].values
                standardized_series = (series_to_scale - mean) / scale
                
                # 4. 裁剪 (與訓練時一致)
                processed_df[feature_col] = np.clip(standardized_series, -5.0, 5.0)
            else:
                logger.warning(f"[{instrument}] 預期特徵 '{feature_col}' 未在處理過程中生成，將以 0 填充。")
                processed_df[feature_col] = 0.0
        
        # 確保所有預期特徵都存在且順序正確
        final_df = processed_df[expected_feature_order]

        # 移除因計算log return而在開頭產生的NaN值
        final_df.dropna(inplace=True)

        if final_df.empty:
            logger.warning(f"[{instrument}] 預處理後數據為空（可能因dropna）。")
            return np.array([])
            
        # 5. 確保返回的數據長度與模型回看窗口一致
        if len(final_df) > self.model_lookback_window:
            final_df = final_df.tail(self.model_lookback_window)
        elif len(final_df) < self.model_lookback_window:
            logger.warning(f"[{instrument}] 預處理後的數據長度 ({len(final_df)}) 小於模型所需的回看窗口 ({self.model_lookback_window})。上層調用者需處理填充。")
            # 在此處不進行填充，讓上層調用者(如PredictionService)決定如何處理長度不足的情況
            # 因為可能需要組合多個標的，統一進行padding
            pass

        logger.info(f"[{instrument}] 成功將 {len(raw_candles)} 條蠟燭數據轉換為 {final_df.shape[0]}x{final_df.shape[1]} 的特徵矩陣。")
        
        return final_df.values

# --- 範例 ---
if __name__ == '__main__':
    # 假設我們有一個從訓練中保存的 scaler 檔案
    dummy_scalers = {
        "EUR_USD": {
            "bid_close_log_ret": {"mean": [1.2e-05], "scale": [0.00015]},
            "volume_log": {"mean": [8.5], "scale": [1.2]}
        },
        "USD_JPY": {
            "bid_close_log_ret": {"mean": [1.1e-05], "scale": [0.0001]},
            "volume_log": {"mean": [7.5], "scale": [1.1]}
        }
    }
    scaler_file = "dummy_scalers.json"
    with open(scaler_file, 'w') as f:
        json.dump(dummy_scalers, f)

    # 創建一個預處理器實例
    config = {"data_freshness_threshold_minutes": 10, "model_lookback_window": 128}
    preprocessor = LivePreprocessor(scaler_file, config)

    # 模擬從 Oanda API 收到的數據
    mock_api_data = [
        {'complete': True, 'volume': 1500, 'time': '2025-06-24T10:00:00.000000000Z', 'mid': {'o': '1.07000', 'h': '1.07010', 'l': '1.06990', 'c': '1.07005'}},
        {'complete': True, 'volume': 1800, 'time': '2025-06-24T10:05:00.000000000Z', 'mid': {'o': '1.07005', 'h': '1.07025', 'l': '1.07000', 'c': '1.07020'}},
        {'complete': True, 'volume': 1650, 'time': '2025-06-24T10:10:00.000000000Z', 'mid': {'o': '1.07020', 'h': '1.07030', 'l': '1.07015', 'c': '1.07018'}},
    ]

    # 進行轉換
    processed_data = preprocessor.transform(mock_api_data, "EUR_USD")

    logger.info("轉換後的數據 (Numpy Array):")
    logger.info(processed_data)
    logger.info(f"形狀: {processed_data.shape}")

    # 清理 dummy 檔案
    import os
    os.remove(scaler_file)
