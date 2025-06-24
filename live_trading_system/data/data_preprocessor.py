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
from sklearn.preprocessing import StandardScaler
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger("LiveTradingSystem")

class LivePreprocessor:
    """
    對即時數據進行預處理，使其符合模型輸入要求。
    
    關鍵在於，此預處理器使用在訓練階段就已保存的 scaler 參數 (mean, std)
    來對新數據進行標準化，而不是對新數據進行擬合 (fit)。
    """
    def __init__(self, scalers_path: str, config: Dict[str, Any]):
        """
        初始化預處理器。
        
        Args:
            scalers_path (str): 指向保存了 scaler 參數 (mean, std) 的 JSON 檔案路徑。
            config (Dict[str, Any]): 系統設定檔，用於獲取新鮮度閾值。
        """
        self.scalers = self._load_scalers(scalers_path)
        self.freshness_threshold_minutes = config.get('data_freshness_threshold_minutes', 10)
        logger.info(f"預處理器已成功從 {scalers_path} 載入 Scaler 參數。")
        logger.info(f"數據新鮮度檢查閾值設定為 {self.freshness_threshold_minutes} 分鐘。")

    def _load_scalers(self, path: str) -> Dict[str, Dict[str, float]]:
        """
        從 JSON 檔案載入均值和標準差。
        預期的 JSON 格式: {"feature_name": {"mean": 0.1, "std": 1.2}, ...}
        """
        try:
            with open(path, 'r') as f:
                scalers_json = json.load(f)
            # 將 list 轉換為 numpy array
            scalers = {}
            for feature, params in scalers_json.items():
                scalers[feature] = {
                    'mean': np.array(params['mean']),
                    'scale': np.array(params['scale'])
                }
            return scalers
        except FileNotFoundError:
            logger.error(f"Scaler 檔案未找到: {path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"無法解析 Scaler 檔案: {path}")
            raise
        except Exception as e:
            logger.error(f"載入 Scaler 檔案時發生未知錯誤: {e}")
            raise

    def _calculate_log_returns(self, series: pd.Series) -> pd.Series:
        """計算單個價格序列的對數回報率。 log(price_t / price_{t-1})"""
        return np.log(series / series.shift(1)).fillna(0.0)

    def _is_data_fresh(self, last_candle_time: pd.Timestamp) -> bool:
        """
        檢查數據是否新鮮。

        Args:
            last_candle_time (pd.Timestamp): 最新一根 K 線的時間。

        Returns:
            bool: 如果數據在設定的閾值內則返回 True，否則返回 False。
        """
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

    def transform(self, raw_candles: List[Dict[str, Any]]) -> np.ndarray:
        """
        將從 Oanda API 獲取的原始蠟燭圖數據列表轉換為模型輸入的 NumPy 陣列。

        Args:
            raw_candles (List[Dict[str, Any]]): Oanda API 返回的蠟燭圖字典列表。

        Returns:
            np.ndarray: 預處理和標準化後的特徵陣列，形狀為 (timesteps, features)。
        """
        if not raw_candles:
            logger.warning("收到的蠟燭圖數據為空，無法進行預處理。")
            return np.array([])

        # 0. 數據新鮮度檢查
        last_candle_time = pd.to_datetime(raw_candles[-1]['time'])
        if not self._is_data_fresh(last_candle_time):
            return np.array([])

        # 1. 將原始數據轉換為 DataFrame
        df = pd.DataFrame(raw_candles)
        df['time'] = pd.to_datetime(df['time'])
        
        # 處理價格數據，通常 Oanda 返回的 mid, bid, ask 是字典
        price_types = ['mid', 'bid', 'ask']
        for p_type in price_types:
            if p_type in df.columns and isinstance(df[p_type].iloc[0], dict):
                price_df = df[p_type].apply(pd.Series).astype(float)
                price_df.columns = [f"{p_type}_{col}" for col in price_df.columns]
                df = pd.concat([df, price_df], axis=1)
        df = df.drop(columns=price_types, errors='ignore')
        df['volume'] = df['volume'].astype(float)

        # 2. 計算特徵 (與訓練時保持一致)
        processed_df = pd.DataFrame(index=df.index)
        features_to_standardize = []

        # 計算價格回報率
        for col in df.columns:
            if "_o" in col or "_h" in col or "_l" in col or "_c" in col:
                ret_col_name = f"{col}_log_ret"
                processed_df[ret_col_name] = self._calculate_log_returns(df[col])
                features_to_standardize.append(ret_col_name)

        # 處理成交量
        if 'volume' in df.columns:
            vol_col_name = "volume_log"
            processed_df[vol_col_name] = np.log1p(df['volume'])
            features_to_standardize.append(vol_col_name)

        # 3. 使用載入的 Scaler 進行標準化
        for feature in features_to_standardize:
            if feature in self.scalers:
                mean = self.scalers[feature]['mean']
                scale = self.scalers[feature]['scale']
                if scale == 0: # 避免除以零
                    scale = 1e-9
                processed_df[feature] = (processed_df[feature] - mean) / scale
                # 裁剪極端值
                processed_df[feature] = np.clip(processed_df[feature], -5.0, 5.0)
            else:
                logger.warning(f"特徵 '{feature}' 在 Scaler 檔案中未找到，將不會被標準化。")
                # 如果某个特徵在 scaler 中不存在，可以選擇填充0或從特徵列表中移除
                processed_df[feature] = 0 

        # 確保特徵順序與訓練時一致
        final_features = [f for f in self.scalers.keys() if f in processed_df.columns]
        processed_df = processed_df[final_features]
        
        # 刪除因 shift(1) 操作產生的第一行 NaN
        processed_df = processed_df.iloc[1:]

        logger.info(f"數據預處理完成。最終特徵形狀: {processed_data.shape}")
        return processed_df.values

# --- 範例 ---
if __name__ == '__main__':
    # 假設我們有一個從訓練中保存的 scaler 檔案
    dummy_scalers = {
        "mid_c_log_ret": {"mean": 1.2e-05, "scale": 0.00015},
        "volume_log": {"mean": 8.5, "scale": 1.2}
    }
    scaler_file = "dummy_scalers.json"
    with open(scaler_file, 'w') as f:
        json.dump(dummy_scalers, f)

    # 創建一個預處理器實例
    config = {"data_freshness_threshold_minutes": 10}
    preprocessor = LivePreprocessor(scaler_file, config)

    # 模擬從 Oanda API 收到的數據
    mock_api_data = [
        {'complete': True, 'volume': 1500, 'time': '2025-06-24T10:00:00.000000000Z', 'mid': {'o': '1.07000', 'h': '1.07010', 'l': '1.06990', 'c': '1.07005'}},
        {'complete': True, 'volume': 1800, 'time': '2025-06-24T10:05:00.000000000Z', 'mid': {'o': '1.07005', 'h': '1.07025', 'l': '1.07000', 'c': '1.07020'}},
        {'complete': True, 'volume': 1650, 'time': '2025-06-24T10:10:00.000000000Z', 'mid': {'o': '1.07020', 'h': '1.07030', 'l': '1.07015', 'c': '1.07018'}},
    ]

    # 進行轉換
    processed_data = preprocessor.transform(mock_api_data)

    logger.info("轉換後的數據 (Numpy Array):")
    logger.info(processed_data)
    logger.info(f"形狀: {processed_data.shape}")

    # 清理 dummy 檔案
    import os
    os.remove(scaler_file)
