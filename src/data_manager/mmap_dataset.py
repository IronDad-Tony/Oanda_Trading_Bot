# src/data_manager/mmap_dataset.py
"""
MemoryMappedDataset module
Efficiently handles large-scale time series data for model training.
It stores the preprocessed feature data into memory-mapped files.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import shutil
import hashlib
import json
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta, timezone
import sys # Ensure sys is imported at the top level for fallback and __main__ usage
import logging
import atexit
import glob
import os

# Flag to prevent duplicate import logging
_import_logged = False

try:
    from common.config import (
        MMAP_DATA_DIR, TIMESTEPS, PRICE_COLUMNS, GRANULARITY,
        get_granularity_seconds, OANDA_API_KEY
    )
    from common.logger_setup import logger
    from data_manager.database_manager import query_historical_data
    from feature_engineer.preprocessor import preprocess_data_for_model
    from data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols
except ImportError:
    # project_root = Path(__file__).resolve().parent.parent.parent # 移除
    # src_path = project_root / "src" # 移除
    # if str(project_root) not in sys.path: # 移除
    #     sys.path.insert(0, str(project_root)) # 移除
    try:
        # 假設 PYTHONPATH 已設定，這些導入應該能工作
        from src.common.config import (
            MMAP_DATA_DIR, TIMESTEPS, PRICE_COLUMNS, GRANULARITY,
            get_granularity_seconds, OANDA_API_KEY
        )
        from src.common.logger_setup import logger
        from src.data_manager.database_manager import query_historical_data
        from src.feature_engineer.preprocessor import preprocess_data_for_model
        from src.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols
        if not _import_logged:
            logger.info("Direct run MMapDataset: Successfully re-imported common modules.")
            _import_logged = True
    except ImportError as e_retry:
        import logging
        logger = logging.getLogger("mmap_dataset_fallback") # type: ignore
        logger.error(f"Direct run MMapDataset: Critical import error: {e_retry}", exc_info=True)
        MMAP_DATA_DIR = Path("./mmap_data_fallback")
        TIMESTEPS = 128
        PRICE_COLUMNS = ['bid_close', 'ask_close']
        GRANULARITY = "S5"
        OANDA_API_KEY = None
        def get_granularity_seconds(g): return 5
        def query_historical_data(*args, **kwargs): logger.error("DB query not available."); return pd.DataFrame()
        def preprocess_data_for_model(*args, **kwargs): logger.error("Preprocessor not available."); return {}, None
        def format_datetime_for_oanda(dt): return dt.isoformat()
        def manage_data_download_for_symbols(*args, **kwargs): logger.error("Downloader not available.")


# 全域變數來追蹤活躍的數據集實例
_active_datasets = []

def cleanup_mmap_temp_files():
    """
    清理 mmap 暫存檔案
    
    這個函數會：
    1. 清理所有活躍數據集的 mmap 檔案
    2. 清理孤立的 mmap 暫存檔案
    """
    current_logger = None
    logger_available = False
    if 'logger' in globals() and globals()['logger'] is not None:
        _logger_candidate = globals()['logger']
        if hasattr(_logger_candidate, 'handlers') and _logger_candidate.handlers and \
           hasattr(_logger_candidate, 'isEnabledFor') and _logger_candidate.isEnabledFor(logging.INFO) and \
           not getattr(_logger_candidate, 'disabled', False):
            for handler in _logger_candidate.handlers:
                if hasattr(handler, 'stream') and handler.stream and not handler.stream.closed:
                    logger_available = True
                    break
            if logger_available:
                current_logger = _logger_candidate

    original_handle_errors = {}
    if current_logger: # current_logger is the one determined above
        for handler in current_logger.handlers:
            if hasattr(handler, 'handleError'):
                original_handle_errors[handler] = handler.handleError
                handler.handleError = lambda record: None # Suppress default error handling

    # Define safer logging functions for use within this atexit handler
    def make_safe_logger_for_cleanup(level_str, fallback_prefix):
        def safe_log_action(msg):
            if logger_available and current_logger: # logger_available based on initial check
                try:
                    log_method = getattr(current_logger, level_str)
                    log_method(msg)
                except Exception as e_log: # Catch any exception during the logging attempt
                    # Since default handleError is suppressed, we print our own fallback.
                    print(f"{fallback_prefix} [CLEANUP_LOG_ATTEMPT_FAILED - {type(e_log).__name__}]: {msg}", file=sys.stderr)
            else:
                # Logger was not available from the start or current_logger is None
                print(f"{fallback_prefix} [CLEANUP_LOG_UNAVAILABLE_FALLBACK]: {msg}", file=sys.stderr)
        return safe_log_action

    log_func = make_safe_logger_for_cleanup('info', 'INFO')
    warn_func = make_safe_logger_for_cleanup('warning', 'WARNING')
    error_func = make_safe_logger_for_cleanup('error', 'ERROR')
    debug_func = make_safe_logger_for_cleanup('debug', 'DEBUG')

    try:
        log_func("Attempting to cleanup mmap temp files via atexit or direct call.")
        # 關閉所有活躍的數據集
        for dataset in _active_datasets[:]:  # 使用副本避免修改時的問題
            try:
                if hasattr(dataset, 'close'):
                    dataset.close()
                    log_func(f"Closed dataset: {getattr(dataset, 'dataset_id', 'unknown')}")
            except Exception as e:
                warn_func(f"Error closing dataset: {e}")
        
        _active_datasets.clear()
        
        # 清理孤立的 mmap 檔案
        try:
            mmap_base_dir = MMAP_DATA_DIR
            if mmap_base_dir.exists():
                # 查找所有 .mmap 檔案
                mmap_files = list(mmap_base_dir.rglob("*.mmap"))
                if mmap_files:
                    log_func(f"Found {len(mmap_files)} mmap temp files, cleaning up...")
                    for mmap_file in mmap_files:
                        try:
                            mmap_file.unlink()
                            debug_func(f"Deleted mmap file: {mmap_file}")
                        except Exception as e:
                            warn_func(f"Cannot delete mmap file {mmap_file}: {e}")
                    
                    # 清理空的目錄
                    for dataset_dir in mmap_base_dir.iterdir():
                        if dataset_dir.is_dir():
                            try:
                                # 如果目錄為空，則刪除
                                if not any(dataset_dir.iterdir()):
                                    dataset_dir.rmdir()
                                    debug_func(f"Deleted empty directory: {dataset_dir}")
                            except Exception as e:
                                # Changed from debug_func to warn_func as failure to delete a dir might be notable
                                warn_func(f"Cannot delete directory {dataset_dir}: {e}")
                else:
                    log_func("No mmap temp files found for cleanup")
        except Exception as e:
            warn_func(f"Error cleaning mmap temp files: {e}")
            
    except Exception as e:
        # Use the safer error_func for the outermost catch block
        error_func(f"Severe error during mmap temp files cleanup: {e}")
    finally:
        # Restore original handleError methods
        if current_logger:
            for handler in current_logger.handlers:
                if handler in original_handle_errors:
                    handler.handleError = original_handle_errors[handler]

def cleanup_old_mmap_files(max_age_hours: int = 24):
    """
    清理超過指定時間的舊 mmap 檔案
    
    Args:
        max_age_hours: 檔案最大保留時間（小時）
    """
    try:
        mmap_base_dir = MMAP_DATA_DIR
        if not mmap_base_dir.exists():
            return
            
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=max_age_hours)
        
        cleaned_count = 0
        for dataset_dir in mmap_base_dir.iterdir():
            if dataset_dir.is_dir():
                try:
                    # 檢查目錄的修改時間
                    dir_mtime = datetime.fromtimestamp(dataset_dir.stat().st_mtime)
                    if dir_mtime < cutoff_time:
                        # 刪除整個數據集目錄
                        shutil.rmtree(dataset_dir)
                        cleaned_count += 1
                        logger.info(f"Cleaned up old dataset directory: {dataset_dir}")
                except Exception as e:
                    logger.warning(f"Error cleaning up old directory {dataset_dir}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old dataset directories")
        else:
            logger.debug("No old dataset directories found for cleanup")
            
    except Exception as e:
        logger.warning(f"Error cleaning up old mmap files: {e}")

# 註冊程式退出時的清理函數
atexit.register(cleanup_mmap_temp_files)
class UniversalMemoryMappedDataset(Dataset):
    def __init__(self,
                 symbols: List[str],
                 start_time_iso: str,
                 end_time_iso: str,
                 granularity: str = GRANULARITY,
                 timesteps_history: int = TIMESTEPS,
                 force_reload: bool = False,
                 mmap_mode: str = 'r'
                ):
        self.symbols = sorted(list(set(symbols)))
        self.start_time_iso = start_time_iso
        self.end_time_iso = end_time_iso
        self.granularity = granularity
        self.timesteps_history = timesteps_history
        self.mmap_mode = mmap_mode
        dataset_signature_str = f"{'_'.join(self.symbols)}_{self.start_time_iso}_{self.end_time_iso}_{self.granularity}"
        self.dataset_id = hashlib.md5(dataset_signature_str.encode()).hexdigest()[:16]
        self.dataset_mmap_dir = MMAP_DATA_DIR / self.dataset_id
        logger.info(f"UniversalMemoryMappedDataset initialized: ID={self.dataset_id}, Symbols={self.symbols}")
        logger.info(f"Time range: {self.start_time_iso} to {self.end_time_iso}, Granularity: {self.granularity}")
        logger.info(f"MMap data directory: {self.dataset_mmap_dir}")
        self.processed_features_memmaps: Dict[str, Optional[np.memmap]] = {sym: None for sym in self.symbols}
        self.raw_prices_memmaps: Dict[str, Optional[np.memmap]] = {sym: None for sym in self.symbols}
        self.aligned_timestamps: Optional[pd.DatetimeIndex] = None
        self.num_features_per_symbol: int = -1
        self.total_aligned_steps: int = 0
        self.scalers_map: Optional[Dict[str, Dict[str, StandardScaler]]] = None
        self.feature_columns_ordered_for_metadata: List[str] = [] # 用於存储特徵列順序
        self.raw_price_columns_ordered: List[str] = []

        if force_reload and self.dataset_mmap_dir.exists():
            logger.info(f"Force reload: 正在刪除已存在的mmap目錄: {self.dataset_mmap_dir}")
            try: shutil.rmtree(self.dataset_mmap_dir)
            except OSError as e: logger.error(f"刪除mmap目錄 {self.dataset_mmap_dir} 失敗: {e}", exc_info=True)
        self.dataset_mmap_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file_path = self.dataset_mmap_dir / "dataset_metadata.json"
        # 确保所有必要的货币对数据已下载
        try:
            from src.common.config import ACCOUNT_CURRENCY
            from src.data_manager.currency_manager import ensure_currency_data_for_trading
            
            success, all_symbols = ensure_currency_data_for_trading(
                trading_symbols=self.symbols,
                account_currency=ACCOUNT_CURRENCY,  # 使用配置中的账户货币
                start_time_iso=self.start_time_iso,
                end_time_iso=self.end_time_iso,
                granularity=self.granularity
            )
            if success:
                # 使用扩展后的symbols列表（包含汇率转换货币对）
                self.symbols = sorted(list(set(all_symbols)))
                logger.info(f"数据集使用扩展后的symbols列表: {self.symbols}")
            else:
                logger.warning("确保货币数据失败，使用原始symbols列表")
        except ImportError as e:
            logger.warning(f"无法导入currency_manager或config: {e}, 使用原始symbols列表")
        
        if not force_reload and self.metadata_file_path.exists():
            self._load_from_existing_mmap()
        else:
            self._load_and_preprocess_data()
        if self.total_aligned_steps <= 0 or self.num_features_per_symbol <=0 :
            logger.error("數據集初始化失敗或為空，請檢查數據源和時間範圍。")
            raise ValueError("數據集初始化失敗或為空。")
        logger.info(f"數據集初始化完成。總對齊時間步: {self.total_aligned_steps}, "
                    f"每個Symbol的特徵數量: {self.num_features_per_symbol}, "
                    f"樣本歷史步長: {self.timesteps_history}")
# 註冊到活躍數據集列表，用於程式退出時清理
        _active_datasets.append(self)
        
        # 在初始化時清理舊的 mmap 檔案（超過24小時的）
        cleanup_old_mmap_files(max_age_hours=24)

    def _load_and_preprocess_data(self):
        logger.info("開始從數據庫加載和預處理數據...")
        raw_data_map: Dict[str, pd.DataFrame] = {}
        min_len = float('inf'); max_len = 0
        for symbol in self.symbols:
            logger.debug(f"查詢 symbol: {symbol} 的原始數據...")
            df_raw = query_historical_data(symbol, self.granularity, self.start_time_iso, self.end_time_iso)
            if df_raw.empty:
                logger.warning(f"Symbol {symbol} 在指定時間範圍內沒有數據。")
                raw_data_map[symbol] = pd.DataFrame(columns=['time'] + PRICE_COLUMNS); raw_data_map[symbol]['time'] = pd.to_datetime(raw_data_map[symbol]['time']); min_len = 0
            else:
                raw_data_map[symbol] = df_raw; min_len = min(min_len, len(df_raw)); max_len = max(max_len, len(df_raw))
        if min_len == 0 and max_len == 0 : logger.error("所有請求的symbols在指定範圍內均無數據。無法創建數據集。"); self.total_aligned_steps = 0; return
        logger.info("開始對齊各symbol的時間序列...")
        try:
            start_dt = pd.to_datetime(self.start_time_iso, utc=True); end_dt = pd.to_datetime(self.end_time_iso, utc=True)
        except Exception as e: logger.error(f"解析時間字符串失敗: {self.start_time_iso}, {self.end_time_iso}. Error: {e}", exc_info=True); return
        granularity_freq_str = f"{get_granularity_seconds(self.granularity)}s" # <--- 'S' 改為 's'
        self.aligned_timestamps = pd.date_range(start=start_dt, end=end_dt, freq=granularity_freq_str, name='time')
        if self.aligned_timestamps.empty: logger.error("生成的對齊時間戳為空，請檢查時間範圍和粒度。"); self.total_aligned_steps = 0; return
        self.total_aligned_steps = len(self.aligned_timestamps)
        logger.info(f"生成對齊時間戳 {self.total_aligned_steps} 條，頻率: {granularity_freq_str}")
        aligned_raw_data_map: Dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            df_sym_raw = raw_data_map[symbol]
            if not df_sym_raw.empty and 'time' in df_sym_raw.columns:
                df_sym_raw = df_sym_raw.set_index('time'); df_aligned = df_sym_raw.reindex(self.aligned_timestamps)
                df_aligned = df_aligned.ffill().bfill()
                for col in PRICE_COLUMNS:
                    if col in df_aligned.columns and pd.api.types.is_numeric_dtype(df_aligned[col]): df_aligned[col] = df_aligned[col].fillna(0)
                df_aligned = df_aligned.reset_index()
            else:
                df_aligned = pd.DataFrame(index=self.aligned_timestamps).reset_index()
                for col in PRICE_COLUMNS:
                    if col != 'time': df_aligned[col] = 0.0
            aligned_raw_data_map[symbol] = df_aligned; logger.debug(f"Symbol {symbol} 對齊後數據行數: {len(df_aligned)}")
        logger.info("開始預處理對齊後的數據...")
        processed_features_map, self.scalers_map = preprocess_data_for_model(aligned_raw_data_map, fit_scalers=True)
        for symbol in self.symbols:
            if symbol in processed_features_map and not processed_features_map[symbol].empty:
                self.feature_columns_ordered_for_metadata = sorted([col for col in processed_features_map[symbol].columns if col != 'time']) # 存儲列順序
                self.num_features_per_symbol = len(self.feature_columns_ordered_for_metadata)
                break
        if self.num_features_per_symbol == -1: logger.error("未能確定預處理後的特徵數量。"); return
        logger.info("開始創建和寫入內存映射文件...")
        self.raw_price_columns_ordered = ['bid_close', 'ask_close', 'bid_high', 'bid_low', 'ask_high', 'ask_low']
        for symbol in self.symbols:
            df_features = processed_features_map.get(symbol)
            if df_features is None or df_features.empty:
                logger.warning(f"Symbol {symbol} 沒有預處理後的特徵數據，將創建全零的mmap文件。")
                features_array = np.zeros((self.total_aligned_steps, self.num_features_per_symbol), dtype=np.float32)
            else:
                if len(self.feature_columns_ordered_for_metadata) != self.num_features_per_symbol: # 應該不會發生
                    logger.error(f"Symbol {symbol} 的特徵數量 ({len(self.feature_columns_ordered_for_metadata)}) 與預期 ({self.num_features_per_symbol}) 不符。")
                    features_array = np.zeros((self.total_aligned_steps, self.num_features_per_symbol), dtype=np.float32)
                else:
                    features_array = df_features[self.feature_columns_ordered_for_metadata].values.astype(np.float32)
            mmap_features_path = self.dataset_mmap_dir / f"{symbol}_features.mmap"
            self.processed_features_memmaps[symbol] = np.memmap(str(mmap_features_path), dtype=np.float32, mode='w+', shape=(self.total_aligned_steps, self.num_features_per_symbol))
            self.processed_features_memmaps[symbol][:] = features_array[:]; self.processed_features_memmaps[symbol].flush()
            logger.debug(f"Symbol {symbol} 的特徵mmap文件已創建: {mmap_features_path}")
            df_raw_aligned = aligned_raw_data_map.get(symbol)
            raw_price_cols_to_store = self.raw_price_columns_ordered
            if df_raw_aligned is None or df_raw_aligned.empty or not all(col in df_raw_aligned.columns for col in raw_price_cols_to_store):
                logger.warning(f"Symbol {symbol} 沒有對齊後的原始價格數據，將創建全零的價格mmap文件。")
                prices_array = np.zeros((self.total_aligned_steps, len(raw_price_cols_to_store)), dtype=np.float32)
            else: prices_array = df_raw_aligned[raw_price_cols_to_store].values.astype(np.float32)
            mmap_prices_path = self.dataset_mmap_dir / f"{symbol}_raw_prices.mmap"
            self.raw_prices_memmaps[symbol] = np.memmap(str(mmap_prices_path), dtype=np.float32, mode='w+', shape=(self.total_aligned_steps, len(raw_price_cols_to_store)))
            self.raw_prices_memmaps[symbol][:] = prices_array[:]; self.raw_prices_memmaps[symbol].flush()
            logger.debug(f"Symbol {symbol} 的原始價格mmap文件已創建: {mmap_prices_path}")
        self._save_metadata()
        logger.info("數據加載、預處理和mmap文件創建完成。")

    def _save_metadata(self):
        metadata = {
            "dataset_id": self.dataset_id, "symbols": self.symbols,
            "start_time_iso": self.start_time_iso, "end_time_iso": self.end_time_iso,
            "granularity": self.granularity, "timesteps_history": self.timesteps_history,
            "total_aligned_steps": self.total_aligned_steps,
            "num_features_per_symbol": self.num_features_per_symbol,
            "feature_columns_ordered": self.feature_columns_ordered_for_metadata, # 使用存儲的列順序
            "raw_price_columns_ordered": self.raw_price_columns_ordered
        }
        try:
            with open(self.metadata_file_path, 'w') as f: json.dump(metadata, f, indent=4)
            logger.info(f"數據集元數據已保存到: {self.metadata_file_path}")
        except Exception as e: logger.error(f"保存元數據失敗: {e}", exc_info=True)

    def _load_from_existing_mmap(self):
        logger.info(f"嘗試從現有mmap文件加載數據集: {self.dataset_mmap_dir}")
        try:
            with open(self.metadata_file_path, 'r') as f: metadata = json.load(f)
            if sorted(metadata.get("symbols", [])) != self.symbols or \
               metadata.get("start_time_iso") != self.start_time_iso or \
               metadata.get("end_time_iso") != self.end_time_iso or \
               metadata.get("granularity") != self.granularity:
                logger.warning("現有mmap元數據與當前參數不完全匹配，將強制重新加載。")
                self._load_and_preprocess_data(); return
            self.total_aligned_steps = metadata["total_aligned_steps"]
            self.num_features_per_symbol = metadata["num_features_per_symbol"]
            # 確保 timesteps_history 與配置一致，避免維度不匹配
            metadata_timesteps = metadata.get("timesteps_history", self.timesteps_history)
            if metadata_timesteps != self.timesteps_history:
                logger.warning(f"元數據中的 timesteps_history ({metadata_timesteps}) 與當前配置 ({self.timesteps_history}) 不一致！")
                logger.warning(f"將使用當前配置值 {self.timesteps_history} 並強制重新加載數據。")
                self._load_and_preprocess_data(); return
            self.timesteps_history = metadata_timesteps
            self.feature_columns_ordered_for_metadata = metadata.get("feature_columns_ordered", []) # 加載列順序
            self.raw_price_columns_ordered = metadata.get("raw_price_columns_ordered", [])
            if not self.raw_price_columns_ordered:
                logger.warning("`raw_price_columns_ordered` not found in metadata or is empty. Forcing reload.")
                self._load_and_preprocess_data()
                return
            start_dt = pd.to_datetime(self.start_time_iso, utc=True); end_dt = pd.to_datetime(self.end_time_iso, utc=True)
            granularity_freq_str = f"{get_granularity_seconds(self.granularity)}s" # <--- 'S' 改為 's'
            self.aligned_timestamps = pd.date_range(start=start_dt, end=end_dt, freq=granularity_freq_str, name='time')
            for symbol in self.symbols:
                mmap_features_path = self.dataset_mmap_dir / f"{symbol}_features.mmap"
                if mmap_features_path.exists():
                    self.processed_features_memmaps[symbol] = np.memmap(str(mmap_features_path), dtype=np.float32, mode=self.mmap_mode, shape=(self.total_aligned_steps, self.num_features_per_symbol))
                else: raise FileNotFoundError(f"特徵mmap文件未找到: {mmap_features_path}")
                mmap_prices_path = self.dataset_mmap_dir / f"{symbol}_raw_prices.mmap"
                raw_price_cols_count = len(self.raw_price_columns_ordered)
                if mmap_prices_path.exists():
                    self.raw_prices_memmaps[symbol] = np.memmap(str(mmap_prices_path), dtype=np.float32, mode=self.mmap_mode, shape=(self.total_aligned_steps, raw_price_cols_count))
                else: raise FileNotFoundError(f"原始價格mmap文件未找到: {mmap_prices_path}")
            logger.info("成功從現有的mmap文件和元數據加載。")
        except FileNotFoundError:
            logger.warning(f"元數據文件 {self.metadata_file_path} 或部分mmap文件未找到，強制重新加載數據。")
            self._load_and_preprocess_data()
        except Exception as e:
            logger.error(f"從現有mmap加載數據時發生錯誤: {e}。強制重新加載數據。", exc_info=True)
            self._load_and_preprocess_data()

    def __len__(self) -> int:
        if self.total_aligned_steps < self.timesteps_history: return 0
        return self.total_aligned_steps - self.timesteps_history + 1

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0 or index >= len(self): raise IndexError(f"索引 {index} 超出範圍 [0, {len(self)-1}]")
        start_slice = index; end_slice = index + self.timesteps_history
        all_sym_features_list = [] ; all_sym_raw_prices_list = []
        for symbol in self.symbols:
            if self.processed_features_memmaps[symbol] is not None:
                all_sym_features_list.append(self.processed_features_memmaps[symbol][start_slice:end_slice])
            else: all_sym_features_list.append(np.zeros((self.timesteps_history, self.num_features_per_symbol), dtype=np.float32))
            if self.raw_prices_memmaps[symbol] is not None:
                all_sym_raw_prices_list.append(self.raw_prices_memmaps[symbol][start_slice:end_slice])
            else: all_sym_raw_prices_list.append(np.zeros((self.timesteps_history, 2), dtype=np.float32))
        features_tensor = torch.from_numpy(np.stack(all_sym_features_list, axis=0).astype(np.float32))
        raw_prices_tensor = torch.from_numpy(np.stack(all_sym_raw_prices_list, axis=0).astype(np.float32))
        return {"features": features_tensor, "raw_prices": raw_prices_tensor}

    def close(self):
        """關閉所有內存映射文件。重要：在程序退出前調用."""
        # 更健壯地檢查 logger 是否仍然可用
        logger_available = False
        current_logger_instance = None

        if 'logger' in globals() and globals()['logger'] is not None:
            _logger_candidate = globals()['logger']
            if hasattr(_logger_candidate, 'handlers') and _logger_candidate.handlers and \
               hasattr(_logger_candidate, 'isEnabledFor') and _logger_candidate.isEnabledFor(logging.INFO) and \
               not getattr(_logger_candidate, 'disabled', False):
                for handler in _logger_candidate.handlers:
                    if hasattr(handler, 'stream') and handler.stream and not handler.stream.closed:
                        logger_available = True
                        break
                if logger_available:
                    current_logger_instance = _logger_candidate
        
        log_func = current_logger_instance.info if logger_available else lambda msg: print(f"INFO (UniversalMemoryMappedDataset.close fallback): {msg}", file=sys.stderr)
        
        dataset_id_to_log = getattr(self, 'dataset_id', 'UnknownDatasetID_in_close')
        log_func(f"Closing memory-mapped files for dataset {dataset_id_to_log}...") # Translated to English

        symbols_to_iterate = getattr(self, 'symbols', [])
        processed_features_memmaps_dict = getattr(self, 'processed_features_memmaps', {})
        raw_prices_memmaps_dict = getattr(self, 'raw_prices_memmaps', {})

        for symbol_item in symbols_to_iterate:
            if processed_features_memmaps_dict.get(symbol_item) is not None:
                # Ensure the memmap object itself is deleted to close the file
                try:
                    del processed_features_memmaps_dict[symbol_item] # This should trigger __del__ on memmap if ref count is 0
                except Exception as e:
                    print(f"FALLBACK_CLOSE_ERROR: Error deleting processed_features_memmap for {symbol_item}: {e}")
                self.processed_features_memmaps[symbol_item] = None # Explicitly set to None

            if raw_prices_memmaps_dict.get(symbol_item) is not None:
                try:
                    del raw_prices_memmaps_dict[symbol_item] # This should trigger __del__ on memmap if ref count is 0
                except Exception as e:
                    print(f"FALLBACK_CLOSE_ERROR: Error deleting raw_prices_memmap for {symbol_item}: {e}")
                self.raw_prices_memmaps[symbol_item] = None # Explicitly set to None
        
        # import gc # Generally not needed here, __del__ of memmap objects handles file closing.
        # gc.collect()
        
        log_func(f"Memory-mapped files for dataset {dataset_id_to_log} have been closed.") # Translated to English
        
        # 從活躍數據集列表中移除
        try:
            if self in _active_datasets:
                _active_datasets.remove(self)
        except (ValueError, NameError):
            pass  # 如果已經不在列表中或列表不存在，忽略錯誤

    def __del__(self):
        # __del__ 應該盡可能簡單，避免複雜操作和依賴外部狀態
        # 主要的清理工作應該通過顯式調用 close() 完成
        # 這裡調用 close() 是一個後備措施
        try:
            # 檢查必要的屬性是否存在，防止在不完整初始化時出錯
            if hasattr(self, 'processed_features_memmaps') and hasattr(self, 'raw_prices_memmaps'):
                 self.close()
        except Exception:
            # 在 __del__ 中最好不要拋出異常，也不要依賴 logger
            # print("Warning: Error during UniversalMemoryMappedDataset.__del__.", file=sys.stderr)
            pass # 靜默失敗

if __name__ == "__main__":
    logger.info("正在直接運行 UniversalMemoryMappedDataset.py 進行測試...")
    if OANDA_API_KEY is None:
        logger.error("OANDA_API_KEY 未配置，無法執行完整的數據集創建測試。")
        sys.exit(1)
    logger.info("MMap數據集測試開始。")
    test_symbols_list = ["EUR_USD", "USD_JPY"]
    try:
        test_start_datetime = datetime(2024, 5, 22, 10, 0, 0, tzinfo=timezone.utc)
        test_end_datetime = datetime(2024, 5, 22, 11, 0, 0, tzinfo=timezone.utc)
    except ValueError as e_date: logger.error(f"測試用的固定日期時間無效: {e_date}", exc_info=True); sys.exit(1)
    if test_start_datetime >= test_end_datetime: logger.error("測試時間範圍無效：開始時間必須早於結束時間。"); sys.exit(1)
    test_start_iso_str = format_datetime_for_oanda(test_start_datetime)
    test_end_iso_str = format_datetime_for_oanda(test_end_datetime)
    test_granularity_val = "S5"; test_timesteps_history_val = 64
    logger.info(f"測試參數: symbols={test_symbols_list}, start={test_start_iso_str}, end={test_end_iso_str}, granularity={test_granularity_val}, history_len={test_timesteps_history_val}")
    dataset_train = None; dataset_test_load = None; dataset_for_loader = None
    try:
        try: # 確保 manage_data_download_for_symbols 在此作用域可見
            from src.data_manager.oanda_downloader import manage_data_download_for_symbols
        except ImportError: logger.critical("在 __main__ 塊中無法導入 manage_data_download_for_symbols。"); sys.exit(1)
        logger.info("確保數據庫中有測試時間段的數據 (如果沒有則下載)...")
        manage_data_download_for_symbols(symbols=test_symbols_list, overall_start_str=test_start_iso_str, overall_end_str=test_end_iso_str, granularity=test_granularity_val)
        logger.info("數據庫數據準備完成/已檢查。")
        logger.info("--- 第一次創建/加載數據集 (force_reload=True) ---")
        dataset_train = UniversalMemoryMappedDataset(symbols=test_symbols_list, start_time_iso=test_start_iso_str, end_time_iso=test_end_iso_str, granularity=test_granularity_val, timesteps_history=test_timesteps_history_val, force_reload=True)
        logger.info(f"訓練數據集長度: {len(dataset_train)}")
        sample_train_features_first = None # 用於比較
        if len(dataset_train) > 0:
            logger.info("從訓練數據集獲取第一個樣本..."); sample_train = dataset_train[0]
            logger.info(f"  樣本 'features' shape: {sample_train['features'].shape}"); logger.info(f"  樣本 'raw_prices' shape: {sample_train['raw_prices'].shape}")
            assert sample_train['features'].shape[0] == len(test_symbols_list); assert sample_train['features'].shape[1] == test_timesteps_history_val
            assert sample_train['raw_prices'].shape[0] == len(test_symbols_list); assert sample_train['raw_prices'].shape[1] == test_timesteps_history_val; assert sample_train['raw_prices'].shape[2] == 2
            sample_train_features_first = sample_train['features'].clone() # 複製用於比較
            logger.info("從訓練數據集獲取最後一個樣本..."); last_sample_train = dataset_train[len(dataset_train)-1]
            logger.info(f"  最後樣本 'features' shape: {last_sample_train['features'].shape}")
        logger.info("\n--- 第二次加載數據集 (force_reload=False, 應從mmap加載) ---")
        dataset_test_load = UniversalMemoryMappedDataset(symbols=test_symbols_list, start_time_iso=test_start_iso_str, end_time_iso=test_end_iso_str, granularity=test_granularity_val, timesteps_history=test_timesteps_history_val, force_reload=False)
        logger.info(f"測試加載數據集長度: {len(dataset_test_load)}")
        if len(dataset_test_load) > 0:
            sample_test_load = dataset_test_load[0]
            logger.info(f"  測試加載樣本 'features' shape: {sample_test_load['features'].shape}")
            if sample_train_features_first is not None and torch.allclose(sample_train_features_first, sample_test_load['features']):
                logger.info("第二次加載的樣本數據與第一次一致 (基本驗證)。")
            else: logger.warning("第二次加載的樣本數據與第一次不一致，或第一次無樣本。")
        logger.info("\n--- 測試 DataLoader ---")
        if dataset_test_load is not None and len(dataset_test_load) > 0:
            dataset_for_loader = UniversalMemoryMappedDataset(symbols=test_symbols_list,start_time_iso=test_start_iso_str,end_time_iso=test_end_iso_str, granularity=test_granularity_val,timesteps_history=test_timesteps_history_val,force_reload=False)
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset_for_loader, batch_size=4, shuffle=True, num_workers=0)
            batch_count = 0
            for i, batch in enumerate(dataloader):
                logger.info(f"  Batch {i+1}: features shape: {batch['features'].shape}, raw_prices shape: {batch['raw_prices'].shape}")
                assert batch['features'].shape[0] <= 4; assert batch['features'].shape[1] == len(test_symbols_list); assert batch['features'].shape[2] == test_timesteps_history_val
                batch_count += 1
                if batch_count >= 2: break
            logger.info(f"DataLoader 測試完成，遍歷了 {batch_count} 個批次。")
        else: logger.info("數據集為空或未成功加載，跳過 DataLoader 測試。")
    except Exception as e: logger.error(f"UniversalMemoryMappedDataset 測試過程中發生嚴重錯誤: {e}", exc_info=True)
    finally:
        if dataset_train: dataset_train.close()
        if dataset_test_load: dataset_test_load.close()
        if dataset_for_loader: dataset_for_loader.close()
    logger.info("UniversalMemoryMappedDataset.py 測試執行完畢。")