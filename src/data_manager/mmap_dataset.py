# src/data_manager/mmap_dataset.py
"""
MemoryMappedDataset 模組
高效處理大規模時間序列數據，用於模型訓練。
它會將預處理後的特徵數據存儲到內存映射文件中。
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
import sys # 確保 sys 在頂層導入，以便 fallback 和 __main__ 使用
import logging

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
    project_root = Path(__file__).resolve().parent.parent.parent
    src_path = project_root / "src"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from src.common.config import (
            MMAP_DATA_DIR, TIMESTEPS, PRICE_COLUMNS, GRANULARITY,
            get_granularity_seconds, OANDA_API_KEY
        )
        from src.common.logger_setup import logger
        from src.data_manager.database_manager import query_historical_data
        from src.feature_engineer.preprocessor import preprocess_data_for_model
        from src.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols
        logger.info("Direct run MMapDataset: Successfully re-imported common modules.")
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
        logger.info(f"初始化UniversalMemoryMappedDataset: ID={self.dataset_id}, Symbols={self.symbols}")
        logger.info(f"時間範圍: {self.start_time_iso} to {self.end_time_iso}, Granularity: {self.granularity}")
        logger.info(f"MMap數據目錄: {self.dataset_mmap_dir}")
        self.processed_features_memmaps: Dict[str, Optional[np.memmap]] = {sym: None for sym in self.symbols}
        self.raw_prices_memmaps: Dict[str, Optional[np.memmap]] = {sym: None for sym in self.symbols}
        self.aligned_timestamps: Optional[pd.DatetimeIndex] = None
        self.num_features_per_symbol: int = -1
        self.total_aligned_steps: int = 0
        self.scalers_map: Optional[Dict[str, Dict[str, StandardScaler]]] = None
        self.feature_columns_ordered_for_metadata: List[str] = [] # 用於存儲特徵列順序

        if force_reload and self.dataset_mmap_dir.exists():
            logger.info(f"Force reload: 正在刪除已存在的mmap目錄: {self.dataset_mmap_dir}")
            try: shutil.rmtree(self.dataset_mmap_dir)
            except OSError as e: logger.error(f"刪除mmap目錄 {self.dataset_mmap_dir} 失敗: {e}", exc_info=True)
        self.dataset_mmap_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file_path = self.dataset_mmap_dir / "dataset_metadata.json"
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
            raw_price_cols_to_store = ['bid_close', 'ask_close']
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
            "raw_price_columns_ordered": ['bid_close', 'ask_close']
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
            self.timesteps_history = metadata.get("timesteps_history", self.timesteps_history)
            self.feature_columns_ordered_for_metadata = metadata.get("feature_columns_ordered", []) # 加載列順序
            start_dt = pd.to_datetime(self.start_time_iso, utc=True); end_dt = pd.to_datetime(self.end_time_iso, utc=True)
            granularity_freq_str = f"{get_granularity_seconds(self.granularity)}s" # <--- 'S' 改為 's'
            self.aligned_timestamps = pd.date_range(start=start_dt, end=end_dt, freq=granularity_freq_str, name='time')
            for symbol in self.symbols:
                mmap_features_path = self.dataset_mmap_dir / f"{symbol}_features.mmap"
                if mmap_features_path.exists():
                    self.processed_features_memmaps[symbol] = np.memmap(str(mmap_features_path), dtype=np.float32, mode=self.mmap_mode, shape=(self.total_aligned_steps, self.num_features_per_symbol))
                else: raise FileNotFoundError(f"特徵mmap文件未找到: {mmap_features_path}")
                mmap_prices_path = self.dataset_mmap_dir / f"{symbol}_raw_prices.mmap"
                raw_price_cols_count = len(metadata.get("raw_price_columns_ordered", ['bid_close', 'ask_close']))
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
        """關閉所有內存映射文件。重要：在程序退出前調用。"""
        # 更健壯地檢查 logger 是否仍然可用
        can_log_info = False
        # 首先檢查 logger 是否在全局作用域中定義並且不是 None
        if 'logger' in globals() and globals()['logger'] is not None:
            # 然後檢查 logger 是否有 info 方法
            current_logger = globals()['logger']
            if hasattr(current_logger, 'info') and callable(current_logger.info):
                # 再檢查是否有有效的 handlers，表明它還能輸出
                if hasattr(current_logger, 'hasHandlers') and current_logger.hasHandlers() and current_logger.handlers:
                     # 最後一層保護，確保 isEnabledFor INFO
                     if hasattr(current_logger, 'isEnabledFor') and current_logger.isEnabledFor(logging.INFO): # logging 需要導入
                        can_log_info = True
        
        log_func = logger.info if can_log_info else lambda msg: print(f"INFO (fallback print): {msg}")
        
        log_func(f"正在關閉數據集 {self.dataset_id} 的內存映射文件...") # dataset_id 可能在此時未定義如果__init__失敗
        dataset_id_to_log = getattr(self, 'dataset_id', 'UnknownDatasetID_in_close')


        log_func(f"正在關閉數據集 {dataset_id_to_log} 的內存映射文件...")


        for symbol in self.symbols: # self.symbols 也可能未定義如果__init__失敗
            symbols_to_iterate = getattr(self, 'symbols', [])
            for symbol_item in symbols_to_iterate:
                if self.processed_features_memmaps.get(symbol_item) is not None:
                    del self.processed_features_memmaps[symbol_item]
                    self.processed_features_memmaps[symbol_item] = None
                if self.raw_prices_memmaps.get(symbol_item) is not None:
                    del self.raw_prices_memmaps[symbol_item]
                    self.raw_prices_memmaps[symbol_item] = None
        
        # import gc; gc.collect() # 在 __del__ 中通常不建議強制gc
        
        log_func(f"數據集 {dataset_id_to_log} 的內存映射文件已關閉。")

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