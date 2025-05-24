# src/data_manager/database_manager.py
"""
數據庫管理模組
負責處理所有與SQLite資料庫的交互，包括表創建、數據插入和查詢。
"""
import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import logging # <--- 將 import logging 移到文件頂部，確保始終可用

# --- 嘗試導入配置和日誌記錄器 ---
# 這種結構允許在直接運行此文件進行測試時，也能正確設置PYTHONPATH
try:
    # 這種導入方式假設 Oanda_Trading_Bot 的根目錄在 PYTHONPATH 中
    # 或者此文件是被根目錄下的腳本調用的
    from common.config import DATABASE_PATH, PRICE_COLUMNS, GRANULARITY
    from common.logger_setup import logger
except ModuleNotFoundError:
    # 如果直接運行此文件 (python src/data_manager/database_manager.py)
    # 並且 Oanda_Trading_Bot 根目錄不在 PYTHONPATH 中，則會觸發此處
    # 我們需要手動將 src 的父目錄 (即專案根目錄) 添加到 sys.path
    import sys
    # Path(__file__) 是 database_manager.py 的路徑
    # .resolve().parent 是 data_manager 目錄
    # .parent 是 src 目錄
    # .parent 是專案根目錄 Oanda_Trading_Bot
    project_root = Path(__file__).resolve().parent.parent.parent
    src_path = project_root / "src" # 確保我們添加的是包含 common 的 src 的父目錄
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        # print(f"Added to sys.path for direct run: {project_root}") # 調試信息

    # 再次嘗試導入
    try:
        from src.common.config import DATABASE_PATH, PRICE_COLUMNS, GRANULARITY
        from src.common.logger_setup import logger
        logger.info("Direct run: Successfully re-imported common modules after path adjustment.")
    except ImportError as e_retry:
        # 如果重試仍然失敗，則使用後備
        logger = logging.getLogger("database_manager_fallback")
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(ch)
        logger.error(f"Direct run: Failed to re-import common modules even after path adjustment: {e_retry}. Using fallback logger and config.", exc_info=True)
        # 定義後備的 DATABASE_PATH 等，如果需要
        DATABASE_PATH = project_root / "data" / "database" / "fallback_oanda_s5_data.db"
        PRICE_COLUMNS = ['bid_open', 'bid_high', 'bid_low', 'bid_close',
                         'ask_open', 'ask_high', 'ask_low', 'ask_close', 'volume'] # 基礎的
        GRANULARITY = "S5"
        logger.warning(f"Using fallback DATABASE_PATH: {DATABASE_PATH}")


# --- 表定義 ---
# 歷史K線數據表
TABLE_HISTORICAL_DATA = "historical_price_data"
# 數據下載元數據表 (記錄每個symbol和時間段是否已下載)
TABLE_DOWNLOAD_METADATA = "download_metadata"
# 模型註冊表 (記錄訓練好的模型信息)
TABLE_MODEL_REGISTRY = "model_registry"


def get_db_connection() -> sqlite3.Connection:
    """
    獲取並返回一個資料庫連接。
    如果資料庫文件所在的目錄不存在，會嘗試創建它。
    """
    try:
        db_dir = DATABASE_PATH.parent
        db_dir.mkdir(parents=True, exist_ok=True) # 確保目錄存在
        conn = sqlite3.connect(DATABASE_PATH, timeout=10) # 增加timeout防止鎖定問題
        conn.execute("PRAGMA journal_mode=WAL;")  # 啟用WAL模式以提高並發性能和減少鎖定
        conn.execute("PRAGMA busy_timeout = 5000;") # 設置忙碌超時為5秒
        return conn
    except sqlite3.Error as e:
        logger.error(f"連接資料庫 {DATABASE_PATH} 失敗: {e}", exc_info=True)
        raise
    except Exception as e: # 捕獲可能的 Path.mkdir 錯誤等
        logger.error(f"創建資料庫目錄或連接時發生未知錯誤: {e}", exc_info=True)
        raise


def create_tables_if_not_exist():
    """
    檢查並創建專案所需的資料庫表 (如果它們還不存在)。
    """
    logger.info(f"檢查並初始化資料庫表於: {DATABASE_PATH}")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 歷史價格數據表
            price_cols_for_sql = []
            for col in PRICE_COLUMNS:
                if col not in ['symbol', 'time', 'granularity', 'volume']:
                    price_cols_for_sql.append(f"    {col} REAL")
            
            price_columns_sql_str = ",\n".join(price_cols_for_sql)
            volume_sql_str = "    volume INTEGER" if 'volume' in PRICE_COLUMNS else ""

            final_columns_list = []
            if price_columns_sql_str:
                final_columns_list.append(price_columns_sql_str)
            if volume_sql_str:
                final_columns_list.append(volume_sql_str)
            
            all_extra_columns_sql = ",\n".join(final_columns_list)
            # 如果 all_extra_columns_sql 不為空，則在其前面加上逗號和換行
            if all_extra_columns_sql:
                all_extra_columns_sql = ",\n" + all_extra_columns_sql


            create_historical_data_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {TABLE_HISTORICAL_DATA} (
                symbol TEXT NOT NULL,
                time TEXT NOT NULL,
                granularity TEXT NOT NULL{all_extra_columns_sql},
                PRIMARY KEY (symbol, time, granularity)
            );
            """
            cursor.execute(create_historical_data_table_sql)
            logger.debug(f"表 '{TABLE_HISTORICAL_DATA}' 已檢查/創建。")

            # 創建索引以加速查詢
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_historical_symbol_time ON {TABLE_HISTORICAL_DATA} (symbol, time);")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_historical_granularity ON {TABLE_HISTORICAL_DATA} (granularity);")


            # 數據下載元數據表
            create_download_metadata_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {TABLE_DOWNLOAD_METADATA} (
                symbol TEXT NOT NULL,
                granularity TEXT NOT NULL,
                start_time_iso TEXT NOT NULL,
                end_time_iso TEXT NOT NULL,
                is_complete BOOLEAN DEFAULT FALSE,
                last_downloaded_candle_time TEXT,
                downloaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, granularity, start_time_iso, end_time_iso)
            );
            """
            cursor.execute(create_download_metadata_table_sql)
            logger.debug(f"表 '{TABLE_DOWNLOAD_METADATA}' 已檢查/創建。")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_metadata_symbol_granularity ON {TABLE_DOWNLOAD_METADATA} (symbol, granularity);")


            # 模型註冊表
            create_model_registry_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {TABLE_MODEL_REGISTRY} (
                model_id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,
                symbols_trained_with TEXT,
                timesteps_setting INTEGER,
                max_symbols_setting INTEGER,
                granularity_setting TEXT,
                model_path TEXT NOT NULL,
                training_completed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_loaded_at TEXT,
                performance_metrics TEXT,
                notes TEXT
            );
            """
            cursor.execute(create_model_registry_table_sql)
            logger.debug(f"表 '{TABLE_MODEL_REGISTRY}' 已檢查/創建。")

            conn.commit()
        logger.info("資料庫表初始化完成。")
    except sqlite3.Error as e:
        logger.error(f"創建資料庫表時發生 SQLite 錯誤: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"創建資料庫表時發生未知錯誤: {e}", exc_info=True)

# ... (insert_historical_data, query_historical_data, update_download_metadata, get_download_metadata, register_model, get_model_registry_entry 函數保持不變，
#      但它們現在會使用正確導入的 logger 和 DATABASE_PATH)

# --- 保持原有的函數實現 ---
def insert_historical_data(df: pd.DataFrame, symbol: str, granularity: str):
    if df.empty:
        logger.debug(f"傳入 insert_historical_data 的 DataFrame 為空 (symbol: {symbol}, granularity: {granularity})，不執行插入。")
        return
    logger.debug(f"準備為 {symbol} ({granularity}) 插入 {len(df)} 條歷史數據...")
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        try:
            df['time'] = pd.to_datetime(df['time'])
        except Exception as e:
            logger.error(f"轉換 'time' 列到 datetime 對象失敗: {e}", exc_info=True)
            return
    try:
        if df['time'].dt.tz is None:
            df['time_iso'] = df['time'].dt.tz_localize('UTC').dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        else:
            df['time_iso'] = df['time'].dt.tz_convert('UTC').dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    except Exception as e:
        logger.error(f"將時間轉換為 ISO 格式時出錯 for {symbol} ({granularity}): {e}", exc_info=True)
        return

    # 列的順序: symbol, time_iso, granularity, 然後是 PRICE_COLUMNS 中的其他列 (除了 symbol, time, volume), 最後是 volume (如果存在)
    cols_for_insert_df = ['symbol_val', 'time_iso', 'granularity_val']
    for col in PRICE_COLUMNS:
        if col not in ['symbol', 'time', 'granularity', 'volume']:
            cols_for_insert_df.append(col)
    if 'volume' in PRICE_COLUMNS:
        cols_for_insert_df.append('volume')
    
    df_to_insert = df.copy()
    df_to_insert['symbol_val'] = symbol
    df_to_insert['granularity_val'] = granularity

    for col in PRICE_COLUMNS:
        if col in df_to_insert.columns and df_to_insert[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_to_insert[col]):
                 df_to_insert[col] = df_to_insert[col].astype(object).where(pd.notnull(df_to_insert[col]), None)
    try:
        # 確保 df_to_insert 包含所有 cols_for_insert_df 中除了 'symbol_val', 'time_iso', 'granularity_val' 之外的列
        # 這些列應該來自 PRICE_COLUMNS
        actual_data_cols_in_df = [c for c in cols_for_insert_df if c not in ['symbol_val', 'time_iso', 'granularity_val']]
        
        # 檢查 df_to_insert 是否有所有需要的列
        missing_cols = [c for c in actual_data_cols_in_df if c not in df_to_insert.columns]
        if missing_cols:
            logger.error(f"DataFrame for {symbol} ({granularity}) 缺少列: {missing_cols}. 無法插入數據。Available: {df_to_insert.columns.tolist()}")
            return

        data_tuples = [tuple(row) for row in df_to_insert[cols_for_insert_df].itertuples(index=False, name=None)]
    except KeyError as e:
        logger.error(f"生成插入元組時缺少列: {e}. DataFrame columns: {df_to_insert.columns.tolist()}", exc_info=True)
        logger.error(f"Expected columns in order: {cols_for_insert_df}")
        return
    except Exception as e:
        logger.error(f"生成插入元組時發生未知錯誤: {e}", exc_info=True)
        return

    if not data_tuples:
        logger.warning(f"沒有為 {symbol} ({granularity}) 生成任何數據元組進行插入。")
        return

    placeholders = ', '.join(['?'] * len(cols_for_insert_df))
    sql_insert = f"INSERT OR IGNORE INTO {TABLE_HISTORICAL_DATA} VALUES ({placeholders});"
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql_insert, data_tuples)
            conn.commit()
            logger.info(f"成功為 {symbol} ({granularity}) 插入/忽略了 {len(data_tuples)} 條數據。受影響行數: {cursor.rowcount}")
    except sqlite3.Error as e:
        logger.error(f"插入歷史數據到資料庫時發生 SQLite 錯誤 for {symbol} ({granularity}): {e}", exc_info=True)
    except Exception as e:
        logger.error(f"插入歷史數據時發生未知錯誤 for {symbol} ({granularity}): {e}", exc_info=True)

def query_historical_data(symbol: str, granularity: str,
                          start_time_iso: Optional[str] = None,
                          end_time_iso: Optional[str] = None,
                          limit: Optional[int] = None) -> pd.DataFrame:
    logger.debug(f"查詢歷史數據: symbol={symbol}, granularity={granularity}, start={start_time_iso}, end={end_time_iso}, limit={limit}")
    conditions = ["symbol = ?", "granularity = ?"]
    params: List[Any] = [symbol, granularity]
    if start_time_iso:
        conditions.append("time >= ?")
        params.append(start_time_iso)
    if end_time_iso:
        conditions.append("time <= ?")
        params.append(end_time_iso)

    # 確保查詢的列與 PRICE_COLUMNS 一致，並且 time 在最前面
    select_cols_str = "time, " + ", ".join(PRICE_COLUMNS)
    query = f"SELECT {select_cols_str} FROM {TABLE_HISTORICAL_DATA}"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY time ASC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'], utc=True)
            for col in PRICE_COLUMNS: # PRICE_COLUMNS 中不包含 'symbol' 或 'granularity'
                if col not in ['symbol', 'time', 'volume'] and col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'volume' in PRICE_COLUMNS and 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype('int64')
        logger.info(f"查詢到 {len(df)} 條歷史數據 for {symbol} ({granularity}).")
        return df
    except sqlite3.Error as e:
        logger.error(f"查詢歷史數據時發生 SQLite 錯誤: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"查詢歷史數據時發生未知錯誤: {e}", exc_info=True)
        return pd.DataFrame()

def update_download_metadata(symbol: str, granularity: str,
                             start_time_iso: str, end_time_iso: str,
                             is_complete: bool, last_downloaded_candle_time: Optional[str] = None):
    logger.debug(f"更新元數據: {symbol}, {granularity}, {start_time_iso}-{end_time_iso}, complete={is_complete}, last_candle={last_downloaded_candle_time}")
    sql = f"""
    INSERT INTO {TABLE_DOWNLOAD_METADATA} (symbol, granularity, start_time_iso, end_time_iso, is_complete, last_downloaded_candle_time, downloaded_at)
    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(symbol, granularity, start_time_iso, end_time_iso) DO UPDATE SET
        is_complete = excluded.is_complete,
        last_downloaded_candle_time = excluded.last_downloaded_candle_time,
        downloaded_at = CURRENT_TIMESTAMP;
    """
    params = (symbol, granularity, start_time_iso, end_time_iso, is_complete, last_downloaded_candle_time)
    try:
        with get_db_connection() as conn:
            conn.execute(sql, params)
            conn.commit()
        logger.info(f"元數據已更新 for {symbol} ({granularity}) for range {start_time_iso} to {end_time_iso}.")
    except sqlite3.Error as e:
        logger.error(f"更新下載元數據時發生 SQLite 錯誤: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"更新下載元數據時發生未知錯誤: {e}", exc_info=True)

def get_download_metadata(symbol: str, granularity: str) -> List[Dict[str, Any]]:
    logger.debug(f"查詢下載元數據 for {symbol} ({granularity})")
    query = f"""
    SELECT symbol, granularity, start_time_iso, end_time_iso, is_complete, last_downloaded_candle_time, downloaded_at
    FROM {TABLE_DOWNLOAD_METADATA}
    WHERE symbol = ? AND granularity = ?
    ORDER BY start_time_iso ASC;
    """
    results = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (symbol, granularity))
            rows = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            for row in rows:
                results.append(dict(zip(column_names, row)))
        logger.info(f"查詢到 {len(results)} 條元數據記錄 for {symbol} ({granularity}).")
        return results
    except sqlite3.Error as e:
        logger.error(f"查詢下載元數據時發生 SQLite 錯誤: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"查詢下載元數據時發生未知錯誤: {e}", exc_info=True)
        return []

def register_model(model_id: str, model_type: str, symbols_list: List[str],
                   timesteps: int, max_symbols: int, granularity_val: str,
                   model_path_str: str, performance_metrics_json: Optional[str] = None, notes: Optional[str] = None):
    symbols_str = ",".join(sorted(list(set(symbols_list))))
    sql = f"""
    INSERT INTO {TABLE_MODEL_REGISTRY}
    (model_id, model_type, symbols_trained_with, timesteps_setting, max_symbols_setting, granularity_setting, model_path, performance_metrics, notes, training_completed_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(model_id) DO UPDATE SET
        model_type = excluded.model_type,
        symbols_trained_with = excluded.symbols_trained_with,
        timesteps_setting = excluded.timesteps_setting,
        max_symbols_setting = excluded.max_symbols_setting,
        granularity_setting = excluded.granularity_setting,
        model_path = excluded.model_path,
        performance_metrics = excluded.performance_metrics,
        notes = excluded.notes,
        training_completed_at = excluded.training_completed_at,
        last_loaded_at = NULL;
    """
    params = (model_id, model_type, symbols_str, timesteps, max_symbols, granularity_val, model_path_str, performance_metrics_json, notes)
    try:
        with get_db_connection() as conn:
            conn.execute(sql, params)
            conn.commit()
        logger.info(f"模型 {model_id} 已成功註冊/更新到數據庫。")
    except sqlite3.Error as e:
        logger.error(f"註冊模型 {model_id} 到數據庫時發生 SQLite 錯誤: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"註冊模型 {model_id} 時發生未知錯誤: {e}", exc_info=True)

def get_model_registry_entry(model_id: str) -> Optional[Dict[str, Any]]:
    query = f"SELECT * FROM {TABLE_MODEL_REGISTRY} WHERE model_id = ?;"
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (model_id,))
            row = cursor.fetchone()
            if row:
                column_names = [description[0] for description in cursor.description]
                entry = dict(zip(column_names, row))
                logger.info(f"查詢到模型註冊信息 for model_id: {model_id}")
                return entry
            else:
                logger.info(f"未找到 model_id: {model_id} 的模型註冊信息。")
                return None
    except sqlite3.Error as e:
        logger.error(f"查詢模型註冊表時發生 SQLite 錯誤 for model_id {model_id}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"查詢模型註冊表時發生未知錯誤 for model_id {model_id}: {e}", exc_info=True)
        return None

# --- `if __name__ == "__main__":` 和 `else:` 塊保持不變 ---
if __name__ == "__main__":
    # 這個部分只在直接運行 database_manager.py 時執行，用於測試
    print(f"正在直接運行 database_manager.py 進行測試...")
    # 確保 logger 在此作用域內已定義 (如果導入失敗，後備 logger 會被使用)
    logger.info(f"使用的資料庫路徑: {DATABASE_PATH.resolve()}")

    # 創建表
    create_tables_if_not_exist()
    logger.info("表創建測試完成 (如果表已存在則不會重複創建)。")

    # 測試插入數據 (示例)
    sample_data = {
        'time': pd.to_datetime(['2023-01-01T00:00:00Z', '2023-01-01T00:00:05Z', '2023-01-01T00:00:10Z']),
        'bid_open': [1.1, 1.1001, 1.1002], 'bid_high': [1.1005, 1.1006, 1.1007],
        'bid_low': [1.099, 1.0991, 1.0992], 'bid_close': [1.1001, 1.1002, 1.1003],
        'ask_open': [1.1002, 1.1003, 1.1004], 'ask_high': [1.1007, 1.1008, 1.1009],
        'ask_low': [1.0992, 1.0993, 1.0994], 'ask_close': [1.1003, 1.1004, 1.1005],
        'volume': [100, 120, 110]
    }
    for col in PRICE_COLUMNS: # 使用全局的PRICE_COLUMNS
        if col not in sample_data and col not in ['symbol', 'time']:
            sample_data[col] = [0.0] * len(sample_data['time'])
    sample_df = pd.DataFrame(sample_data)
    
    test_symbol = "EUR_USD_TEST_DB" # 更改測試符號以避免與之前可能的衝突
    test_granularity = GRANULARITY # 使用全局的GRANULARITY
    logger.info(f"準備插入測試數據 for {test_symbol} ({test_granularity})...")
    insert_historical_data(sample_df.copy(), test_symbol, test_granularity)

    logger.info(f"查詢剛插入的測試數據...")
    queried_df = query_historical_data(test_symbol, test_granularity, limit=5)
    if not queried_df.empty:
        print("\n查詢到的測試數據 (前5條):")
        print(queried_df.head())
        if pd.api.types.is_datetime64_any_dtype(queried_df['time']):
            print("時間列已成功轉換為 datetime 對象。")
        else:
            print("錯誤：時間列未轉換為 datetime 對象。")
    else:
        print(f"未能查詢到 {test_symbol} ({test_granularity}) 的測試數據。")

    test_start_iso = "2023-01-01T00:00:00.000000Z"
    test_end_iso = "2023-01-01T00:00:10.000000Z"
    # last_candle_time = sample_df['time'].iloc[-1].tz_localize('UTC').strftime('%Y-%m-%dT%H:%M:%S.%fZ') if not sample_df.empty else None # 舊的，會引發錯誤
    if not sample_df.empty:
        # sample_df['time'] 在創建時已經是UTC (因為 'Z')
        # 所以可以直接格式化，或者為了保險，先轉換一次確保是UTC
        last_candle_time_ts = sample_df['time'].iloc[-1]
        if last_candle_time_ts.tzinfo is None: # 理論上不應該是None，因為 'Z'
            last_candle_time_ts = last_candle_time_ts.tz_localize('UTC')
        else:
            last_candle_time_ts = last_candle_time_ts.tz_convert('UTC') # 確保是UTC
        last_candle_time = last_candle_time_ts.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    else:
        last_candle_time = None
    update_download_metadata(test_symbol, test_granularity, test_start_iso, test_end_iso, True, last_candle_time)
    metadata_entries = get_download_metadata(test_symbol, test_granularity)
    if metadata_entries:
        print("\n查詢到的元數據記錄:")
        for entry in metadata_entries:
            print(entry)
    else:
        print(f"未能查詢到 {test_symbol} ({test_granularity}) 的元數據。")

    register_model(
        model_id="test_model_002", model_type="SAC_Transformer", symbols_list=["EUR_USD_TEST_DB", "USD_JPY_TEST_DB"],
        timesteps=128, max_symbols=2, granularity_val=GRANULARITY, model_path_str="/path/to/model_v2.zip",
        performance_metrics_json='{"sharpe": 1.8, "max_drawdown": 0.08}', notes="進階測試模型"
    )
    model_entry = get_model_registry_entry("test_model_002")
    if model_entry:
        print("\n查詢到的模型註冊信息:")
        print(model_entry)
    else:
        print("未能查詢到測試模型的註冊信息。")

    print("\ndatabase_manager.py 測試執行完畢。")
else:
    create_tables_if_not_exist()