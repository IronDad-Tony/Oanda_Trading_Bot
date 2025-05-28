# src/data_manager/oanda_downloader.py
"""
OANDA數據下載器模組
負責從OANDA API下載歷史蠟燭圖數據，並將其存儲到數據庫。
"""
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone # 確保導入了datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Tuple
import time
from dateutil import parser as dateutil_parser
import sys # 確保 sys 被導入，以便在 __main__ 和 fallback 中使用
from pathlib import Path # 確保 Path 被導入

# 從 common 模組導入配置和日誌記錄器
try:
    from common.config import (
        OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_BASE_URL, ACCOUNT_CURRENCY,
        GRANULARITY as DEFAULT_GRANULARITY,
        PRICE_COLUMNS, OANDA_MAX_BATCH_CANDLES, OANDA_REQUEST_INTERVAL,
        get_granularity_seconds
    )
    from common.logger_setup import logger
    from data_manager.database_manager import ( # 注意這裡的路徑
        insert_historical_data,
        update_download_metadata,
        get_download_metadata
    )
except ImportError:
    # project_root 和 src_path 的計算保持不變 # 移除
    # project_root = Path(__file__).resolve().parent.parent.parent # 移除
    # src_path = project_root / "src" # 移除
    # if str(project_root) not in sys.path: # 移除
    #     sys.path.insert(0, str(project_root)) # 移除
    try:
        # 假設 PYTHONPATH 已設定，這些導入應該能工作
        from src.common.config import (
            OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_BASE_URL, ACCOUNT_CURRENCY,
            GRANULARITY as DEFAULT_GRANULARITY, PRICE_COLUMNS,
            OANDA_MAX_BATCH_CANDLES, OANDA_REQUEST_INTERVAL, get_granularity_seconds
        )
        from src.common.logger_setup import logger # 使用 src.common
        from src.data_manager.database_manager import ( # 使用 src.data_manager
            insert_historical_data, update_download_metadata, get_download_metadata
        )
        logger.info("Direct run OANDA Downloader: Successfully re-imported common modules after path adjustment.")
    except ImportError as e_retry:
        import logging
        logger = logging.getLogger("oanda_downloader_fallback") # type: ignore
        logger.error(f"Direct run OANDA Downloader: Critical import error: {e_retry}", exc_info=True)
        OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_BASE_URL, ACCOUNT_CURRENCY = None, None, None, None
        DEFAULT_GRANULARITY, PRICE_COLUMNS, OANDA_MAX_BATCH_CANDLES, OANDA_REQUEST_INTERVAL = "S5", [], 4800, 0.1
        def get_granularity_seconds(g): return 5
        def insert_historical_data(*args, **kwargs): logger.error("DB insert_historical_data not available.")
        def update_download_metadata(*args, **kwargs): logger.error("DB update_download_metadata not available.")
        def get_download_metadata(*args, **kwargs): logger.error("DB get_download_metadata not available."); return []

API_SESSION = requests.Session()
if OANDA_API_KEY: # 確保 OANDA_API_KEY 在此作用域可用
    API_SESSION.headers.update({
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type": "application/json",
        "Accept-Datetime-Format": "RFC3339"
    })
else:
    logger.critical("OANDA_API_KEY 未配置，數據下載器無法工作！")

def format_datetime_for_oanda(dt_obj: datetime) -> str:
    if dt_obj.tzinfo is None:
        dt_obj_utc = dt_obj.replace(tzinfo=timezone.utc)
    else:
        dt_obj_utc = dt_obj.astimezone(timezone.utc)
    return dt_obj_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

def fetch_candles_batch(symbol: str, granularity: str,
                        start_time_iso: str, end_time_iso: str,
                        price_type: str, # ADDED: 'B' for Bid, 'A' for Ask, 'M' for Mid
                        count: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
        logger.error("API金鑰或賬戶ID未配置，無法獲取蠟燭圖數據。")
        return None
    endpoint = f"{OANDA_BASE_URL}/instruments/{symbol}/candles"
    params: Dict[str, Any] = {"granularity": granularity, "price": price_type} # MODIFIED: Use price_type
    if count is not None:
        params["count"] = min(count, OANDA_MAX_BATCH_CANDLES)
        params["from"] = start_time_iso
    else:
        params["from"] = start_time_iso
        params["to"] = end_time_iso
    max_retries = 3
    for attempt in range(max_retries):
        response_obj = None
        try:
            logger.debug(f"請求OANDA API: {endpoint} with params: {params}")
            response_obj = API_SESSION.get(endpoint, params=params, timeout=20)
            response_obj.raise_for_status()
            data = response_obj.json()
            candles = data.get("candles", [])
            if not candles:
                logger.info(f"OANDA API for {symbol} ({granularity}) from {start_time_iso} to {end_time_iso} 返回了空的蠟燭列表。")
                return []
            logger.debug(f"OANDA API 響應 for {symbol} ({granularity}): 收到 {len(candles)} 根蠟燭。")
            return candles
        except requests.exceptions.HTTPError as http_err:
            logger.warning(f"OANDA API HTTP 錯誤 (第 {attempt + 1}/{max_retries} 次嘗試) for {symbol}: {http_err}")
            if http_err.response is not None: response_obj = http_err.response
            if response_obj is not None:
                if response_obj.status_code == 400 and "maximum pips" in response_obj.text.lower():
                     logger.error(f"OANDA錯誤：請求的範圍過大 (maximum pips) for {symbol}。Params: {params}")
                     return None
                if response_obj.status_code == 400 and "future" in response_obj.text.lower():
                     logger.warning(f"OANDA警告：請求的時間在未來 for {symbol}. Params: {params}. 返回空列表。")
                     return []
                if response_obj.status_code == 401 or response_obj.status_code == 403:
                    logger.error(f"OANDA API 認證失敗 (401/403)。請檢查API金鑰。")
                    return None
                if response_obj.status_code == 404:
                    logger.error(f"OANDA API 資源未找到 (404) for {symbol}. 可能交易對不支持該粒度或已退市。")
                    return None
        except requests.exceptions.RequestException as req_err:
            logger.warning(f"OANDA API 請求錯誤 (第 {attempt + 1}/{max_retries} 次嘗試) for {symbol}: {req_err}")
        except Exception as e:
            logger.error(f"處理OANDA API響應時發生未知錯誤 (第 {attempt + 1}/{max_retries} 次嘗試) for {symbol}: {e}", exc_info=True)
        if attempt < max_retries - 1:
            time.sleep((attempt + 1) * 2)
        else:
            logger.error(f"獲取 {symbol} ({granularity}) 蠟燭圖數據失敗，已達最大重試次數。")
            return None
    return None

def parse_candles_to_dataframe(candles_data: List[Dict[str, Any]], symbol: str) -> pd.DataFrame:
    if not candles_data: return pd.DataFrame()
    records = []
    # This function expects candles_data to have 'bid' and 'ask' keys if it's from a 'BA' request.
    # For the new approach, this function will NOT be directly used by download_historical_data_for_period.
    # It's kept for potential other uses or if 'BA' requests are ever fixed/used elsewhere.
    # The new logic in download_historical_data_for_period will construct records directly.
    for candle in candles_data:
        if not candle.get("complete", True):
            logger.debug(f"跳過不完整的蠟燭圖數據: {candle.get('time')} for {symbol}")
            continue
        try:
            record = {
                "time": dateutil_parser.isoparse(candle["time"]), "symbol": symbol,
                "bid_open": float(candle["bid"]["o"]), "bid_high": float(candle["bid"]["h"]),
                "bid_low": float(candle["bid"]["l"]), "bid_close": float(candle["bid"]["c"]),
                "ask_open": float(candle["ask"]["o"]), "ask_high": float(candle["ask"]["h"]),
                "ask_low": float(candle["ask"]["l"]), "ask_close": float(candle["ask"]["c"]),
                "volume": int(candle["volume"]),
            }
            for pc_col in PRICE_COLUMNS: # PRICE_COLUMNS 應從config導入
                if pc_col not in record and pc_col not in ['symbol', 'time']:
                    record[pc_col] = float('nan')
            records.append(record)
        except KeyError as e: logger.warning(f"解析蠟燭數據時缺少鍵: {e} in candle: {candle}")
        except ValueError as e: logger.warning(f"解析蠟燭數據時值錯誤: {e} in candle: {candle}")
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.sort_values(by="time").reset_index(drop=True)

def download_historical_data_for_period(symbol: str, granularity: str,
                                        period_start_dt: datetime, period_end_dt: datetime,
                                        progress_callback: Optional[callable] = None,
                                        total_segments_for_progress: int = 1,
                                        current_segment_for_progress: int = 1):
    logger.info(f"開始為 {symbol} ({granularity}) 下載 Bid/Ask 數據，時段: {format_datetime_for_oanda(period_start_dt)} 至 {format_datetime_for_oanda(period_end_dt)}")
    all_candles_df = pd.DataFrame()
    current_fetch_start_dt = period_start_dt
    granularity_delta_seconds = get_granularity_seconds(granularity)
    api_request_failed_mid_period = False
    while current_fetch_start_dt < period_end_dt:
        batch_start_iso = format_datetime_for_oanda(current_fetch_start_dt)
        max_duration_for_batch = timedelta(seconds=(OANDA_MAX_BATCH_CANDLES - 1) * granularity_delta_seconds)
        batch_end_dt_theoretical = current_fetch_start_dt + max_duration_for_batch
        batch_end_dt_actual = min(batch_end_dt_theoretical, period_end_dt)
        batch_end_iso = format_datetime_for_oanda(batch_end_dt_actual)

        if batch_start_iso >= batch_end_iso and granularity_delta_seconds > 0:
            logger.debug(f"計算出的批次開始({batch_start_iso})已在結束({batch_end_iso})之後，停止批次下載。")
            break

        logger.debug(f"準備獲取批次 (Bid): {symbol} from {batch_start_iso} to {batch_end_iso}")
        bid_candles_data = fetch_candles_batch(symbol, granularity, batch_start_iso, batch_end_iso, price_type='B')
        time.sleep(OANDA_REQUEST_INTERVAL)
        
        logger.debug(f"準備獲取批次 (Ask): {symbol} from {batch_start_iso} to {batch_end_iso}")
        ask_candles_data = fetch_candles_batch(symbol, granularity, batch_start_iso, batch_end_iso, price_type='A')
        time.sleep(OANDA_REQUEST_INTERVAL)

        if bid_candles_data is None or ask_candles_data is None:
            logger.error(f"獲取 Bid 或 Ask 批次 for {symbol} from {batch_start_iso} to {batch_end_iso} 失敗。停止此大時間段的下載。")
            api_request_failed_mid_period = True
            break
        
        bid_records = []
        if bid_candles_data:
            for candle in bid_candles_data:
                if not candle.get("complete", True): continue
                try:
                    record = {
                        "time": dateutil_parser.isoparse(candle["time"]), "symbol": symbol,
                        "bid_open": float(candle["bid"]["o"]), "bid_high": float(candle["bid"]["h"]),
                        "bid_low": float(candle["bid"]["l"]), "bid_close": float(candle["bid"]["c"]),
                        "volume": int(candle["volume"]),
                    }
                    bid_records.append(record)
                except (KeyError, ValueError) as e:
                    logger.warning(f"解析 Bid 蠟燭數據時發生錯誤: {e} in candle: {candle} for symbol {symbol}")
        df_bid = pd.DataFrame(bid_records)

        ask_records = []
        if ask_candles_data:
            for candle in ask_candles_data:
                if not candle.get("complete", True): continue
                try:
                    record = {
                        "time": dateutil_parser.isoparse(candle["time"]), "symbol": symbol,
                        "ask_open": float(candle["ask"]["o"]), "ask_high": float(candle["ask"]["h"]),
                        "ask_low": float(candle["ask"]["l"]), "ask_close": float(candle["ask"]["c"]),
                        "volume": int(candle["volume"]), # Also parse volume from ask candles
                    }
                    ask_records.append(record)
                except (KeyError, ValueError) as e:
                    logger.warning(f"解析 Ask 蠟燭數據時發生錯誤: {e} in candle: {candle} for symbol {symbol}")
        df_ask = pd.DataFrame(ask_records)

        df_batch = pd.DataFrame()
        if not df_bid.empty and not df_ask.empty:
            df_batch = pd.merge(df_bid, df_ask.drop(columns=['volume'] if 'volume' in df_ask.columns and 'volume' in df_bid.columns else []), on=["time", "symbol"], how="outer")
        elif not df_bid.empty:
            df_batch = df_bid
            for col_ask in ["ask_open", "ask_high", "ask_low", "ask_close"]: df_batch[col_ask] = float('nan')
        elif not df_ask.empty:
            df_batch = df_ask
            for col_bid in ["bid_open", "bid_high", "bid_low", "bid_close"]: df_batch[col_bid] = float('nan')
            # Ensure 'volume' column exists if only ask data is present
            if 'volume' not in df_batch.columns and ask_candles_data and len(ask_candles_data) > 0 and 'volume' in ask_candles_data[0]:
                 # This case should be covered if df_ask is created with volume
                 pass


        if not df_batch.empty:
            # Ensure all PRICE_COLUMNS are present, filling with NaN if necessary
            # This is important if one side (bid/ask) was completely missing for a timestamp
            # or if the merge didn't create all expected columns.
            # The PRICE_COLUMNS from config should be the definitive list.
            for col_name in PRICE_COLUMNS:
                if col_name not in df_batch.columns and col_name not in ['symbol', 'time']: # symbol and time are from merge keys
                    df_batch[col_name] = float('nan')
            
            # Reorder columns to match PRICE_COLUMNS if defined and available
            # This helps ensure consistency before inserting into DB.
            # Example: expected_cols = [col for col in PRICE_COLUMNS if col in df_batch.columns]
            # df_batch = df_batch[expected_cols]


        if not df_batch.empty:
            all_candles_df = pd.concat([all_candles_df, df_batch], ignore_index=True)
            # Determine last_candle_time_in_batch carefully if df_batch could be sparse
            # It's safer to use current_fetch_start_dt advancement based on batch_end_dt_actual
            # However, if data is gappy, using actual last candle time is better.
            if not df_batch['time'].empty:
                last_candle_time_in_batch = df_batch['time'].max() # Use max time from the merged batch
                current_fetch_start_dt = last_candle_time_in_batch + timedelta(seconds=granularity_delta_seconds)
                if progress_callback:
                    progress_callback(format_datetime_for_oanda(last_candle_time_in_batch), total_segments_for_progress, current_segment_for_progress)
            else: # No data in this specific bid/ask combined batch, advance by theoretical window
                current_fetch_start_dt = batch_end_dt_actual + timedelta(seconds=granularity_delta_seconds)

        elif not bid_candles_data and not ask_candles_data: # Both fetches returned empty lists (not None)
            logger.info(f"批次 for {symbol} (Bid/Ask) from {batch_start_iso} to {batch_end_iso} 未返回任何蠟燭數據。推進時間窗口...")
            current_fetch_start_dt = batch_end_dt_actual + timedelta(seconds=granularity_delta_seconds)
        else: # One or both fetches failed (returned None) - this case is handled by api_request_failed_mid_period check earlier
            # Or, parsing resulted in empty dataframes, but candles_data was not empty.
            logger.warning(f"批次 for {symbol} (Bid/Ask) from {batch_start_iso} to {batch_end_iso} 解析後無有效數據或部分失敗。推進時間窗口...")
            current_fetch_start_dt = batch_end_dt_actual + timedelta(seconds=granularity_delta_seconds)


    if not all_candles_df.empty:
        all_candles_df = all_candles_df.drop_duplicates(subset=['time']).sort_values(by='time').reset_index(drop=True)
        logger.info(f"為 {symbol} ({granularity}) 在時段內下載並解析了 {len(all_candles_df)} 條不重複的蠟燭數據。")
        insert_historical_data(all_candles_df, symbol, granularity) # insert_historical_data 應從database_manager導入
        last_dl_time_str = format_datetime_for_oanda(all_candles_df['time'].iloc[-1])
        is_segment_complete = not api_request_failed_mid_period
        update_download_metadata(symbol, granularity, # update_download_metadata 應從database_manager導入
                                 format_datetime_for_oanda(period_start_dt),
                                 format_datetime_for_oanda(period_end_dt),
                                 is_complete=is_segment_complete,
                                 last_downloaded_candle_time=last_dl_time_str)
    else:
        logger.info(f"時段 {format_datetime_for_oanda(period_start_dt)} 至 {format_datetime_for_oanda(period_end_dt)} for {symbol} ({granularity}) 沒有下載到任何新數據。")
        is_segment_complete_for_empty = not api_request_failed_mid_period
        update_download_metadata(symbol, granularity,
                                 format_datetime_for_oanda(period_start_dt),
                                 format_datetime_for_oanda(period_end_dt),
                                 is_complete=is_segment_complete_for_empty,
                                 last_downloaded_candle_time=None)

def manage_data_download_for_symbols(symbols: List[str],
                                     overall_start_str: str,
                                     overall_end_str: str,
                                     granularity: str = DEFAULT_GRANULARITY, # DEFAULT_GRANULARITY 應從config導入
                                     streamlit_progress_bar: Optional[Any] = None,
                                     streamlit_status_text: Optional[Any] = None):
    logger.info(f"開始數據下載管理任務 for symbols: {symbols}, range: {overall_start_str} to {overall_end_str}, granularity: {granularity}")
    try:
        overall_start_dt = dateutil_parser.isoparse(overall_start_str)
        overall_end_dt = dateutil_parser.isoparse(overall_end_str)
    except ValueError as e:
        logger.error(f"無效的日期時間格式: {overall_start_str} or {overall_end_str}. Error: {e}")
        if streamlit_status_text: streamlit_status_text.error(f"日期格式錯誤: {e}")
        return
    if overall_start_dt.tzinfo is None: overall_start_dt = overall_start_dt.replace(tzinfo=timezone.utc)
    if overall_end_dt.tzinfo is None: overall_end_dt = overall_end_dt.replace(tzinfo=timezone.utc)

    segments_to_download: List[Tuple[str, str, datetime, datetime]] = []
    for symbol_idx, symbol in enumerate(symbols):
        if streamlit_status_text:
            streamlit_status_text.info(f"檢查 {symbol} ({symbol_idx+1}/{len(symbols)}) 的現有數據...")
        metadata_list = sorted(get_download_metadata(symbol, granularity), key=lambda x: x['start_time_iso']) # get_download_metadata 應從database_manager導入
        current_covered_end_dt = overall_start_dt
        merged_complete_intervals: List[Tuple[datetime, datetime]] = []
        for meta in metadata_list:
            if meta['is_complete']:
                meta_start_dt = dateutil_parser.isoparse(meta['start_time_iso'])
                meta_end_dt = dateutil_parser.isoparse(meta['end_time_iso'])
                if meta['last_downloaded_candle_time']:
                     last_candle_dt = dateutil_parser.isoparse(meta['last_downloaded_candle_time'])
                     meta_end_dt = max(meta_end_dt, last_candle_dt)
                if not merged_complete_intervals or meta_start_dt > merged_complete_intervals[-1][1] + timedelta(seconds=get_granularity_seconds(granularity)):
                    merged_complete_intervals.append((meta_start_dt, meta_end_dt))
                else:
                    merged_complete_intervals[-1] = (merged_complete_intervals[-1][0], max(merged_complete_intervals[-1][1], meta_end_dt))
        for start_exist, end_exist in merged_complete_intervals:
            if current_covered_end_dt < start_exist:
                gap_end_dt = min(start_exist, overall_end_dt)
                if gap_end_dt > current_covered_end_dt:
                    segments_to_download.append((symbol, granularity, current_covered_end_dt, gap_end_dt))
                    logger.debug(f"  添加缺失段 for {symbol}: {format_datetime_for_oanda(current_covered_end_dt)} to {format_datetime_for_oanda(gap_end_dt)}")
            current_covered_end_dt = max(current_covered_end_dt, end_exist)
        if current_covered_end_dt < overall_end_dt:
            segments_to_download.append((symbol, granularity, current_covered_end_dt, overall_end_dt))
            logger.debug(f"  添加末尾缺失段 for {symbol}: {format_datetime_for_oanda(current_covered_end_dt)} to {format_datetime_for_oanda(overall_end_dt)}")

    total_download_segments = len(segments_to_download)
    if total_download_segments == 0:
        logger.info("所有請求的數據範圍均已在元數據中標記為已覆蓋或完成。")
        if streamlit_status_text: streamlit_status_text.success("所有數據均已是最新或已下載。")
        if streamlit_progress_bar: streamlit_progress_bar.progress(1.0)
        return
    logger.info(f"總共需要下載 {total_download_segments} 個時間段的數據。")
    if streamlit_progress_bar: streamlit_progress_bar.progress(0.0)

    for idx, (sym, gran, seg_start_dt, seg_end_dt) in enumerate(segments_to_download):
        segment_progress_start = idx / total_download_segments
        segment_progress_end = (idx + 1) / total_download_segments
        def segment_progress_callback(candle_time_iso: str, total_segs: int, current_seg_idx_plus_1: int):
            if streamlit_progress_bar and streamlit_status_text:
                try:
                    current_candle_dt = dateutil_parser.isoparse(candle_time_iso)
                    segment_duration_secs = (seg_end_dt - seg_start_dt).total_seconds()
                    progress_in_segment_secs = (current_candle_dt - seg_start_dt).total_seconds()
                    if segment_duration_secs > 0:
                        fine_progress_in_segment = progress_in_segment_secs / segment_duration_secs
                    else: fine_progress_in_segment = 1.0
                    overall_progress = segment_progress_start + (fine_progress_in_segment * (segment_progress_end - segment_progress_start))
                    streamlit_progress_bar.progress(min(1.0, overall_progress))
                    streamlit_status_text.info(f"下載中: {sym} ({gran}) - 段 {current_seg_idx_plus_1}/{total_download_segments} - 當前到 {candle_time_iso[:19]}")
                except Exception as e_cb: logger.warning(f"進度回調中發生錯誤: {e_cb}")
        if streamlit_status_text:
             streamlit_status_text.info(f"開始下載段 {idx+1}/{total_download_segments}: {sym} ({gran}) from {format_datetime_for_oanda(seg_start_dt)} to {format_datetime_for_oanda(seg_end_dt)}")
        download_historical_data_for_period(sym, gran, seg_start_dt, seg_end_dt, progress_callback=segment_progress_callback, total_segments_for_progress=total_download_segments, current_segment_for_progress=idx + 1)
        if streamlit_progress_bar: streamlit_progress_bar.progress(segment_progress_end)
    if streamlit_status_text: streamlit_status_text.success("所有數據下載任務完成！")
    if streamlit_progress_bar: streamlit_progress_bar.progress(1.0)
    logger.info("數據下載管理任務全部完成。")

if __name__ == "__main__":
    print(f"正在直接運行 oanda_downloader.py 進行測試...")
    # OANDA_API_KEY 應在頂部導入時已從 config 加載
    if OANDA_API_KEY is None:
        print("錯誤: OANDA_API_KEY 未配置，無法執行下載測試。請檢查 .env 文件或頂部的導入邏輯。")
        sys.exit(1)

    logger.info("OANDA下載器測試開始。")
    test_symbols = ["EUR_USD", "USD_JPY"]
    test_end_dt = datetime.now(timezone.utc) # datetime, timezone, timedelta 應在頂部導入
    test_start_dt = test_end_dt - timedelta(minutes=10)
    test_granularity = "S5"
    test_start_iso = format_datetime_for_oanda(test_start_dt) # format_datetime_for_oanda 應在頂部定義
    test_end_iso = format_datetime_for_oanda(test_end_dt)
    logger.info(f"測試下載參數: symbols={test_symbols}, start={test_start_iso}, end={test_end_iso}, granularity={test_granularity}")
    def cli_progress_callback(candle_time_str, total_segments, current_segment):
        print(f"\r  進度 (段 {current_segment}/{total_segments}): 最新蠟燭時間 {candle_time_str}", end="")
    manage_data_download_for_symbols(
        test_symbols, test_start_iso, test_end_iso, granularity=test_granularity,
        streamlit_progress_bar=None, streamlit_status_text=None
    )
    print("\nOANDA下載器測試執行完畢。")
    try:
        # query_historical_data 應在頂部從 database_manager 導入
        from src.data_manager.database_manager import query_historical_data
        for sym_test in test_symbols:
            df_check = query_historical_data(sym_test, test_granularity, test_start_iso, test_end_iso, limit=10)
            if not df_check.empty:
                print(f"\n從數據庫查詢 {sym_test} ({test_granularity}) 的前10條數據:")
                print(df_check)
            else:
                print(f"\n未能從數據庫查詢到 {sym_test} ({test_granularity}) 在測試範圍內的數據。")
    except ImportError:
        logger.critical("在 __main__ 塊中導入 query_historical_data 失敗，即使頂部已嘗試路徑調整。")
        sys.exit(1)
    except Exception as e_query_test:
        logger.error(f"測試查詢時發生錯誤: {e_query_test}", exc_info=True)