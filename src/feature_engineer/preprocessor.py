# src/feature_engineer/preprocessor.py
"""
數據預處理模組
負責將原始的價量數據轉換為模型訓練所需的特徵。
主要步驟包括計算對數回報率、處理成交量以及Z-score標準化。
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional # <--- 已修正導入 Optional
from sklearn.preprocessing import StandardScaler

try:
    from common.config import PRICE_COLUMNS, TIMESTEPS # TIMESTEPS 用於滑動窗口標準化
    from common.logger_setup import logger
except ImportError:
    import sys
    from pathlib import Path
    # project_root = Path(__file__).resolve().parent.parent.parent # 移除
    # src_path = project_root / "src" # 移除
    # if str(project_root) not in sys.path: # 移除
    #     sys.path.insert(0, str(project_root)) # 移除
    try:
        # 假設 PYTHONPATH 已設定，這些導入應該能工作
        from src.common.config import PRICE_COLUMNS, TIMESTEPS
        from src.common.logger_setup import logger
        logger.info("Direct run Preprocessor: Successfully re-imported common modules.")
    except ImportError as e_retry:
        import logging
        logger = logging.getLogger("preprocessor_fallback") # type: ignore
        logger.error(f"Direct run Preprocessor: Critical import error: {e_retry}", exc_info=True)
        PRICE_COLUMNS = ['bid_open', 'bid_high', 'bid_low', 'bid_close',
                         'ask_open', 'ask_high', 'ask_low', 'ask_close', 'volume']
        TIMESTEPS = 128


# --- 核心預處理函數 ---

def calculate_log_returns(series: pd.Series, epsilon: float = 1e-9) -> pd.Series:
    """計算單個價格序列的對數回報率。 log(price_t / price_{t-1})"""
    # series.shift(1) 創建一個滯後一期的序列
    # 確保分母不為零或負數
    shifted_series = series.shift(1)
    # 處理開頭的NaN和可能的0值
    # 如果原始價格為0或負數，對數回報率無意義，應處理為0或NaN
    valid_mask = (series > epsilon) & (shifted_series > epsilon)
    log_ret = pd.Series(np.nan, index=series.index, dtype=np.float64)
    log_ret[valid_mask] = np.log(series[valid_mask] / shifted_series[valid_mask])
    # 第一個回報率會是NaN，用0填充
    return log_ret.fillna(0.0)

def preprocess_symbol_data(df_symbol: pd.DataFrame,
                           fit_scalers: bool = True,
                           scalers: Optional[Dict[str, StandardScaler]] = None,
                           window_size: int = TIMESTEPS * 10 # 用於滑動標準化的窗口大小，例如10倍的TIMESTEPS
                          ) -> Tuple[pd.DataFrame, Optional[Dict[str, StandardScaler]]]:
    """
    對單個symbol的DataFrame進行預處理。

    :param df_symbol: 包含單個symbol原始價量數據的DataFrame，必須有'time'和PRICE_COLUMNS。
    :param fit_scalers: 是否擬合新的StandardScaler。訓練時為True，推斷/回測時為False。
    :param scalers: 如果fit_scalers為False，則傳入已擬合的scalers。
    :param window_size: 計算滑動均值和標準差以進行標準化的窗口大小。
    :return: 包含預處理後特徵的DataFrame，以及擬合/使用的scalers（如果fit_scalers=True）。
    """
    if df_symbol.empty:
        logger.warning("傳入 preprocess_symbol_data 的DataFrame為空。")
        return pd.DataFrame(), scalers

    logger.debug(f"開始預處理 symbol數據，行數: {len(df_symbol)}, fit_scalers: {fit_scalers}")
    processed_df = pd.DataFrame(index=df_symbol.index)
    processed_df['time'] = df_symbol['time'] # 保留時間列

    newly_fitted_scalers = {} if fit_scalers else None

    # 1. 價格特徵：計算主要價格（例如bid_close, ask_close, mid_price）的對數回報率
    # 這裡我們為所有OCHL價格計算回報率，模型可以自己學習哪些有用
    price_feature_cols = []
    for col_type in ['bid', 'ask']: # 可以擴展到 'mid'
        for ohlc in ['open', 'high', 'low', 'close']:
            price_col = f"{col_type}_{ohlc}"
            if price_col in df_symbol.columns:
                return_col_name = f"{price_col}_log_ret"
                processed_df[return_col_name] = calculate_log_returns(df_symbol[price_col])
                price_feature_cols.append(return_col_name)

    # 2. 成交量特徵：取對數並處理可能的0值
    volume_feature_col = None
    if 'volume' in df_symbol.columns:
        volume_col_name = "volume_log"
        # 加1防止log(0)，對於極大成交量，log轉換有助於縮小範圍
        processed_df[volume_col_name] = np.log1p(df_symbol['volume'])
        volume_feature_col = volume_col_name

    # 3. Z-Score 標準化
    # 對於訓練集，我們在滑動窗口上計算均值和標準差來擬合StandardScaler
    # 對於測試/驗證集，我們使用從訓練集得到的均值和標準差
    features_to_standardize = price_feature_cols
    if volume_feature_col:
        features_to_standardize.append(volume_feature_col)

    if not features_to_standardize:
        logger.warning("沒有生成任何需要標準化的特徵。")
        return processed_df.drop(columns=['time']), newly_fitted_scalers # 返回不含time的特徵

    # 進行滑動窗口標準化或使用已有的scalers
    for feature_col in features_to_standardize:
        if feature_col not in processed_df.columns:
            logger.warning(f"特徵列 {feature_col} 在processed_df中不存在，跳過標準化。")
            continue

        series_to_scale = processed_df[feature_col].values.reshape(-1, 1)

        if fit_scalers:
            # 注意：對於時間序列，嚴格來說，StandardScaler應該在訓練數據上擬合，
            # 然後用相同的均值和標準差轉換驗證集和測試集。
            # 另一種方法是使用滑動窗口的標準化（expanding window or rolling window）。
            # 這裡為了簡化，我們先在整個傳入的df_symbol上擬合，
            # 實際應用中，MemoryMappedDataset在準備數據時需要更嚴謹地處理訓練/驗證/測試集的劃分和scaler的擬合。
            # 或者，這裡可以實現一個滑動窗口標準化：
            if len(series_to_scale) >= window_size: # 只有數據足夠長時才進行滑動標準化
                rolling_mean = processed_df[feature_col].rolling(window=window_size, min_periods=max(1, window_size//10)).mean()
                rolling_std = processed_df[feature_col].rolling(window=window_size, min_periods=max(1, window_size//10)).std()
                
                # 填充開頭的NaN（由於rolling操作）
                rolling_mean = rolling_mean.bfill().fillna(0)
                rolling_std = rolling_std.bfill().fillna(1e-6) # 用小值填充std的NaN以避免除零

                processed_df[feature_col] = (processed_df[feature_col] - rolling_mean) / (rolling_std + 1e-9) # 加上epsilon避免除以0
                # 記錄下最後一個窗口的mean和std作為這個scaler的參數（一種近似）
                # 更好的方法是，如果這是訓練階段，則保存所有這些scaler的配置
                # 這裡我們暫不保存scaler對象，因為滑動標準化是在數據本身上操作
                # 如果需要保存，可以創建一個dummy scaler並設置其mean_和scale_
                # scaler = StandardScaler()
                # scaler.mean_ = np.array([rolling_mean.iloc[-1]])
                # scaler.scale_ = np.array([rolling_std.iloc[-1]])
                # newly_fitted_scalers[feature_col] = scaler
            else: # 數據太短，使用整體標準化
                scaler = StandardScaler()
                processed_df[feature_col] = scaler.fit_transform(series_to_scale).flatten()
                if newly_fitted_scalers is not None: # 確保newly_fitted_scalers已初始化
                    newly_fitted_scalers[feature_col] = scaler
            
            # 對標準化後的數據進行裁剪，防止極端值影響模型 (可選)
            processed_df[feature_col] = np.clip(processed_df[feature_col], -5.0, 5.0)

        elif scalers and feature_col in scalers:
            scaler = scalers[feature_col]
            processed_df[feature_col] = scaler.transform(series_to_scale).flatten()
            processed_df[feature_col] = np.clip(processed_df[feature_col], -5.0, 5.0) # 同樣裁剪
        else:
            logger.warning(f"fit_scalers=False 但未提供 {feature_col} 的scaler，或數據不足以滑動標準化。該特徵未標準化。")
            # 如果數據太短，即使fit_scalers=True也可能執行到這裡（如果上面整體標準化邏輯被移除）
            # 在這種情況下，可以選擇不標準化，或者用全局（例如整個symbol歷史）的均值/標準差（如果提前計算了）

    logger.debug(f"完成 symbol數據預處理。生成特徵列: {processed_df.columns.tolist()}")
    
    # 移除原始時間列，因為模型輸入通常不需要它（位置信息由序列順序和位置編碼提供）
    # 但 MemoryMappedDataset 可能仍需要時間列來對齊和切片
    # 這裡我們先返回包含時間的，由調用者決定是否保留
    # return processed_df.drop(columns=['time']), newly_fitted_scalers
    return processed_df, newly_fitted_scalers


# --- 主預處理流程函數 ---
def preprocess_data_for_model(
    data_map: Dict[str, pd.DataFrame], # 鍵是symbol，值是該symbol的原始數據DataFrame
    fit_scalers: bool = True,
    existing_scalers: Optional[Dict[str, Dict[str, StandardScaler]]] = None,
    window_size: int = TIMESTEPS * 10
) -> Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, Dict[str, StandardScaler]]]]:
    """
    對多個symbols的數據進行預處理。

    :param data_map: 字典，鍵為symbol字符串，值為包含該symbol原始數據的Pandas DataFrame。
                     每個DataFrame必須有 'time' 列和 PRICE_COLUMNS。
    :param fit_scalers: 是否為每個symbol的每個特徵擬合新的StandardScaler。
    :param existing_scalers: 如果fit_scalers為False，則傳入已擬合的scalers。
                             格式為 {symbol: {feature_name: StandardScaler_object}}
    :param window_size: 滑動標準化的窗口大小。
    :return: 一個字典，鍵為symbol，值為預處理後的特徵DataFrame。
             以及一個字典，包含所有新擬合的scalers (如果fit_scalers=True)。
    """
    logger.info(f"開始對 {len(data_map)} 個symbols的數據進行預處理。fit_scalers: {fit_scalers}")
    processed_data_map: Dict[str, pd.DataFrame] = {}
    all_fitted_scalers: Dict[str, Dict[str, StandardScaler]] = {} if fit_scalers else {}

    for symbol, df_raw_symbol_data in data_map.items():
        logger.debug(f"正在處理 symbol: {symbol}")
        symbol_scalers = None
        if not fit_scalers and existing_scalers and symbol in existing_scalers:
            symbol_scalers = existing_scalers[symbol]

        df_processed_symbol, fitted_symbol_scalers = preprocess_symbol_data(
            df_raw_symbol_data,
            fit_scalers=fit_scalers,
            scalers=symbol_scalers,
            window_size=window_size
        )
        processed_data_map[symbol] = df_processed_symbol
        if fit_scalers and fitted_symbol_scalers:
            all_fitted_scalers[symbol] = fitted_symbol_scalers
            logger.debug(f"為 {symbol} 擬合了 {len(fitted_symbol_scalers)} 個 scalers。")


    num_features_per_symbol = -1
    if processed_data_map:
        first_symbol = next(iter(processed_data_map))
        if not processed_data_map[first_symbol].empty:
            # 從處理後的DataFrame中移除 'time' 列（如果存在）來計算特徵數量
            cols_for_feature_count = [col for col in processed_data_map[first_symbol].columns if col != 'time']
            num_features_per_symbol = len(cols_for_feature_count)
    logger.info(f"數據預處理完成。每個symbol的特徵數量: {num_features_per_symbol}")

    return processed_data_map, (all_fitted_scalers if fit_scalers else None)


if __name__ == "__main__":
    # 這個部分只在直接運行 preprocessor.py 時執行，用於測試
    logger.info("正在直接運行 preprocessor.py 進行測試...")

    # 創建一個假的DataFrame作為輸入 (模擬從database_manager查詢到的數據)
    # 包含多個symbol的數據
    raw_data_eur_usd = {
        'time': pd.to_datetime(['2023-01-01T00:00:00Z', '2023-01-01T00:00:05Z',
                                '2023-01-01T00:00:10Z', '2023-01-01T00:00:15Z',
                                '2023-01-01T00:00:20Z'] * 200), # 乘以200使其長度為1000，大於TIMESTEPS*窗口因子
        'bid_open': np.random.rand(1000) * 0.01 + 1.1,
        'bid_high': np.random.rand(1000) * 0.01 + 1.105,
        'bid_low': np.random.rand(1000) * 0.01 + 1.095,
        'bid_close': np.random.rand(1000) * 0.01 + 1.1,
        'ask_open': np.random.rand(1000) * 0.01 + 1.101,
        'ask_high': np.random.rand(1000) * 0.01 + 1.106,
        'ask_low': np.random.rand(1000) * 0.01 + 1.096,
        'ask_close': np.random.rand(1000) * 0.01 + 1.101,
        'volume': np.random.randint(100, 1000, size=1000)
    }
    df_eur_usd = pd.DataFrame(raw_data_eur_usd)
    # 確保所有PRICE_COLUMNS都存在
    for col in PRICE_COLUMNS:
        if col not in df_eur_usd.columns and col not in ['symbol', 'time']:
            df_eur_usd[col] = 0.0


    raw_data_usd_jpy = {
        'time': pd.to_datetime(['2023-01-01T00:00:00Z', '2023-01-01T00:00:05Z',
                                '2023-01-01T00:00:10Z', '2023-01-01T00:00:15Z',
                                '2023-01-01T00:00:20Z'] * 200),
        'bid_open': np.random.rand(1000) * 1 + 150.0,
        'bid_close': np.random.rand(1000) * 1 + 150.0,
        'ask_open': np.random.rand(1000) * 1 + 150.1,
        'ask_close': np.random.rand(1000) * 1 + 150.1,
        'volume': np.random.randint(500, 2000, size=1000)
    }
    # 簡化 USD_JPY 的列，只用 open/close 和 volume，其他用0填充
    df_usd_jpy = pd.DataFrame(raw_data_usd_jpy)
    for col in PRICE_COLUMNS:
        if col not in df_usd_jpy.columns and col not in ['symbol', 'time']:
            df_usd_jpy[col] = 0.0


    test_data_map = {
        "EUR_USD": df_eur_usd,
        "USD_JPY": df_usd_jpy
    }

    logger.info("--- 測試 fit_scalers = True ---")
    processed_map_train, fitted_scalers_map = preprocess_data_for_model(
        test_data_map.copy(), # 傳遞副本
        fit_scalers=True,
        window_size=TIMESTEPS * 2 # 測試時用較小的滑動窗口
    )

    if processed_map_train and fitted_scalers_map:
        print("\nEUR_USD 處理後數據 (前5條):")
        print(processed_map_train["EUR_USD"].head())
        print(f"\nEUR_USD 擬合的Scalers數量: {len(fitted_scalers_map.get('EUR_USD', {}))}")
        # print(f"EUR_USD bid_close_log_ret Scaler mean: {fitted_scalers_map['EUR_USD']['bid_close_log_ret'].mean_}")
        # print(f"EUR_USD bid_close_log_ret Scaler scale (std): {fitted_scalers_map['EUR_USD']['bid_close_log_ret'].scale_}")

        print("\nUSD_JPY 處理後數據 (前5條):")
        print(processed_map_train["USD_JPY"].head())
    else:
        print("預處理失敗或未返回結果 (fit_scalers=True)。")

    logger.info("\n--- 測試 fit_scalers = False (使用之前擬合的scalers) ---")
    # 只有當 fitted_scalers_map 真的包含有效的、非滑動窗口的scaler時，這個測試才有意義
    # 對於當前的滑動窗口實現，這個分支通常會被跳過或不產生預期效果
    if fitted_scalers_map is not None and fitted_scalers_map: # 只有當真的有傳統scaler時才執行
        processed_map_test, _ = preprocess_data_for_model(
            test_data_map.copy(), # 傳遞副本
            fit_scalers=False,
            existing_scalers=fitted_scalers_map,
            window_size=TIMESTEPS * 2
        )
        if processed_map_test:
            print("\nEUR_USD 使用已有Scaler處理後數據 (前5條):")
            print(processed_map_test["EUR_USD"].head())
        else:
            print("預處理失敗或未返回結果 (fit_scalers=False)。")
    else:
        print("由於未使用傳統StandardScaler擬合（或擬合失敗），跳過 fit_scalers=False 的測試。")

    print("\npreprocessor.py 測試執行完畢。")