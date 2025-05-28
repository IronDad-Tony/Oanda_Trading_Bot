# src/trainer/trainer.py
"""
訓練流程管理器 (V2.2 - 修正Decimal與float運算)
"""
# ... (頂部的導入和後備導入邏輯與 V2.1 版本相同) ...
# <在此處粘貼您上一個版本 trainer.py 中從文件頂部到 get_default_policy_kwargs 函數定義之前的全部內容>
# 我將重新提供頂部導入，確保所有內容都在
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Callable
import time
import os
from datetime import datetime, timezone, timedelta # 確保 timedelta 導入
from decimal import Decimal, getcontext # 確保 Decimal 和 getcontext 導入
import sys
import logging

getcontext().prec = 30 # 設置Decimal精度

# Flag to prevent duplicate import logging
_import_logged = False

_logger_trainer_v22: logging.Logger # 使用新版本號
_config_trainer_v22: Dict[str, Any] = {}
try:
    from common.logger_setup import logger as common_logger_trainer_v22; _logger_trainer_v22 = common_logger_trainer_v22; logger = _logger_trainer_v22
    if not _import_logged:
        logger.debug("trainer.py (V2.2): Successfully imported logger from common.logger_setup.")
    from common.config import (TIMESTEPS as _TIMESTEPS, MAX_SYMBOLS_ALLOWED as _MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY as _ACCOUNT_CURRENCY, INITIAL_CAPITAL as _INITIAL_CAPITAL_CONFIG, OANDA_MARGIN_CLOSEOUT_LEVEL as _OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE as _TRADE_COMMISSION_PERCENTAGE, OANDA_API_KEY as _OANDA_API_KEY, ATR_PERIOD as _ATR_PERIOD, STOP_LOSS_ATR_MULTIPLIER as _STOP_LOSS_ATR_MULTIPLIER, MAX_ACCOUNT_RISK_PERCENTAGE as _MAX_ACCOUNT_RISK_PERCENTAGE, TRAINER_DEFAULT_TOTAL_TIMESTEPS as _TRAINER_DEFAULT_TOTAL_TIMESTEPS, TRAINER_MODEL_NAME_PREFIX as _TRAINER_MODEL_NAME_PREFIX, WEIGHTS_DIR as _WEIGHTS_DIR, TRAINER_SAVE_FREQ_STEPS as _TRAINER_SAVE_FREQ_STEPS, TRAINER_EVAL_FREQ_STEPS as _TRAINER_EVAL_FREQ_STEPS, TRAINER_N_EVAL_EPISODES as _TRAINER_N_EVAL_EPISODES, TRAINER_DETERMINISTIC_EVAL as _TRAINER_DETERMINISTIC_EVAL, BEST_MODEL_SUBDIR as _BEST_MODEL_SUBDIR, EARLY_STOPPING_PATIENCE as _EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA_PERCENT as _EARLY_STOPPING_MIN_DELTA_PERCENT, EARLY_STOPPING_MIN_EVALS as _EARLY_STOPPING_MIN_EVALS, LOG_TRANSFORMER_NORM_FREQ_STEPS as _LOG_TRANSFORMER_NORM_FREQ_STEPS, DEFAULT_TRAIN_START_ISO as _DEFAULT_TRAIN_START_ISO, DEFAULT_TRAIN_END_ISO as _DEFAULT_TRAIN_END_ISO, DEFAULT_EVAL_START_ISO as _DEFAULT_EVAL_START_ISO, DEFAULT_EVAL_END_ISO as _DEFAULT_EVAL_END_ISO, SAC_LEARNING_RATE as _SAC_LEARNING_RATE, SAC_BATCH_SIZE as _SAC_BATCH_SIZE, SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR as _SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR, SAC_LEARNING_STARTS_FACTOR as _SAC_LEARNING_STARTS_FACTOR, SAC_GAMMA as _SAC_GAMMA, SAC_ENT_COEF as _SAC_ENT_COEF, SAC_TRAIN_FREQ_STEPS as _SAC_TRAIN_FREQ_STEPS, SAC_GRADIENT_STEPS as _SAC_GRADIENT_STEPS, SAC_TAU as _SAC_TAU, TRANSFORMER_MODEL_DIM as _TRANSFORMER_MODEL_DIM, TRANSFORMER_NUM_LAYERS as _TRANSFORMER_NUM_LAYERS, TRANSFORMER_NUM_HEADS as _TRANSFORMER_NUM_HEADS, TRANSFORMER_FFN_DIM as _TRANSFORMER_FFN_DIM, TRANSFORMER_DROPOUT_RATE as _TRANSFORMER_DROPOUT_RATE, TRANSFORMER_OUTPUT_DIM_PER_SYMBOL as _TRANSFORMER_OUTPUT_DIM_PER_SYMBOL, DEVICE as _DEVICE, GRANULARITY as _GRANULARITY)
    _config_trainer_v22 = {name: val for name, val in locals().items() if name.isupper() and not name.startswith('_') and isinstance(val, (str, int, float, bool, Path, Decimal, torch.device))}
    _config_trainer_v22["DEFAULT_INITIAL_CAPITAL"] = _INITIAL_CAPITAL_CONFIG # 確保使用正確的鍵名
    if not _import_logged:
        logger.info("trainer.py (V2.2): Successfully imported and stored common.config values.")
    from data_manager.mmap_dataset import UniversalMemoryMappedDataset; from data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols; from data_manager.instrument_info_manager import InstrumentInfoManager, InstrumentDetails; from environment.trading_env import UniversalTradingEnvV4; from agent.sac_policy import CustomSACPolicy; from agent.sac_agent_wrapper import SACAgentWrapper; from trainer.callbacks import UniversalCheckpointCallback; from stable_baselines3.common.vec_env import DummyVecEnv
    if not _import_logged:
        logger.info("trainer.py (V2.2): Successfully imported other dependencies.")
        _import_logged = True
except ImportError as e_initial_import_trainer_v22:
    # ... (後備導入邏輯與 V2.1 版本相同) ...
    logger_temp_trainer_v22 = logging.getLogger("trainer_v22_fallback_initial"); logger_temp_trainer_v22.addHandler(logging.StreamHandler(sys.stdout)); logger_temp_trainer_v22.setLevel(logging.DEBUG); logger = logger_temp_trainer_v22
    if not _import_logged:
        logger.warning(f"trainer.py (V2.2): Initial import failed: {e_initial_import_trainer_v22}. Assuming PYTHONPATH is set correctly or this is a critical issue.")
    # project_root_trainer_v22 = Path(__file__).resolve().parent.parent.parent # 移除
    # if str(project_root_trainer_v22) not in sys.path: sys.path.insert(0, str(project_root_trainer_v22)); logger.info(f"trainer.py (V2.2): Added project root to sys.path: {project_root_trainer_v22}") # 移除
    try:
        # 假設 PYTHONPATH 已設定，這些導入應該能工作
        from src.common.logger_setup import logger as common_logger_retry_trainer_v22; logger = common_logger_retry_trainer_v22; 
        if not _import_logged:
            logger.info("trainer.py (V2.2): Successfully re-imported common_logger.")
        from src.common.config import *; 
        if not _import_logged:
            logger.info("trainer.py (V2.2): Successfully re-imported common.config.")
        _config_trainer_v22 = {k: v for k, v in locals().items() if k.isupper() and not k.startswith('_')}
        from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset; from src.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols; from src.data_manager.instrument_info_manager import InstrumentInfoManager, InstrumentDetails; from src.environment.trading_env import UniversalTradingEnvV4; from src.agent.sac_policy import CustomSACPolicy; from src.agent.sac_agent_wrapper import SACAgentWrapper; from src.trainer.callbacks import UniversalCheckpointCallback; from stable_baselines3.common.vec_env import DummyVecEnv
        if not _import_logged:
            logger.info("trainer.py (V2.2): Successfully re-imported other dependencies after path adjustment.")
            _import_logged = True
    except ImportError as e_retry_critical_trainer_v22:
        logger.error(f"trainer.py (V2.2): Critical import error after path adjustment: {e_retry_critical_trainer_v22}", exc_info=True); logger.warning("trainer.py (V2.2): Using fallback values for config (critical error during import).")
        _config_trainer_v22 = {"TIMESTEPS": 128, "MAX_SYMBOLS_ALLOWED": 20, "ACCOUNT_CURRENCY": "AUD", "DEFAULT_INITIAL_CAPITAL": 100000.0, "OANDA_MARGIN_CLOSEOUT_LEVEL": Decimal('0.50'), "TRADE_COMMISSION_PERCENTAGE": Decimal('0.0001'), "OANDA_API_KEY": None, "ATR_PERIOD": 14, "STOP_LOSS_ATR_MULTIPLIER": Decimal('2.0'), "MAX_ACCOUNT_RISK_PERCENTAGE": Decimal('0.01'), "TRAINER_DEFAULT_TOTAL_TIMESTEPS": 1000, "TRAINER_MODEL_NAME_PREFIX":"fallback", "WEIGHTS_DIR":Path("./fb_w"), "TRAINER_SAVE_FREQ_STEPS":100, "TRAINER_EVAL_FREQ_STEPS":50, "TRAINER_N_EVAL_EPISODES":1, "TRAINER_DETERMINISTIC_EVAL":True, "BEST_MODEL_SUBDIR":"best", "EARLY_STOPPING_PATIENCE":3, "EARLY_STOPPING_MIN_DELTA_PERCENT":0.01, "EARLY_STOPPING_MIN_EVALS":2, "LOG_TRANSFORMER_NORM_FREQ_STEPS":20, "DEFAULT_TRAIN_START_ISO":"2023-01-01T00:00:00Z", "DEFAULT_TRAIN_END_ISO":"2023-01-02T00:00:00Z", "DEFAULT_EVAL_START_ISO":"2023-01-02T00:00:00Z", "DEFAULT_EVAL_END_ISO":"2023-01-03T00:00:00Z", "DEVICE":torch.device("cpu")}
        for k_fallback, v_fallback in _config_trainer_v22.items(): globals()[k_fallback] = v_fallback
        sys.exit("Trainer critical import error in fallback for V2.2.")

# 從 _config_trainer_v22 中獲取值並賦給模組級常量
# (與V2.1版本相同，確保所有都從字典獲取)
# <在此處粘貼您上一個版本 trainer.py 中從 _config_trainer_v21 獲取全局常量的代碼塊>
TIMESTEPS = _config_trainer_v22.get("TIMESTEPS", 128); MAX_SYMBOLS_ALLOWED = _config_trainer_v22.get("MAX_SYMBOLS_ALLOWED", 20); ACCOUNT_CURRENCY = _config_trainer_v22.get("ACCOUNT_CURRENCY", "AUD"); DEFAULT_INITIAL_CAPITAL_CONFIG = _config_trainer_v22.get("DEFAULT_INITIAL_CAPITAL", 100000.0); OANDA_MARGIN_CLOSEOUT_LEVEL = _config_trainer_v22.get("OANDA_MARGIN_CLOSEOUT_LEVEL", Decimal('0.50')); TRADE_COMMISSION_PERCENTAGE = _config_trainer_v22.get("TRADE_COMMISSION_PERCENTAGE", Decimal('0.0001')); OANDA_API_KEY = _config_trainer_v22.get("OANDA_API_KEY", None); ATR_PERIOD = _config_trainer_v22.get("ATR_PERIOD", 14); STOP_LOSS_ATR_MULTIPLIER = _config_trainer_v22.get("STOP_LOSS_ATR_MULTIPLIER", Decimal('2.0')); MAX_ACCOUNT_RISK_PERCENTAGE = _config_trainer_v22.get("MAX_ACCOUNT_RISK_PERCENTAGE", Decimal('0.01')); TRAINER_DEFAULT_TOTAL_TIMESTEPS = _config_trainer_v22.get("TRAINER_DEFAULT_TOTAL_TIMESTEPS",1000000); TRAINER_MODEL_NAME_PREFIX = _config_trainer_v22.get("TRAINER_MODEL_NAME_PREFIX","sac_universal_trader"); WEIGHTS_DIR = _config_trainer_v22.get("WEIGHTS_DIR",Path("./weights")); TRAINER_SAVE_FREQ_STEPS = _config_trainer_v22.get("TRAINER_SAVE_FREQ_STEPS",20000); TRAINER_EVAL_FREQ_STEPS = _config_trainer_v22.get("TRAINER_EVAL_FREQ_STEPS",10000); TRAINER_N_EVAL_EPISODES = _config_trainer_v22.get("TRAINER_N_EVAL_EPISODES",3); TRAINER_DETERMINISTIC_EVAL = _config_trainer_v22.get("TRAINER_DETERMINISTIC_EVAL",True); BEST_MODEL_SUBDIR = _config_trainer_v22.get("BEST_MODEL_SUBDIR","best_model"); EARLY_STOPPING_PATIENCE = _config_trainer_v22.get("EARLY_STOPPING_PATIENCE",10); EARLY_STOPPING_MIN_DELTA_PERCENT = _config_trainer_v22.get("EARLY_STOPPING_MIN_DELTA_PERCENT",0.1); EARLY_STOPPING_MIN_EVALS = _config_trainer_v22.get("EARLY_STOPPING_MIN_EVALS",20); LOG_TRANSFORMER_NORM_FREQ_STEPS = _config_trainer_v22.get("LOG_TRANSFORMER_NORM_FREQ_STEPS",1000); DEFAULT_TRAIN_START_ISO = _config_trainer_v22.get("DEFAULT_TRAIN_START_ISO","2024-05-20T00:00:00Z"); DEFAULT_TRAIN_END_ISO = _config_trainer_v22.get("DEFAULT_TRAIN_END_ISO","2024-05-21T23:59:59Z"); DEFAULT_EVAL_START_ISO = _config_trainer_v22.get("DEFAULT_EVAL_START_ISO","2024-05-22T00:00:00Z"); DEFAULT_EVAL_END_ISO = _config_trainer_v22.get("DEFAULT_EVAL_END_ISO","2024-05-22T11:59:59Z"); SAC_LEARNING_RATE = _config_trainer_v22.get("SAC_LEARNING_RATE",3e-5); SAC_BATCH_SIZE = _config_trainer_v22.get("SAC_BATCH_SIZE",64); SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR = _config_trainer_v22.get("SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR",10000); SAC_LEARNING_STARTS_FACTOR = _config_trainer_v22.get("SAC_LEARNING_STARTS_FACTOR",200); SAC_GAMMA = _config_trainer_v22.get("SAC_GAMMA",0.95); SAC_ENT_COEF = _config_trainer_v22.get("SAC_ENT_COEF",'auto'); SAC_TRAIN_FREQ_STEPS = _config_trainer_v22.get("SAC_TRAIN_FREQ_STEPS",64); SAC_GRADIENT_STEPS = _config_trainer_v22.get("SAC_GRADIENT_STEPS",64); SAC_TAU = _config_trainer_v22.get("SAC_TAU",0.005); TRANSFORMER_MODEL_DIM = _config_trainer_v22.get("TRANSFORMER_MODEL_DIM",512); TRANSFORMER_NUM_LAYERS = _config_trainer_v22.get("TRANSFORMER_NUM_LAYERS",6); TRANSFORMER_NUM_HEADS = _config_trainer_v22.get("TRANSFORMER_NUM_HEADS",8); TRANSFORMER_FFN_DIM = _config_trainer_v22.get("TRANSFORMER_FFN_DIM",2048); TRANSFORMER_DROPOUT_RATE = _config_trainer_v22.get("TRANSFORMER_DROPOUT_RATE",0.1); TRANSFORMER_OUTPUT_DIM_PER_SYMBOL = _config_trainer_v22.get("TRANSFORMER_OUTPUT_DIM_PER_SYMBOL",128); DEVICE = _config_trainer_v22.get("DEVICE",torch.device("cpu")); GRANULARITY = _config_trainer_v22.get("GRANULARITY", "S5")


def get_default_policy_kwargs() -> Dict[str, Any]:
    # (與V2.1版本相同)
    return {"features_extractor_kwargs": dict(transformer_output_dim_per_symbol=TRANSFORMER_OUTPUT_DIM_PER_SYMBOL, model_dim=TRANSFORMER_MODEL_DIM, num_time_encoder_layers=TRANSFORMER_NUM_LAYERS // 2, num_cross_asset_layers=TRANSFORMER_NUM_LAYERS // 2, num_heads=TRANSFORMER_NUM_HEADS, ffn_dim=TRANSFORMER_FFN_DIM, dropout_rate=TRANSFORMER_DROPOUT_RATE, use_fourier_block=True, fourier_num_modes=min(32, TIMESTEPS // 2 + 1 if TIMESTEPS > 0 else 16), use_wavelet_block=True, wavelet_levels=3,), "net_arch": dict(pi=[256, 256], qf=[256, 256])}

def run_training_session(
    symbols_to_trade: List[str], all_symbols_for_data: List[str],
    train_start_iso: Optional[str] = None, train_end_iso: Optional[str] = None,
    eval_start_iso: Optional[str] = None, eval_end_iso: Optional[str] = None,
    granularity: str = GRANULARITY, total_timesteps: Optional[int] = None,
    initial_capital: Optional[float] = None, load_model_path: Optional[Union[str, Path]] = None,
    session_name_suffix: str = "", force_dataset_reload: bool = False,
    learning_rate: Optional[Union[float, Callable[[float], float]]] = None,
    batch_size: Optional[int] = None,
    policy_kwargs_override: Optional[Dict[str, Any]] = None,
    save_freq_override: Optional[int] = None, eval_freq_override: Optional[int] = None,
    streamlit_status_text: Optional[Any] = None, streamlit_progress_bar: Optional[Any] = None,
):
    train_start_iso = train_start_iso or DEFAULT_TRAIN_START_ISO
    train_end_iso = train_end_iso or DEFAULT_TRAIN_END_ISO
    eval_start_iso = eval_start_iso or DEFAULT_EVAL_START_ISO
    eval_end_iso = eval_end_iso or DEFAULT_EVAL_END_ISO
    total_timesteps = total_timesteps or TRAINER_DEFAULT_TOTAL_TIMESTEPS
    initial_capital_val = initial_capital if initial_capital is not None else float(DEFAULT_INITIAL_CAPITAL_CONFIG)

    logger.info("="*60); logger.info(f"開始新的訓練會話 (Trainer V2.2)..."); # 更新版本號
    # ... (後續的 run_training_session 日誌和步驟與 V2.1 版本相同，直到 checkpoint_callback 實例化) ...
    logger.info(f"可交易對象: {symbols_to_trade}"); logger.info(f"數據集所需對象: {all_symbols_for_data}"); logger.info(f"訓練數據: {train_start_iso} to {train_end_iso}"); logger.info(f"評估數據: {eval_start_iso} to {eval_end_iso}"); logger.info(f"目標總步數: {total_timesteps}"); logger.info(f"初始資金: {initial_capital_val} {ACCOUNT_CURRENCY}");
    if load_model_path: logger.info(f"嘗試從以下路徑加載模型: {load_model_path}")
    logger.info("="*60)
    if streamlit_status_text: streamlit_status_text.info("步驟 0/6: 初始化交易品種信息管理器...")
    instrument_manager = InstrumentInfoManager(force_refresh=False)
    if streamlit_status_text: streamlit_status_text.info("步驟1/6: 準備訓練數據集...")
    logger.info("準備訓練數據集 (檢查/下載)...")
    manage_data_download_for_symbols(
        all_symbols_for_data, train_start_iso, train_end_iso, granularity,
        streamlit_progress_bar=streamlit_progress_bar, # <--- 傳遞
        streamlit_status_text=streamlit_status_text     # <--- 傳遞
    )
    dataset_train = UniversalMemoryMappedDataset(
        symbols=all_symbols_for_data, start_time_iso=train_start_iso, end_time_iso=train_end_iso,
        granularity=granularity, timesteps_history=TIMESTEPS, force_reload=force_dataset_reload
    )
    if len(dataset_train) == 0: logger.error("訓練數據集為空！訓練無法繼續。"); return
    eval_env_vec: Optional[DummyVecEnv] = None; dataset_eval = None
    if eval_start_iso and eval_end_iso:
        if streamlit_status_text: streamlit_status_text.info("步驟 2/6: 準備評估數據集...")
        logger.info("準備評估數據集 (檢查/下載)...")
        manage_data_download_for_symbols(all_symbols_for_data, eval_start_iso, eval_end_iso, granularity, streamlit_progress_bar=streamlit_progress_bar, streamlit_status_text=streamlit_status_text)
        dataset_eval = UniversalMemoryMappedDataset(symbols=all_symbols_for_data, start_time_iso=eval_start_iso, end_time_iso=eval_end_iso, granularity=granularity, timesteps_history=TIMESTEPS, force_reload=force_dataset_reload)
        if len(dataset_eval) == 0: logger.warning("評估數據集為空！評估將被跳過。");
        else: eval_env = UniversalTradingEnvV4(dataset=dataset_eval, instrument_info_manager=instrument_manager, active_symbols_for_episode=symbols_to_trade, initial_capital=initial_capital_val); eval_env_vec = DummyVecEnv([lambda: eval_env])
    else: logger.info("未提供評估數據時間範圍，跳過評估環境創建。")
    if streamlit_status_text: streamlit_status_text.info("步驟 3/6: 創建訓練環境...")
    train_env = UniversalTradingEnvV4(dataset=dataset_train, instrument_info_manager=instrument_manager, active_symbols_for_episode=symbols_to_trade, initial_capital=initial_capital_val)
    train_env_vec = DummyVecEnv([lambda: train_env])
    if streamlit_status_text: streamlit_status_text.info("步驟 4/6: 初始化或加載SAC智能體...")
    current_policy_kwargs = get_default_policy_kwargs()
    if policy_kwargs_override: current_policy_kwargs.update(policy_kwargs_override)
    agent_wrapper = SACAgentWrapper(env=train_env_vec, policy_class=CustomSACPolicy, policy_kwargs=current_policy_kwargs, learning_rate=learning_rate or SAC_LEARNING_RATE, batch_size=batch_size or SAC_BATCH_SIZE, buffer_size_factor=SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR, learning_starts_factor=SAC_LEARNING_STARTS_FACTOR, gamma=SAC_GAMMA, ent_coef=SAC_ENT_COEF, train_freq_steps=SAC_TRAIN_FREQ_STEPS, gradient_steps=SAC_GRADIENT_STEPS, tau=SAC_TAU, verbose=0, seed=int(time.time()))
    session_id = f"{TRAINER_MODEL_NAME_PREFIX}{session_name_suffix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    session_save_dir = WEIGHTS_DIR / session_id; session_save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"本次訓練會話的模型將保存到: {session_save_dir}")
    latest_model_checkpoint_path = WEIGHTS_DIR / f"{TRAINER_MODEL_NAME_PREFIX}_latest.zip"
    path_to_load_from: Optional[Path] = None
    if load_model_path:
        path_to_load_from = Path(load_model_path)
        if not path_to_load_from.exists(): logger.warning(f"指定的模型路徑 {path_to_load_from} 不存在，將從頭訓練。"); path_to_load_from = None
    elif latest_model_checkpoint_path.exists():
        logger.info(f"檢測到最新的模型 checkpoint: {latest_model_checkpoint_path}"); path_to_load_from = latest_model_checkpoint_path
    if path_to_load_from:
        logger.info(f"正在從 {path_to_load_from} 加載預訓練模型...")
        try: agent_wrapper.load(path_to_load_from, env=train_env_vec)
        except Exception as e_load: logger.error(f"加載模型 {path_to_load_from} 失敗: {e_load}。將從頭訓練。", exc_info=True); path_to_load_from = None
    else: logger.info("沒有指定或有效的預訓練模型路徑，將從頭開始訓練。")
    if streamlit_status_text: streamlit_status_text.info("步驟 5/6: 設置訓練回調...")
    best_model_in_session_path = session_save_dir / BEST_MODEL_SUBDIR
    
    # --- 修正 early_stopping_min_delta_abs 的計算 ---
    es_min_delta_percent_decimal = Decimal(str(EARLY_STOPPING_MIN_DELTA_PERCENT))
    initial_capital_decimal = Decimal(str(initial_capital_val))
    min_delta_abs_val_for_callback: float
    if es_min_delta_percent_decimal > Decimal('0'):
        min_delta_abs_val_for_callback = float(initial_capital_decimal * (es_min_delta_percent_decimal / Decimal('100.0')))
    else:
        min_delta_abs_val_for_callback = float(Decimal('100.0')) # 默認絕對值，例如100 AUD

    checkpoint_callback = UniversalCheckpointCallback(
        save_freq=save_freq_override or TRAINER_SAVE_FREQ_STEPS,
        save_path=session_save_dir,
        name_prefix=TRAINER_MODEL_NAME_PREFIX,
        eval_env=eval_env_vec,
        eval_freq=eval_freq_override or TRAINER_EVAL_FREQ_STEPS,
        n_eval_episodes=TRAINER_N_EVAL_EPISODES,
        deterministic_eval=TRAINER_DETERMINISTIC_EVAL,
        best_model_save_path=best_model_in_session_path,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta_abs=min_delta_abs_val_for_callback, # <--- 使用修正後的值
        early_stopping_metric="eval/mean_final_portfolio_value",
        early_stopping_min_evals=EARLY_STOPPING_MIN_EVALS,
        log_transformer_norm_freq=LOG_TRANSFORMER_NORM_FREQ_STEPS,
        verbose=1
    )
    # --- 結束修正 ---

    if streamlit_status_text: streamlit_status_text.info("步驟 6/6: 開始訓練循環...")
    try:
        agent_wrapper.train(total_timesteps=total_timesteps, callback=[checkpoint_callback], reset_num_timesteps= (path_to_load_from is None))
        logger.info("訓練會話正常完成。");
        if streamlit_status_text: streamlit_status_text.success("訓練完成！")
    except KeyboardInterrupt: 
        logger.warning("訓練被用戶手動中斷 (KeyboardInterrupt)。")
        if streamlit_status_text: streamlit_status_text.warning("訓練已手動停止。")
    except Exception as e_train_main: 
        logger.error(f"訓練過程中發生嚴重錯誤: {e_train_main}", exc_info=True)
        if streamlit_status_text: streamlit_status_text.error(f"訓練出錯: {e_train_main}")
    finally:
        logger.info("訓練循環結束。正在關閉環境和數據集...")
        if hasattr(train_env_vec, 'close'): train_env_vec.close()
        if hasattr(eval_env_vec, 'close') and eval_env_vec is not None : eval_env_vec.close()
        if hasattr(dataset_train, 'close'): dataset_train.close()
        if 'dataset_eval' in locals() and hasattr(dataset_eval, 'close') and dataset_eval is not None: dataset_eval.close()
        logger.info("所有資源已關閉。")
        if streamlit_status_text: streamlit_status_text.info("所有資源已釋放。")


# --- if __name__ == "__main__": 測試塊 (與V2.1版本相同) ---
if __name__ == "__main__":
    # ... (與您上一個版本 trainer.py __main__ 測試塊相同的代碼) ...
    logger.info("正在直接運行 trainer.py 進行測試...")
    # 確保所有頂層導入的變量在此作用域可用
    if 'UniversalTradingEnvV4' not in globals() or UniversalTradingEnvV4 is None: logger.error("UniversalTradingEnvV4 is None. Test cannot proceed."); sys.exit(1)
    if 'SACAgentWrapper' not in globals() or SACAgentWrapper is None: logger.error("SACAgentWrapper is None. Test cannot proceed."); sys.exit(1)
    if 'OANDA_API_KEY' not in globals() or OANDA_API_KEY is None: logger.error("OANDA_API_KEY is None in __main__."); sys.exit(1)
    if 'DEFAULT_TRAIN_START_ISO' not in globals(): logger.error("DEFAULT_TRAIN_START_ISO is None in __main__."); sys.exit(1)

    main_test_symbols_to_trade = ["EUR_USD"]
    main_test_all_symbols_for_data = ["EUR_USD", "AUD_USD"]
    try:
        fixed_start_dt = datetime(2024, 5, 20, 10, 0, 0, tzinfo=timezone.utc)
        fixed_train_end_dt = datetime(2024, 5, 20, 10, 30, 0, tzinfo=timezone.utc)
        fixed_eval_start_dt = datetime(2024, 5, 20, 10, 30, 0, tzinfo=timezone.utc)
        fixed_eval_end_dt = datetime(2024, 5, 20, 10, 45, 0, tzinfo=timezone.utc)
    except ValueError: logger.error("固定測試日期無效"); sys.exit(1)
    main_train_start = format_datetime_for_oanda(fixed_start_dt)
    main_train_end = format_datetime_for_oanda(fixed_train_end_dt)
    main_eval_start = format_datetime_for_oanda(fixed_eval_start_dt)
    main_eval_end = format_datetime_for_oanda(fixed_eval_end_dt)
    main_total_timesteps = 200; main_save_freq = 50; main_eval_freq = 40
    logger.info(f"Trainer __main__ 測試: symbols={main_test_symbols_to_trade}, total_steps={main_total_timesteps}")
    try:
        run_training_session(
            symbols_to_trade=main_test_symbols_to_trade, all_symbols_for_data=main_test_all_symbols_for_data,
            train_start_iso=main_train_start, train_end_iso=main_train_end,
            eval_start_iso=main_eval_start, eval_end_iso=main_eval_end,
            total_timesteps=main_total_timesteps, initial_capital=5000.0,
            save_freq_override=main_save_freq, eval_freq_override=main_eval_freq,
            policy_kwargs_override={"features_extractor_kwargs": dict(model_dim=64, num_time_encoder_layers=1, num_cross_asset_layers=1, num_heads=2, ffn_dim=128, transformer_output_dim_per_symbol=32, fourier_num_modes=8, wavelet_levels=1),"net_arch": dict(pi=[32,32], qf=[32,32])}
        )
    except ValueError as ve: logger.error(f"Trainer __main__ 測試因ValueError終止: {ve}", exc_info=True)
    except RuntimeError as rte: logger.error(f"Trainer __main__ 測試因RuntimeError終止: {rte}", exc_info=True)
    except Exception as e_main_train: logger.error(f"Trainer __main__ 測試過程中發生未知嚴重錯誤: {e_main_train}", exc_info=True)
    logger.info("trainer.py __main__ 測試執行完畢。")