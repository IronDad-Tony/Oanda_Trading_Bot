# streamlit_app.py
"""
Streamlit GUI 應用程序，用於OANDA通用交易模型訓練系統。
"""

from typing import List, Optional
import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
import threading # 用於在後台運行訓練任務
import time

# --- 系統路徑設置和模組導入 ---
# 確保能找到我們自己創建的模組
# 將專案根目錄下的 'src' 目錄添加到 Python 模組搜索路徑
# 這樣 'common', 'data_manager', 'trainer' 等模組才能被正確導入
project_root = Path(__file__).resolve().parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from common.config import MAX_SYMBOLS_ALLOWED
from environment.trading_env import DEFAULT_INITIAL_CAPITAL, format_datetime_for_oanda # 用於模擬進度和更新UI


try:
    from common.logger_setup import logger # 使用我們配置好的logger
    from common.config import (
        DEFAULT_SYMBOLS, GRANULARITY, ACCOUNT_CURRENCY,
        DEFAULT_TRAIN_START_ISO, DEFAULT_TRAIN_END_ISO,
        DEFAULT_EVAL_START_ISO, DEFAULT_EVAL_END_ISO,
        TRAINER_DEFAULT_TOTAL_TIMESTEPS, OANDA_API_KEY, OANDA_ACCOUNT_ID,
        LOGS_DIR
    )
    from data_manager.instrument_info_manager import InstrumentInfoManager # 用於獲取可交易品種
    from trainer.trainer import run_training_session # 我們的核心訓練函數
    # from trainer.callbacks import UniversalCheckpointCallback # Callback在trainer內部使用
except ImportError as e:
    # 如果在Streamlit環境中直接運行此文件，有時導入會出問題
    # 嘗試更明確地添加路徑
    st.error(f"導入模組失敗: {e}. 請確保專案結構正確，並且所有依賴已安裝。")
    st.write("當前Python搜索路徑:", sys.path)
    # 如果真的無法導入，Streamlit應用無法繼續
    # 在這種情況下，確保從 Oanda_Trading_Bot 根目錄運行 streamlit run streamlit_app.py
    sys.exit(1)


# --- Streamlit 頁面配置 ---
st.set_page_config(
    page_title="通用交易模型訓練平台",
    page_icon="🤖",
    layout="wide", # "centered" 或 "wide"
    initial_sidebar_state="expanded" # "auto", "expanded", "collapsed"
)

# --- 全局狀態管理 (使用 Streamlit Session State) ---
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'training_thread' not in st.session_state:
    st.session_state.training_thread = None
if 'training_log_messages' not in st.session_state:
    st.session_state.training_log_messages = [] # 存儲來自訓練過程的日誌/狀態信息
if 'stop_training_requested' not in st.session_state:
    st.session_state.stop_training_requested = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'total_steps_for_run' not in st.session_state:
    st.session_state.total_steps_for_run = TRAINER_DEFAULT_TOTAL_TIMESTEPS
if 'estimated_eta_str' not in st.session_state:
    st.session_state.estimated_eta_str = "N/A"
if 'steps_per_second' not in st.session_state:
    st.session_state.steps_per_second = 0.0
if 'start_train_time' not in st.session_state:
    st.session_state.start_train_time = 0.0

# --- 輔助函數 ---
@st.cache_resource # 緩存InstrumentInfoManager實例以避免重複API調用
def get_instrument_manager():
    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
        st.error("OANDA API金鑰或賬戶ID未在.env文件中配置！無法獲取交易品種。")
        return None
    try:
        manager = InstrumentInfoManager(force_refresh=False) # 首次加載或緩存過期時會刷新
        return manager
    except Exception as e:
        st.error(f"初始化InstrumentInfoManager失敗: {e}")
        return None

def get_available_symbols_options(manager: Optional[InstrumentInfoManager]) -> List[str]:
    if manager:
        try:
            symbols = manager.get_all_available_symbols()
            return sorted(list(set(symbols))) # 去重並排序
        except Exception as e:
            st.warning(f"獲取可用交易品種列表失敗: {e}")
            return DEFAULT_SYMBOLS # 返回默認值
    return DEFAULT_SYMBOLS

# --- UI 佈局 ---
st.title("🤖 通用交易模型訓練平台")
st.markdown("---")

# --- 側邊欄：參數配置 ---
with st.sidebar:
    st.header("🛠️ 訓練配置")

    # 1. 選擇交易對象
    st.subheader("1. 選擇交易對象")
    instrument_manager = get_instrument_manager()
    available_symbols = get_available_symbols_options(instrument_manager)
    
    selected_symbols_for_trading = st.multiselect(
        "選擇要訓練模型進行交易的品種 (核心品種):",
        options=available_symbols,
        default=[s for s in ["EUR_USD", "USD_JPY"] if s in available_symbols] # 默認選幾個常見的
    )
    st.caption(f"提示: 您最多可以選擇 {MAX_SYMBOLS_ALLOWED} 個核心交易品種。必要的匯率轉換對將自動處理。")

    # 2. 數據時間範圍
    st.subheader("2. 數據時間範圍")
    # Streamlit 的 date_input 返回 datetime.date 對象
    # 我們需要將其轉換為 datetime.datetime 並設置時間和時區
    default_start = datetime.strptime(DEFAULT_TRAIN_START_ISO.split("T")[0], "%Y-%m-%d").date()
    default_end_train = datetime.strptime(DEFAULT_TRAIN_END_ISO.split("T")[0], "%Y-%m-%d").date()
    default_start_eval = datetime.strptime(DEFAULT_EVAL_START_ISO.split("T")[0], "%Y-%m-%d").date()
    default_end_eval = datetime.strptime(DEFAULT_EVAL_END_ISO.split("T")[0], "%Y-%m-%d").date()

    train_start_date = st.date_input("訓練數據開始日期:", value=default_start, min_value=date(2000,1,1), max_value=datetime.now().date() - timedelta(days=1))
    train_end_date = st.date_input("訓練數據結束日期:", value=default_end_train, min_value=train_start_date, max_value=datetime.now().date())
    
    st.markdown("---")
    eval_start_date = st.date_input("評估數據開始日期:", value=default_start_eval, min_value=train_end_date + timedelta(days=1), max_value=datetime.now().date())
    eval_end_date = st.date_input("評估數據結束日期:", value=default_end_eval, min_value=eval_start_date, max_value=datetime.now().date())

    st.caption(f"數據粒度固定為: {GRANULARITY}")

    # 3. 訓練參數
    st.subheader("3. 訓練參數")
    total_timesteps_train = st.number_input("總訓練步數:", min_value=1000, value=TRAINER_DEFAULT_TOTAL_TIMESTEPS, step=10000, format="%d")
    initial_capital_train = st.number_input(f"初始資金 ({ACCOUNT_CURRENCY}):", min_value=1000.0, value=float(DEFAULT_INITIAL_CAPITAL), step=1000.0, format="%.2f")
    
    model_load_path_input = st.text_input("從指定路徑加載模型 (可選, 留空則嘗試加載latest或新建):", placeholder="例如: weights/sac_universal_model_xxxx/sac_universal_model_latest.zip")
    force_reload_data_checkbox = st.checkbox("強制重新加載和預處理數據集 (mmap文件)", value=False)

    # 存儲總步數以便進度條使用
    st.session_state.total_steps_for_run = total_timesteps_train

# --- 主區域：訓練控制和監控 ---
main_area = st.container()

with main_area:
    col1, col2 = st.columns([3, 1]) # 比例

    with col1:
        st.subheader("🚀 訓練控制")

    with col2:
        start_button_disabled = st.session_state.training_in_progress
        start_button_text = "🚀 正在訓練中..." if st.session_state.training_in_progress else "開始訓練"
        if st.button(start_button_text, disabled=start_button_disabled, type="primary", use_container_width=True):
            if not selected_symbols_for_trading:
                st.error("請至少選擇一個核心交易品種！")
            elif train_end_date < train_start_date:
                st.error("訓練結束日期必須在開始日期之後！")
            elif eval_end_date < eval_start_date:
                st.error("評估結束日期必須在開始日期之後！")
            elif eval_start_date <= train_end_date:
                st.error("評估數據時間段必須在訓練數據時間段之後且不重疊！")
            else:
                st.session_state.training_in_progress = True
                st.session_state.stop_training_requested = False
                st.session_state.training_log_messages = ["訓練任務已啟動..."]
                st.session_state.current_step = 0
                st.session_state.start_train_time = time.time()


                # 準備傳遞給訓練函數的參數
                train_start_iso_str = format_datetime_for_oanda(datetime.combine(train_start_date, datetime.min.time()))
                train_end_iso_str = format_datetime_for_oanda(datetime.combine(train_end_date, datetime.max.time().replace(microsecond=0)))
                eval_start_iso_str = format_datetime_for_oanda(datetime.combine(eval_start_date, datetime.min.time()))
                eval_end_iso_str = format_datetime_for_oanda(datetime.combine(eval_end_date, datetime.max.time().replace(microsecond=0)))
                
                # 自動確定 all_symbols_for_data
                all_symbols_needed = list(set(selected_symbols_for_trading))
                if ACCOUNT_CURRENCY != "USD":
                    aud_usd_pair = "AUD_USD" # 或 USD_AUD，假設我們下載AUD_USD
                    if aud_usd_pair not in all_symbols_needed: all_symbols_needed.append(aud_usd_pair)
                for sym_trade in selected_symbols_for_trading:
                    parts = sym_trade.split("_")
                    if len(parts) == 2:
                        quote_c = parts[1]
                        if quote_c != "USD" and quote_c != ACCOUNT_CURRENCY:
                            needed_usd_pair = f"{quote_c}_USD"
                            if needed_usd_pair not in all_symbols_needed and f"USD_{quote_c}" not in all_symbols_needed:
                                all_symbols_needed.append(needed_usd_pair)
                all_symbols_needed = sorted(list(set(all_symbols_needed)))


                # 創建UI組件的引用，以便在訓練線程中更新
                # 狀態文本和進度條需要在主循環中創建和更新
                # 這裡我們先啟動訓練線程

                logger.info(f"Streamlit UI: 準備啟動訓練線程...")
                training_args = {
                    "symbols_to_trade": selected_symbols_for_trading,
                    "all_symbols_for_data": all_symbols_needed,
                    "train_start_iso": train_start_iso_str,
                    "train_end_iso": train_end_iso_str,
                    "eval_start_iso": eval_start_iso_str,
                    "eval_end_iso": eval_end_iso_str,
                    "granularity": GRANULARITY,
                    "total_timesteps": total_timesteps_train,
                    "initial_capital": initial_capital_train,
                    "load_model_path": Path(model_load_path_input) if model_load_path_input else None,
                    "force_dataset_reload": force_reload_data_checkbox,
                    # streamlit_status_text 和 streamlit_progress_bar 將在主循環中獲取引用
                }
                
                # 創建一個函數來運行訓練，以便可以在線程中調用
                # 並允許它修改 session_state 中的日誌
                def training_thread_func(args_dict, st_log_list_ref):
                    try:
                        # 在線程內部獲取UI組件的引用是不安全的
                        # 我們通過回調或隊列來更新UI
                        # 這裡我們先簡單地將日誌添加到 session_state.training_log_messages
                        def ui_status_update(message, level="info"):
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            st_log_list_ref.append(f"[{timestamp} - {level.upper()}] {message}")
                            # 為了避免過多日誌，可以限制長度
                            if len(st_log_list_ref) > 200:
                                st_log_list_ref.pop(0)

                        args_dict["streamlit_status_text"] = type('obj', (object,), {'info': lambda m: ui_status_update(m, 'info'), 'warning': lambda m: ui_status_update(m, 'warning'), 'error': lambda m: ui_status_update(m, 'error'), 'success': lambda m: ui_status_update(m, 'success')})()
                        # progress_bar 需要一個 set_progress 方法
                        # args_dict["streamlit_progress_bar"] = type('obj', (object,), {'progress': lambda p: st.session_state.update({'current_progress_val_from_thread':p})  })()


                        run_training_session(**args_dict) # type: ignore
                        st_log_list_ref.append("訓練線程執行完畢。")
                    except Exception as e_thread:
                        logger.error(f"訓練線程發生嚴重錯誤: {e_thread}", exc_info=True)
                        st_log_list_ref.append(f"訓練線程錯誤: {e_thread}")
                    finally:
                        st.session_state.training_in_progress = False # 訓練結束
                        logger.info("訓練線程結束，設置 training_in_progress = False")


                st.session_state.training_thread = threading.Thread(
                    target=training_thread_func,
                    args=(training_args, st.session_state.training_log_messages), # 將列表引用傳入
                    daemon=True # 確保主程序退出時線程也退出
                )
                st.session_state.training_thread.start()
                st.rerun() # 重新運行Streamlit腳本以更新按鈕狀態

        # 停止按鈕 (只有在訓練進行中才顯示)
        if st.session_state.training_in_progress:
            if st.button("⏹️ 停止訓練", type="secondary", use_container_width=True):
                st.session_state.stop_training_requested = True
                st.warning("已請求停止訓練，請等待當前回調完成後安全退出並保存模型...")
                # 實際的停止邏輯會在 UniversalCheckpointCallback 中通過 self.interrupted 實現
                # 或者 trainer.py 的主訓練循環中檢查這個 st.session_state.stop_training_requested
                # 我們需要在UniversalCheckpointCallback中能夠訪問到這個session_state，這比較困難
                # 一個簡單的方法是，如果UniversalCheckpointCallback檢測到 self.interrupted (Ctrl+C),
                # 它會保存並返回False。Streamlit的停止按鈕可以嘗試更優雅地通知訓練循環。
                # 目前先依賴Ctrl+C或訓練自然結束。
                # TODO: 實現更優雅的UI停止機制，可能需要修改回調或trainer。
                st.rerun()


    st.markdown("---")
    st.subheader("📊 訓練監控")
    
    # 實時進度條和狀態文本
    progress_bar_ui = st.progress(0)
    status_text_ui = st.empty() # 用於顯示狀態信息和下載進度
    eta_text_ui = st.empty()

    # 實時日誌區域
    st.text_area("訓練日誌/狀態:", value="\n".join(st.session_state.training_log_messages), height=200, key="log_display_area")

    # 繪圖區域 (TensorBoard鏈接 和 未來的實時圖表)
    # TODO: 當訓練開始後，這裡應該顯示TensorBoard的鏈接或嵌入TensorBoard（如果可行）
    # TODO: 或者使用Plotly/Matplotlib繪製從回調中收集的關鍵指標
    # 例如，可以創建佔位符 st.empty() 然後在訓練循環中用 st.line_chart 更新
    
    # 模擬進度更新 (在真實訓練中，這部分數據來自回調或訓練循環)
    if st.session_state.training_in_progress:
        while st.session_state.training_thread and st.session_state.training_thread.is_alive():
            # 從回調中獲取 current_step, total_steps, eta, steps_per_sec
            # 這裡我們用 session_state 中的值來模擬
            # 在 UniversalCheckpointCallback 的 _on_step 中，可以通過某種方式更新這些session_state值
            # 例如，將 session_state 作為參數傳遞給回調（不推薦），或者回調寫入一個共享隊列，UI讀取。
            # 最簡單的方式是，SB3的logger會輸出到TensorBoard，Streamlit可以展示TensorBoard的鏈接。
            # 或者，讓 trainer.py 的 run_training_session 定期將進度寫入一個共享的狀態（例如一個文件或隊列）
            
            # --- 為了演示，這裡我們只模擬進度 ---
            # 在真實情況下，這些值應該由訓練線程通過某種機制（如隊列、文件、或Streamlit的session_state回調）更新
            # current_progress_val = st.session_state.current_step # 這個應該由訓練線程更新
            # total_steps_val = st.session_state.total_steps_for_run
            # eta_str_val = st.session_state.estimated_eta_str
            # sps_val = st.session_state.steps_per_second

            # 暫時使用一個簡單的計時器來模擬進度更新，直到我們有真正的回調機制
            # status_text_ui.info(f"訓練進行中... {st.session_state.training_log_messages[-1] if st.session_state.training_log_messages else ''}")
            # progress_percentage = (current_progress_val / total_steps_val) if total_steps_val > 0 else 0
            # progress_bar_ui.progress(progress_percentage)
            # eta_text_ui.text(f"進度: {current_progress_val}/{total_steps_val} ({progress_percentage*100:.1f}%) | {sps_val:.1f} steps/s | ETA: {eta_str_val}")
            
            # 為了讓UI保持響應，我們需要定期 rerun
            time.sleep(1) # 每秒刷新一次UI日誌和進度（如果進度有更新）
            st.rerun() # 這會重新執行整個腳本，但session_state會保留

        # 當訓練線程結束後
        if not st.session_state.training_thread.is_alive() and st.session_state.training_in_progress:
             st.session_state.training_in_progress = False # 確保標記結束
             st.success("訓練已結束！")
             st.rerun() # 刷新UI狀態


# --- 應用程序入口 ---
if __name__ == "__main__":
    # 檢查OANDA API金鑰是否已配置
    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
        st.error("嚴重錯誤: OANDA_API_KEY 或 OANDA_ACCOUNT_ID 未在 .env 文件中配置。應用程序無法啟動。")
        st.stop()
    
    # 這裡可以添加一些應用啟動時的日誌
    logger.info("Streamlit 應用程序啟動。")
    # (主UI佈局代碼已在上面)