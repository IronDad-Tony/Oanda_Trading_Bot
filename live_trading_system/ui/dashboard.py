import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from ..core.system_state import SystemState
from ..trading.position_manager import PositionManager
from ..database.database_manager import DatabaseManager

def display_instrument_status(instrument_monitor: Any):
    """Placeholder for displaying live instrument data."""
    st.header("Instrument Monitor")
    st.info("Live instrument status display is under development.")

# 將 streamlit_app_complete.py 中的分類邏輯移植過來
def get_categorized_instruments(state: SystemState) -> Dict[str, List[str]]:
    """
    獲取所有 OANDA 標的並進行分類。
    利用 SystemState 快取結果以避免重複 API 調用。
    """
    if state.categorized_instruments:
        return state.categorized_instruments

    iim = state.get_instrument_manager()
    all_symbols = iim.get_all_available_symbols()
    
    categorized = {
        'Major Pairs': [], 'Minor Pairs': [], 'Precious Metals': [], 'Indices': [],
        'Energy': [], 'Commodities': [], 'Crypto': [], 'Bonds': []
    }
    major_pairs = {'EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CHF', 'USD_CAD', 'NZD_USD'}
    
    # 關鍵字可以根據需要擴充
    index_keywords = ["SPX", "NAS", "US30", "UK100", "DE30", "JP225"]
    metal_keywords = ["XAU", "XAG", "GOLD", "SILVER"]
    energy_keywords = ["OIL", "BRENT", "NATGAS"]

    for sym in all_symbols:
        details = iim.get_details(sym)
        if not details:
            continue
        
        symbol = details.symbol
        display_name = details.display_name
        
        # 簡易分類邏輯
        if symbol in major_pairs:
            categorized['Major Pairs'].append(symbol)
        elif any(k in symbol for k in index_keywords):
            categorized['Indices'].append(symbol)
        elif any(k in symbol for k in metal_keywords):
            categorized['Precious Metals'].append(symbol)
        elif any(k in symbol for k in energy_keywords):
            categorized['Energy'].append(symbol)
        elif details.type == 'CURRENCY':
            categorized['Minor Pairs'].append(symbol)
        elif details.type == 'CFD':
             if 'BOND' in display_name.upper():
                 categorized['Bonds'].append(symbol)
             else:
                 categorized['Commodities'].append(symbol)
        elif details.type == 'CRYPTO':
            categorized['Crypto'].append(symbol)

    # 移除空的分類並快取結果
    state.categorized_instruments = {k: sorted(v) for k, v in categorized.items() if v}
    return state.categorized_instruments


def display_system_status(state: SystemState):
    """Displays the current status of the trading system."""
    st.header("System Status")
    status = "Running" if state.is_running else "Stopped"
    instruments = state.get_selected_instruments()
    instruments_str = ", ".join(instruments) if instruments else "Not Selected"
    model = state.get_current_model() or "Not Loaded"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("System Control", status)
    # 顯示多個標的，如果太長則截斷
    if len(instruments_str) > 30:
        instruments_str = f"{len(instruments)} selected"
    col2.metric("Active Instruments", instruments_str)
    col3.metric("Active Model", model)

def display_open_positions(position_manager: PositionManager):
    """Displays a table of all open positions."""
    st.header("Open Positions")
    positions = position_manager.get_all_positions()
    if not positions:
        st.info("No open positions.")
        return

    data = []
    for inst, pos in positions.items():
        data.append({
            "Instrument": pos.instrument,
            "Type": pos.position_type.upper(),
            "Units": pos.units,
            "Entry Price": f"{pos.entry_price:.5f}"
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

def display_trade_history(db_manager: DatabaseManager):
    """Displays a table of the most recent trade history from the database."""
    st.header("Recent Trade History")
    trade_history = db_manager.get_trade_history(limit=50)
    if not trade_history:
        st.info("No trade history found.")
        return

    df = pd.DataFrame(trade_history)
    # Reorder for better readability
    df = df[['timestamp', 'instrument', 'action', 'units', 'price', 'details']]
    st.dataframe(df, use_container_width=True)

def display_performance_metrics():
    """Displays key performance indicators (KPIs). Placeholder for now."""
    st.header("Performance Metrics")
    st.warning("Performance metrics calculation will be implemented in a future phase.")
    # Example of what could be here:
    # col1, col2, col3 = st.columns(3)
    # col1.metric("Total P/L", "$1,250.50")
    # col2.metric("Win Rate", "62%")
    # col3.metric("Sharpe Ratio", "1.8")

def create_control_panel(system_state: SystemState, start_func, stop_func):
    """Creates control buttons for the system."""
    st.sidebar.header("Control Panel")
    
    # --- System Control ---
    # 增加一個禁言狀態，防止在選擇過程中誤觸
    ui_enabled = not system_state.is_running

    if ui_enabled:
        if st.sidebar.button("▶️ Start System", disabled=not system_state.get_selected_instruments()):
            start_func()
            st.sidebar.success("System started!")
            st.rerun()
    else:
        if st.sidebar.button("⏹️ Stop System"):
            stop_func()
            st.sidebar.warning("System stopping...")
            st.rerun()

    # --- Instrument and Model Selection ---
    st.sidebar.subheader("Configuration")
    
    # 獲取分類好的標的
    categorized_instruments = get_categorized_instruments(system_state)
    
    # 使用 session state 來保存展開狀態
    if 'expanded_categories' not in st.session_state:
        st.session_state.expanded_categories = set()

    # 顯示分類選擇器
    selected_instruments = []
    for category, instruments in categorized_instruments.items():
        # 檢查是否有任何已選中的標的屬於此類別，以決定是否預設展開
        is_expanded = any(item in system_state.get_selected_instruments() for item in instruments)
        
        with st.sidebar.expander(f"{category} ({len(instruments)})", expanded=is_expanded):
            selections = st.multiselect(
                f"Select from {category}",
                options=instruments,
                default=[inst for inst in instruments if inst in system_state.get_selected_instruments()],
                key=f"multiselect_{category}",
                disabled=not ui_enabled
            )
            selected_instruments.extend(selections)

    # 檢查選擇是否有變化
    if set(selected_instruments) != set(system_state.get_selected_instruments()):
        system_state.set_selected_instruments(selected_instruments)
        st.rerun()

    # --- 動態模型選擇 ---
    num_selected = len(system_state.get_selected_instruments())
    
    # 動態讀取 weights 資料夾下所有模型，根據選擇的標的數量過濾可用模型
    import os
    import re
    weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'weights')
    available_models = []
    model_pattern = re.compile(r'sac_model_symbols(\d+)\.zip')
    if num_selected == 0:
        st.sidebar.warning("Please select at least one instrument.")
    else:
        for fname in os.listdir(weights_dir):
            match = model_pattern.match(fname)
            if match:
                max_symbols = int(match.group(1))
                if max_symbols >= num_selected:
                    available_models.append(fname)
        available_models.sort(key=lambda x: int(model_pattern.match(x).group(1)))

    # 確保當前模型在可用列表中，如果不在，則選擇第一個作為預設
    current_model = system_state.get_current_model()
    if not current_model or current_model not in available_models:
        current_model = available_models[0] if available_models else None
        if current_model:
            system_state.set_current_model(current_model)

    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=available_models,
        index=available_models.index(current_model) if current_model and available_models else 0,
        key="model_select",
        disabled=not ui_enabled or not available_models
    )
    
    if selected_model and selected_model != system_state.get_current_model():
        system_state.set_current_model(selected_model)
        st.rerun()
