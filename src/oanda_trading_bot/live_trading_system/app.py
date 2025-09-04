# live_trading_system/ui/app.py
"""
Streamlit UI for the Live Trading System.

Provides a professional dashboard to monitor and control the trading bot.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import threading
from datetime import datetime

# --- Import System Components ---
from oanda_trading_bot.live_trading_system.main import initialize_system, trading_loop
# from .core.system_state import SystemState # SystemState is accessed through components dict

# --- Placeholder Data and Functions ---
# These will be replaced by actual system calls.
def get_system_status():
    if st.session_state.components:
        return st.session_state.components["system_state"].get_status()
    return ('STOPPED', 'red')

def get_api_connection_status():
    if st.session_state.components:
        # A simple check. A more robust implementation could involve a heartbeat.
        return st.session_state.components["client"] is not None
    return False

def get_key_metrics():
    if st.session_state.components and get_system_status()[0] == 'RUNNING':
        summary = st.session_state.components["client"].get_account_summary()
        if summary and 'account' in summary:
            account_info = summary['account']
            return {
                'equity': float(account_info.get('equity', 0)),
                'pnl': float(account_info.get('pl', 0)),
                'margin_used': float(account_info.get('marginUsed', 0)),
                'open_positions': int(account_info.get('openPositionCount', 0))
            }
    # Return last known or default if not running
    return st.session_state.get('metrics', {
        'equity': 100000.00, 'pnl': 0.00, 'margin_used': 0.00, 'open_positions': 0
    })

def get_active_symbols():
    if st.session_state.components:
        pm = st.session_state.components["position_manager"]
        return list(pm.get_all_positions().keys())
    return []

def get_candlestick_data(symbol):
    if st.session_state.components:
        client = st.session_state.components["client"]
        candles = client.get_candles(symbol, count=100, granularity="S5")
        if candles:
            df = pd.DataFrame([{
                'time': pd.to_datetime(c['time']),
                'open': float(c['mid']['o']),
                'high': float(c['mid']['h']),
                'low': float(c['mid']['l']),
                'close': float(c['mid']['c']),
            } for c in candles])
            return df
    return pd.DataFrame()

def get_trade_history():
    if st.session_state.components:
        db_manager = st.session_state.components["db_manager"]
        history = db_manager.get_trade_history(limit=100)
        if history:
            return pd.DataFrame(history)
    return pd.DataFrame()

def get_position_cost_basis(symbol):
    if st.session_state.components:
        pm = st.session_state.components["position_manager"]
        pos = pm.get_position(symbol)
        return pos.entry_price if pos else None
    return None

def start_system_thread():
    # st.session_state.system_status = ('RUNNING', 'green') # Status is now handled by SystemState
    if st.session_state.components:
        st.session_state.trading_thread = threading.Thread(target=trading_loop, args=(st.session_state.components,), daemon=True)
        st.session_state.trading_thread.start()

def stop_system():
    # st.session_state.system_status = ('STOPPED', 'red') # Status is now handled by SystemState
    if st.session_state.components:
        st.session_state.components['system_state'].stop()
    if st.session_state.trading_thread:
        st.session_state.trading_thread.join(timeout=10)


# --- Page Configuration ---
st.set_page_config(
    page_title="Oanda 量化交易監控儀表板",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Initialize Session State ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.system_status = ('STOPPED', 'red') # ('RUNNING', 'green'), ('ERROR', 'orange')
    st.session_state.api_status = True # Placeholder
    st.session_state.metrics = {
        'equity': 100000.00,
        'pnl': 0.00,
        'margin_used': 0.00,
        'open_positions': 0
    }
    st.session_state.positions = {
        'EUR_USD': {'direction': 'LONG', 'units': 1000, 'entry_price': 1.0725, 'pnl': 12.50},
        'USD_JPY': {'direction': 'SHORT', 'units': 2000, 'entry_price': 157.80, 'pnl': -35.00}
    }
    st.session_state.orders = pd.DataFrame({
        'time': [datetime.now(), datetime.now() - pd.Timedelta(minutes=5)],
        'symbol': ['EUR_USD', 'USD_JPY'],
        'type': ['MARKET', 'MARKET'],
        'side': ['BUY', 'SELL'],
        'units': [1000, 2000],
        'price': [1.0725, 157.80],
        'status': ['FILLED', 'FILLED']
    })
    st.session_state.logs = ["[INFO] UI Initialized.", "[INFO] Awaiting system start command."]
    st.session_state.chart_data = {}
    st.session_state.trades = {
        'EUR_USD': pd.DataFrame({
            'time': [datetime.now() - pd.Timedelta(minutes=10)],
            'action': ['BUY_OPEN'],
            'price': [1.0725]
        })
    }
    st.session_state.components = None # To store initialized system components
    st.session_state.trading_thread = None


# --- UI Layout ---

# --- Header ---
header_cols = st.columns([3, 1, 1, 1])
with header_cols[0]:
    st.title("🤖 Oanda 量化交易監控儀表板")

with header_cols[1]:
    status, color = get_system_status()
    st.markdown(f"<div style='text-align:center; padding: 5px; border-radius: 5px; color: white; background-color: {color};'>系統狀態: {status}</div>", unsafe_allow_html=True)

with header_cols[2]:
    api_ok = get_api_connection_status()
    api_color = "green" if api_ok else "red"
    api_text = "正常" if api_ok else "斷線"
    st.markdown(f"<div style='text-align:center; padding: 5px; border-radius: 5px; color: white; background-color: {api_color};'>API 連線: {api_text}</div>", unsafe_allow_html=True)

with header_cols[3]:
    if get_system_status()[0] == 'STOPPED':
        if st.button("🚀 啟動系統", use_container_width=True):
            with st.spinner("正在初始化系統組件..."):
                st.session_state.components = initialize_system()
                if st.session_state.components:
                    start_system_thread()
                    st.success("系統啟動成功！")
                    st.rerun()
                else:
                    st.error("系統初始化失敗，請檢查日誌。")

    else:
        if st.button("🛑 停止系統", type="primary", use_container_width=True):
            with st.spinner("正在停止系統..."):
                stop_system()
                st.warning("系統已停止。")


st.divider()

# --- Key Metrics ---
metrics = get_key_metrics()
st.session_state.metrics = metrics # Cache the latest metrics
metric_cols = st.columns(4)
metric_cols[0].metric("帳戶淨值 (Equity)", f"${metrics['equity']:.2f}")
metric_cols[1].metric("當日盈虧 (P/L)", f"${metrics['pnl']:.2f}", delta=f"{metrics['pnl']:.2f}")
metric_cols[2].metric("已用保證金 (Margin)", f"${metrics['margin_used']:.2f}")
metric_cols[3].metric("持倉數量 (Positions)", metrics['open_positions'])

st.divider()

# --- Trading Activity Chart ---
st.subheader("交易活動圖表")
active_symbols = get_active_symbols()
if not active_symbols:
    st.info("目前沒有任何持倉或活躍的交易對。")
else:
    selected_symbol = st.selectbox("選擇要查看的交易對", options=active_symbols)

    if selected_symbol:
        chart_placeholder = st.empty()
        df = get_candlestick_data(selected_symbol)

        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name=selected_symbol))

            cost_basis = get_position_cost_basis(selected_symbol)
            if cost_basis:
                fig.add_hline(y=cost_basis, line_width=2, line_dash="dash", line_color="blue", annotation_text=f"成本價: {cost_basis}", annotation_position="bottom right")

            fig.update_layout(title=f"{selected_symbol} - 5秒 K線圖", xaxis_title="時間", yaxis_title="價格", xaxis_rangeslider_visible=False, height=500)
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"無法獲取 {selected_symbol} 的 K線圖數據。")


st.divider()

# --- Positions and Logs ---
tab_pos, tab_ord, tab_log = st.tabs(["當前持倉", "今日訂單", "系統日誌"])

with tab_pos:
    st.subheader("當前持倉")
    if st.session_state.components:
        pm = st.session_state.components["position_manager"]
        all_pos = pm.get_all_positions()
        if all_pos:
            pos_data = [{
                'direction': p.position_type.upper(),
                'units': p.units,
                'entry_price': p.entry_price
            } for inst, p in all_pos.items()]
            pos_df = pd.DataFrame(pos_data, index=all_pos.keys())
            st.dataframe(pos_df, use_container_width=True)
        else:
            st.info("目前沒有任何持倉。")
    else:
        st.info("系統未啟動，無法獲取持倉資訊。")

with tab_ord:
    st.subheader("最近 100 筆歷史訂單")
    trade_history_df = get_trade_history()
    if not trade_history_df.empty:
        st.dataframe(trade_history_df, use_container_width=True)
    else:
        st.info("沒有可用的交易歷史記錄。")

with tab_log:
    st.subheader("系統日誌")
    log_container = st.container(height=300)
    for log in st.session_state.logs:
        log_container.text(log)


# --- Auto-refresh loop ---
# This is a simple way to create a refresh loop in Streamlit
time.sleep(5)
st.rerun()
