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
# This section will be populated with imports from the live_trading_system
# For now, we use placeholder functions and data.
# from live_trading_system.main import initialize_system, trading_loop
# from live_trading_system.core.system_state import SystemState

# --- Placeholder Data and Functions ---
# These will be replaced by actual system calls.
def get_system_status():
    # ('RUNNING', 'green'), ('STOPPED', 'red'), ('ERROR', 'orange')
    return st.session_state.get('system_status', ('STOPPED', 'red'))

def get_api_connection_status():
    return st.session_state.get('api_status', True)

def get_key_metrics():
    # Simulating data updates
    if st.session_state.get('system_status', ('STOPPED', 'red'))[0] == 'RUNNING':
        st.session_state.metrics['equity'] *= 1.00001
        st.session_state.metrics['pnl'] += (np.random.rand() - 0.5) * 10
    return st.session_state.metrics

def get_active_symbols():
    return list(st.session_state.positions.keys())

def get_candlestick_data(symbol):
    # In a real scenario, this would fetch data from the system
    if 'chart_data' not in st.session_state or symbol not in st.session_state.chart_data:
        # Create initial random data
        base_price = 1.0700 if 'EUR' in symbol else 150.0
        dates = pd.to_datetime(pd.date_range(end=datetime.now(), periods=100, freq='5s'))
        df = pd.DataFrame({
            'time': dates,
            'open': base_price + (np.random.rand(100) - 0.5) * 0.001,
            'high': base_price + (np.random.rand(100) - 0.5) * 0.001,
            'low': base_price + (np.random.rand(100) - 0.5) * 0.001,
            'close': base_price + (np.random.rand(100) - 0.5) * 0.001,
        })
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.rand(100) * 0.0005
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.rand(100) * 0.0005
        st.session_state.chart_data[symbol] = df
    else:
        # Add a new candle
        df = st.session_state.chart_data[symbol]
        last_row = df.iloc[-1]
        new_time = last_row['time'] + pd.Timedelta(seconds=5)
        new_open = last_row['close']
        new_close = new_open + (np.random.rand() - 0.5) * 0.0002
        new_high = max(new_open, new_close) + np.random.rand() * 0.0001
        new_low = min(new_open, new_close) - np.random.rand() * 0.0001
        new_row = pd.DataFrame([{'time': new_time, 'open': new_open, 'high': new_high, 'low': new_low, 'close': new_close}])
        st.session_state.chart_data[symbol] = pd.concat([df, new_row], ignore_index=True).tail(100)

    return st.session_state.chart_data[symbol]


def get_trade_history(symbol):
    return st.session_state.trades.get(symbol, pd.DataFrame())

def get_position_cost_basis(symbol):
    pos = st.session_state.positions.get(symbol)
    return pos['entry_price'] if pos else None

def start_system_thread():
    st.session_state.system_status = ('RUNNING', 'green')
    # In real app:
    # st.session_state.trading_thread = threading.Thread(target=trading_loop, args=(st.session_state.components,))
    # st.session_state.trading_thread.start()

def stop_system():
    st.session_state.system_status = ('STOPPED', 'red')
    # In real app:
    # st.session_state.components['system_state'].stop()
    # st.session_state.trading_thread.join()


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
                # st.session_state.components = initialize_system()
                # if st.session_state.components:
                #     start_system_thread()
                #     st.success("系統啟動成功！")
                # else:
                #     st.error("系統初始化失敗，請檢查日誌。")
                start_system_thread() # Placeholder
                st.success("系統啟動成功！")

    else:
        if st.button("🛑 停止系統", type="primary", use_container_width=True):
            with st.spinner("正在停止系統..."):
                stop_system()
                st.warning("系統已停止。")


st.divider()

# --- Key Metrics ---
metrics = get_key_metrics()
metric_cols = st.columns(4)
metric_cols[0].metric("帳戶淨值 (Equity)", f"${metrics['equity']:.2f}")
metric_cols[1].metric("當日盈虧 (P/L)", f"${metrics['pnl']:.2f}", delta=f"{metrics['pnl']:.2f}")
metric_cols[2].metric("已用保證金 (Margin)", f"${metrics['margin_used']:.2f}")
metric_cols[3].metric("持倉數量 (Positions)", len(st.session_state.positions))

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

        # Create chart
        df = get_candlestick_data(selected_symbol)
        trades = get_trade_history(selected_symbol)
        cost_basis = get_position_cost_basis(selected_symbol)

        fig = go.Figure()

        # 1. Candlestick
        fig.add_trace(go.Candlestick(x=df['time'],
                        open=df['open'], high=df['high'],
                        low=df['low'], close=df['close'],
                        name=selected_symbol))

        # 2. Trade Markers
        buy_trades = trades[trades['action'].str.contains('BUY')]
        sell_trades = trades[trades['action'].str.contains('SELL')]
        close_trades = trades[trades['action'].str.contains('CLOSE')]

        fig.add_trace(go.Scatter(
            x=buy_trades['time'], y=buy_trades['price'],
            mode='markers', name='Buy',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
        fig.add_trace(go.Scatter(
            x=sell_trades['time'], y=sell_trades['price'],
            mode='markers', name='Sell',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))

        # 3. Cost Basis Line
        if cost_basis:
            fig.add_hline(y=cost_basis, line_width=2, line_dash="dash", line_color="blue",
                          annotation_text=f"成本價: {cost_basis}", annotation_position="bottom right")

        fig.update_layout(
            title=f"{selected_symbol} - 5秒 K線圖與交易點位",
            xaxis_title="時間",
            yaxis_title="價格",
            xaxis_rangeslider_visible=False,
            height=500
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)


st.divider()

# --- Positions and Logs ---
tab_pos, tab_ord, tab_log = st.tabs(["當前持倉", "今日訂單", "系統日誌"])

with tab_pos:
    st.subheader("當前持倉")
    st.dataframe(pd.DataFrame.from_dict(st.session_state.positions, orient='index'), use_container_width=True)

with tab_ord:
    st.subheader("今日訂單")
    st.dataframe(st.session_state.orders, use_container_width=True)

with tab_log:
    st.subheader("系統日誌")
    log_container = st.container(height=300)
    for log in st.session_state.logs:
        log_container.text(log)


# --- Auto-refresh loop ---
# This is a simple way to create a refresh loop in Streamlit
time.sleep(5)
st.rerun()