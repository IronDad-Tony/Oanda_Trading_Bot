import streamlit as st
import threading
import time
import sys
import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta

# Add project root to sys.path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from live_trading_system.main import initialize_system, trading_loop
from live_trading_system.ui import dashboard
from src.data_manager.instrument_info_manager import InstrumentInfoManager
from live_trading_system.core.oanda_client import OandaClient
from live_trading_system.trading.position_manager import PositionManager

def generate_candlestick_chart(candles, symbol):
    """生成帶有技術指標的K線圖表"""
    # 轉換為DataFrame
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    df = df[['time', 'mid']].copy()
    df[['o', 'h', 'l', 'c']] = pd.DataFrame(df['mid'].tolist(), index=df.index)
    
    # 創建基礎K線圖
    fig = go.Figure(data=[go.Candlestick(
        x=df['time'],
        open=df['o'],
        high=df['h'],
        low=df['l'],
        close=df['c'],
        name=symbol
    )])
    
    # 添加技術指標
    if st.session_state.get(f"macd_{symbol}", True):
        # 計算MACD (簡化版)
        df['ema12'] = df['c'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['c'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['macd'],
            name='MACD',
            line=dict(color='blue'),
            yaxis='y2'
        ))
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['signal'],
            name='Signal',
            line=dict(color='orange'),
            yaxis='y2'
        ))
    
    if st.session_state.get(f"rsi_{symbol}", True):
        # 計算RSI
        delta = df['c'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['rsi'],
            name='RSI',
            line=dict(color='purple'),
            yaxis='y3'
        ))
    
    # 設置圖表佈局
    fig.update_layout(
        title=f"{symbol} K線圖",
        xaxis_title="時間",
        yaxis_title="價格",
        yaxis2=dict(title="MACD", overlaying='y', side='right'),
        yaxis3=dict(title="RSI", overlaying='y', side='right', position=0.95),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    return fig

def get_categorized_symbols():
    """Fetch all OANDA symbols and categorize them"""
    iim = InstrumentInfoManager(force_refresh=False)
    all_symbols = iim.get_all_available_symbols()
    
    categorized = {
        'Major Pairs': [],
        'Minor Pairs': [],
        'Precious Metals': [],
        'Indices': [],
        'Energy': [],
        'Commodities': [],
        'Crypto': [],
        'Bonds': []
    }
    major_pairs = {
        'EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CHF', 'USD_CAD', 'NZD_USD'
    }
    
    for sym in all_symbols:
        details = iim.get_details(sym)
        if not details:
            continue
            
        symbol = details.symbol
        display = details.display_name if hasattr(details, 'display_name') else sym
        t = details.type.upper() if details.type else ''
        
        if symbol in major_pairs:
            categorized['Major Pairs'].append((symbol, display, t))
        elif t == 'CURRENCY' and '_' in symbol:
            base, quote = symbol.split('_')
            if not (base.startswith('XAU') or base.startswith('XAG')):
                categorized['Minor Pairs'].append((symbol, display, t))
        elif 'XAU' in symbol or 'XAG' in symbol or 'GOLD' in symbol or 'SILVER' in symbol:
            categorized['Precious Metals'].append((symbol, display, t))
        elif 'SPX' in symbol or 'NAS' in symbol or 'US30' in symbol or 'UK100' in symbol:
            categorized['Indices'].append((symbol, display, t))
        elif 'OIL' in symbol or 'NATGAS' in symbol:
            categorized['Energy'].append((symbol, display, t))
        elif 'CORN' in symbol or 'WHEAT' in symbol or 'SOYBN' in symbol:
            categorized['Commodities'].append((symbol, display, t))
        elif 'BTC' in symbol or 'ETH' in symbol or 'LTC' in symbol:
            categorized['Crypto'].append((symbol, display, t))
        else:
            categorized['Bonds'].append((symbol, display, t))
    
    return {k: v for k, v in categorized.items() if v}

def fetch_account_summary():
    """Fetch account summary from Oanda API"""
    oanda_client = OandaClient(
        api_key=os.getenv("OANDA_API_KEY"),
        account_id=os.getenv("OANDA_ACCOUNT_ID")
    )
    return oanda_client.get_account_summary()

def trading_thread_target(components):
    """Target function for the trading logic thread."""
    trading_loop(components)

def start_trading_system():
    """Initializes and starts the trading system in a background thread."""
    if 'components' not in st.session_state:
        st.session_state.components = initialize_system()

    if st.session_state.components:
        components = st.session_state.components
        system_state = components['system_state']

        if not system_state.is_running:
            system_state.set_running(True)
            thread = threading.Thread(
                target=trading_thread_target,
                args=(components,),
                daemon=True
            )
            thread.start()
            st.session_state.trading_thread = thread
            st.success("Trading system started.")
            try:
                st.experimental_rerun()
            except st.errors.RerunException:
                pass
        else:
            st.warning("Trading system is already running.")
    else:
        st.error("System components not initialized. Cannot start trading logic.")

def stop_and_close_all():
    """Stop trading and close all positions"""
    if 'components' in st.session_state:
        system_state = st.session_state.components['system_state']
        position_manager = st.session_state.components['position_manager']
        
        if system_state.is_running:
            system_state.set_running(False)
            
        # Close all open positions
        position_manager.close_all_positions()
        st.success("All positions closed and trading stopped.")
    else:
        st.warning("System components not initialized.")
        
def pause_trading():
    """Pause trading without closing positions"""
    if 'components' in st.session_state:
        system_state = st.session_state.components['system_state']
        if system_state.is_running:
            system_state.set_running(False)
            st.success("Trading paused. Positions remain open.")
        else:
            st.warning("Trading is not running.")
    else:
        st.warning("System components not initialized.")
        
def resume_trading():
    """Resume paused trading"""
    if 'components' in st.session_state:
        system_state = st.session_state.components['system_state']
        if not system_state.is_running:
            system_state.set_running(True)
            st.success("Trading resumed.")
        else:
            st.warning("Trading is already running.")
    else:
        st.warning("System components not initialized.")

def stop_trading_system():
    """Stops the trading logic."""
    if 'components' in st.session_state:
        system_state = st.session_state.components['system_state']
        if system_state.is_running:
            system_state.set_running(False)
            if 'trading_thread' in st.session_state:
                st.session_state.trading_thread.join(timeout=10)
            st.success("Trading system stopped.")
            try:
                st.experimental_rerun()
            except st.errors.RerunException:
                pass
        else:
            st.warning("Trading system is not running.")

def run_app():
    """
    The main function to build and run the Streamlit UI.
    """
    st.set_page_config(page_title="Oanda Trading System", layout="wide")
    st.title("Oanda Real-Time Trading Dashboard")

    if 'components' not in st.session_state:
        with st.spinner('Initializing trading system components...'):
            st.session_state.components = initialize_system()

        if st.session_state.components is None:
            st.error("Failed to initialize trading system. Check logs for details.")
            return

    components = st.session_state.components
    system_state = components['system_state']
    position_manager = components['position_manager']
    db_manager = components['db_manager']
    instrument_monitor = components['instrument_monitor']
    
    # 交易控制面板
    with st.sidebar:
        # 交易標的選擇器（動態計數功能，來源：streamlit_app_complete.py:1770-1790）
        st.subheader("交易標的")
        categorized_symbols = get_categorized_symbols()
        selected_symbols = []
        for category, symbols in categorized_symbols.items():
            with st.expander(f"{category} ({len(symbols)})", expanded=(category=="Major Pairs")):
                for sym, display, _ in symbols:
                    if st.checkbox(f"{display} ({sym})", key=f"sym_{sym}"):
                        selected_symbols.append(sym)
        
        # 交易參數設定（雙列布局）
        st.subheader("交易參數")
        col1, col2 = st.columns(2)
        with col1:
            risk_percentage = st.slider("風險係數 (%)", 0.1, 10.0, 1.0, step=0.1)
        with col2:
            stop_loss_multiplier = st.slider("止損係數", 1.0, 5.0, 2.0, step=0.1)
        
        # 風險管理參數（三層風險控制）
        with st.expander("▣ 風險管理參數", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.per_trade_risk = st.slider("單筆交易風險 (%)", 0.1, 5.0, 1.0, step=0.1)
                st.session_state.total_exposure = st.slider("總部位風險 (%)", 1.0, 30.0, 10.0, step=0.5)
            with col2:
                st.session_state.atr_multiplier = st.slider("ATR止損倍數", 1.0, 5.0, 2.0, step=0.1)
                
                # 計算最大持倉量
                if 'account_summary' in st.session_state and st.session_state.account_summary:
                    balance = float(st.session_state.account_summary['balance'])
                    position_size = balance * st.session_state.per_trade_risk / 100
                    st.metric("最大持倉量", f"{position_size:.极2f} {st.session_state.account_summary['currency']}")
        
        # 開始交易按鈕（狀態感知按鈕）
        if st.button("開始交易", type="primary", use_container_width=True):
            start_trading_system()
        
        # 緊急控制按鈕（雙列布局）
        st.subheader("緊急控制")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("暫停交易並平倉", type="secondary", help="暫停交易並關閉所有持倉", use_container_width=True):
                stop_and_close_all()
        with col2:
            if st.button("暫停交易", type="secondary", help="暫停交易但保持持倉不變", use_container_width=True):
                pause_trading()
        if st.button("恢復交易", type="secondary", help="恢復已暫停的交易", use_container_width=True):
            resume_trading()
        
        # 智能刷新系統（來源：streamlit_app_complete.py:1934-1965）
        st.subheader("智能刷新系統")
        auto_refresh = st.checkbox("啟用自動刷新", value=st.session_state.get('auto_refresh', True))
        st.session_state.auto极refresh = auto_refresh
        if auto_refresh:
            refresh_interval = st.slider("刷新間隔（秒）", 1, 30, st.session_state.get('refresh_interval', 5))
            st.session_state.refresh_interval = refresh_interval
        
        # 顯示設置選項（來源：streamlit_app_complete.py:1939-1954）
        st.subheader("顯示設置")
        chart_display_mode = st.selectbox(
            "圖表顯示模式",
            options=['full', 'lite', 'minimal'],
            index=['full', 'lite', 'minimal'].index(st.session_state.get('chart_display_mode', 'full'))
        )
        st.session_state.chart_display_mode = chart_display_mode

    # 顯示帳戶資訊
    st.subheader("帳戶資訊")
    try:
        st.session_state.account_summary = fetch_account_summary()
        if st.session_state.account_summary:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("帳戶餘額", f"{st.session_state.account_summary['balance']} {st.session_state.account_summary['currency']}")
            with col2:
                st.metric("可用保證金", f"{st.session_state.account_summary['marginAvailable']} {st.session_state.account_summary['currency']}")
            
            # 顯示持倉資訊
            positions = position_manager.get_all_positions()
            if positions:
                st.subheader("目前持倉")
                for pos in positions:
                    units_long = int(pos['long']['units']) if 'long' in pos and 'units' in pos['long'] else 0
                    units_short = int(pos['short']['units']) if 'short' in pos and 'units' in pos['short'] else 0
                    
                    if units_long != 0 or units_short != 0:
                        st.write(f"{pos['instrument']}: {units_long}單位 (多) / {units_short}單位 (空)")
            else:
                st.info("目前沒有持倉")
    except Exception as e:
        st.error(f"獲取帳戶資訊失敗: {str(e)}")
    
    # 動態模型選擇
    st.subheader("模型選擇")
    num_selected = len(selected_symbols)
    available_models = []
    model_pattern = re.compile(r'model_(\d+)\.pkl')
    
    # 掃描weights目錄下的模型文件
    weights_dir = os.path.join(project_root, 'weights')
    if os.path.exists(weights_dir):
        for fname in os.listdir(weights_dir):
            match = model_pattern.match(fname)
            if match:
                max_symbols = int(match.group(1))
                if max_symbols >= num_selected:
                    available_models.append(fname)
    
    if available_models:
        st.session_state.selected_model = st.selectbox("選擇交易模型", options=available_models, index=0)
        st.info(f"已選擇模型: {st.session_state.selected_model} (支持最多 {model_pattern.match(st.session_state.selected_model).group(1)} 個品種)")
    else:
        st.warning("找不到匹配的模型，請選擇更少的品種或添加新模型")
    
    # 主儀表板
    dashboard.create_control_panel(system_state, start_trading_system, stop_trading_system)
    
    # 多品種圖表系統
    if selected_symbols:
        st.subheader("品種圖表")
        tabs = st.tabs([f"📊 {sym}" for sym in selected_symbols])
        for i, symbol in enumerate(selected_symbols):
            with tabs[i]:
                # 獲取K線數據
                oanda_client = OandaClient(
                    api_key=os.getenv("OANDA_API_KEY"),
                    account_id=os.getenv("OANDA_ACCOUNT_ID")
                )
                candles = oanda_client.get_candles(symbol, count=100, granularity="M15")
                
                if candles:
                    # 生成K線圖
                    fig = generate_candlestick_chart(candles, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 技術指標控制
                    with st.expander("技術指標設置"):
                        col1, col2 = st.columns(2)
                        with col1:
                            show_macd = st.checkbox("顯示MACD", True, key=f"macd_{symbol}")
                        with col2:
                            show_rsi = st.checkbox("顯示RSI", True, key=f"rsi_{symbol}")
                else:
                    st.warning(f"無法獲取 {symbol} 的K線數據")
    else:
        st.info("請在左側選擇交易品種以顯示圖表")
    
    col1, col2 = st.columns(2)
    with col1:
        dashboard.display_system_status(system_state)
    with col2:
        dashboard.display_instrument_status(instrument_monitor)
    
    # 淨值曲線和風險分析
    st.subheader("帳戶淨值曲線")
    try:
        # 獲取歷史淨值數據
        oanda_client = OandaClient(
            api_key=os.getenv("OANDA_API_KEY"),
            account_id=os.getenv("OANDA_ACCOUNT_ID")
        )
        equity_history = oanda_client.get_equity_history(period="7D")
        
        if equity_history and 'changes' in equity_history:
            df_equity = pd.DataFrame(equity_history['changes'])
            df_equity['time'] = pd.to_datetime(df_equity['time'])
            df_equity = df_equity.set_index('time')
            
            # 繪製淨值曲線
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=df_equity.index,
                y=df_equity['balance'],
                name="帳戶淨值",
                line=dict(color='green')
            ))
            
            # 添加持倉價值
            if 'openPositionValue' in df_equity:
                fig_equity.add_trace(go.Scatter(
                    x=df_equity.index,
                    y=df_equity['openPositionValue'],
                    name="持倉價值",
                    line=dict(color='blue')
                ))
            
            fig_equity.update_layout(
                title="7日帳戶淨值變化",
                xaxis_title="時間",
                yaxis_title="金額",
                height=300
            )
            st.plotly_chart(fig_equity, use_container_width=True)
        else:
            st.warning("無法獲取淨值歷史數據")
    except Exception as e:
        st.error(f"淨值曲線生成失敗: {str(e)}")
    
    # 風險敞口可視化
    st.subheader("風險敞口分析")
    positions = position_manager.get_all_positions()
    if positions:
        exposure_data = []
        for pos in positions:
            instrument = pos['instrument']
            long_units = int(pos['long']['units']) if 'long' in pos and 'units' in pos['long'] else 0
            short_units = int(pos['short']['units']) if 'short' in pos and 'units' in pos['short'] else 0
            exposure = abs(long_units) + abs(short_units)
            exposure_data.append({"Instrument": instrument, "Exposure": exposure})
        
        df_exposure = pd.DataFrame(exposure_data)
        
        # 繪製風險敞口餅圖
        fig_pie = go.Figure(data=[go.Pie(
            labels=df_exposure['Instrument'],
            values=df_exposure['Exposure'],
            hole=0.3
        )])
        fig_pie.update_layout(
            title="持倉風險分布",
            height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("目前沒有持倉，無風險敞口")
    
    st.divider()
    dashboard.display_open_positions(position_manager)
    st.divider()
    dashboard.display_trade_history(db_manager)
    st.divider()

    if system_state.is_running:
        time.sleep(5)
        try:
            st.experimental_rerun()
        except st.errors.RerunException:
            pass

def start_streamlit_app(components):
    """
    This function is designed to be called from main.py to launch the UI.
    """
    if 'components' not in st.session_state:
        st.session_state.components = components
    
    run_app()

if __name__ == '__main__':
    run_app()