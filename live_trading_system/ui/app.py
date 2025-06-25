import streamlit as st
import threading
import time
import sys
import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone, timedelta

# 添加项目根目录到系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from live_trading_system.main import initialize_system, trading_loop
from src.data_manager.instrument_info_manager import InstrumentInfoManager
from live_trading_system.core.oanda_client import OandaClient
from live_trading_system.trading.position_manager import PositionManager
from live_trading_system.trading.order_manager import OrderManager

# ====================== 工具函数 ======================
def generate_candlestick_chart(candles, symbol, theme="light"):
    """生成带有技术指标的K线图表"""
    # 转换为DataFrame
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    df = df[['time', 'mid']].copy()
    df[['o', 'h', 'l', 'c']] = pd.DataFrame(df['mid'].tolist(), index=df.index)
    
    # 创建基础K线图
    fig = go.Figure(data=[go.Candlestick(
        x=df['time'],
        open=df['o'],
        high=df['h'],
        low=df['l'],
        close=df['c'],
        name=symbol,
        increasing_line_color='#2E7D32' if theme == "light" else '#81C784',
        decreasing_line_color='#C62828' if theme == "light" else '#E57373'
    )])
    
    # 添加技术指标
    if st.session_state.get(f"macd_{symbol}", True):
        # 计算MACD
        df['ema12'] = df['c'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['c'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['macd'],
            name='MACD',
            line=dict(color='#2196F3'),
            yaxis='y2'
        ))
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['signal'],
            name='Signal',
            line=dict(color='#FF9800'),
            yaxis='y2'
        ))
    
    if st.session_state.get(f"rsi_{symbol}", True):
        # 计算RSI
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
            line=dict(color='#9C27B0'),
            yaxis='y3'
        ))
    
    # 设置图表布局
    fig.update_layout(
        title=f"{symbol} K线图",
        xaxis_title="时间",
        yaxis_title="价格",
        yaxis2=dict(title="MACD", overlaying='y', side='right'),
        yaxis3=dict(title="RSI", overlaying='y', side='right', position=0.95),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        template="plotly_dark" if theme == "dark" else "plotly_white",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def get_categorized_symbols():
    """获取所有OANDA品种并分类"""
    iim = InstrumentInfoManager(force_refresh=False)
    all_symbols = iim.get_all_available_symbols()
    
    categorized = {
        '主要货币对': [],
        '次要货币对': [],
        '贵金属': [],
        '指数': [],
        '能源': [],
        '大宗商品': [],
        '加密货币': [],
        '债券': []
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
            categorized['主要货币对'].append((symbol, display, t))
        elif t == 'CURRENCY' and '_' in symbol:
            base, quote = symbol.split('_')
            if not (base.startswith('XAU') or base.startswith('XAG')):
                categorized['次要货币对'].append((symbol, display, t))
        elif 'XAU' in symbol or 'XAG' in symbol or 'GOLD' in symbol or 'SILVER' in symbol:
            categorized['贵金属'].append((symbol, display, t))
        elif 'SPX' in symbol or 'NAS' in symbol or 'US30' in symbol or 'UK100' in symbol:
            categorized['指数'].append((symbol, display, t))
        elif 'OIL' in symbol or 'NATGAS' in symbol:
            categorized['能源'].append((symbol, display, t))
        elif 'CORN' in symbol or 'WHEAT' in symbol or 'SOYBN' in symbol:
            categorized['大宗商品'].append((symbol, display, t))
        elif 'BTC' in symbol or 'ETH' in symbol or 'LTC' in symbol:
            categorized['加密货币'].append((symbol, display, t))
        else:
            categorized['债券'].append((symbol, display, t))
    
    return {k: v for k, v in categorized.items() if v}

def fetch_account_summary():
    """从Oanda API获取账户摘要"""
    oanda_client = OandaClient(
        api_key=os.getenv("OANDA_API_KEY"),
        account_id=os.getenv("OANDA_ACCOUNT_ID")
    )
    return oanda_client.get_account_summary()

def fetch_recent_transactions(count=10):
    """获取最近的交易记录"""
    oanda_client = OandaClient(
        api_key=os.getenv("OANDA_API_KEY"),
        account_id=os.getenv("OANDA_ACCOUNT_ID")
    )
    return oanda_client.get_transactions(count)

def fetch_equity_history(period="7D"):
    """获取账户净值历史"""
    oanda_client = OandaClient(
        api_key=os.getenv("OANDA_API_KEY"),
        account_id=os.getenv("OANDA_ACCOUNT_ID")
    )
    return oanda_client.get_equity_history(period)

def trading_thread_target(components):
    """交易逻辑线程的目标函数"""
    trading_loop(components)

# ====================== 交易控制函数 ======================
def start_trading_system():
    """初始化并在后台线程中启动交易系统"""
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
            st.success("交易系统已启动")
            try:
                st.experimental_rerun()
            except st.errors.RerunException:
                pass
        else:
            st.warning("交易系统已在运行中")
    else:
        st.error("系统组件未初始化，无法启动交易逻辑")

def stop_and_close_all():
    """停止交易并平掉所有仓位"""
    if 'components' in st.session_state:
        system_state = st.session_state.components['system_state']
        position_manager = st.session_state.components['position_manager']
        
        if system_state.is_running:
            system_state.set_running(False)
            
        # 关闭所有开仓
        position_manager.close_all_positions()
        st.success("所有仓位已平仓，交易已停止")
    else:
        st.warning("系统组件未初始化")

def pause_trading():
    """暂停交易但不平仓"""
    if 'components' in st.session_state:
        system_state = st.session_state.components['system_state']
        if system_state.is_running:
            system_state.set_running(False)
            st.success("交易已暂停，仓位保持开放")
        else:
            st.warning("交易未运行")
    else:
        st.warning("系统组件未初始化")
        
def resume_trading():
    """恢复暂停的交易"""
    if 'components' in st.session_state:
        system_state = st.session_state.components['system_state']
        if not system_state.is_running:
            system_state.set_running(True)
            st.success("交易已恢复")
        else:
            st.warning("交易已在运行中")
    else:
        st.warning("系统组件未初始化")

def stop_trading_system():
    """停止交易逻辑"""
    if 'components' in st.session_state:
        system_state = st.session_state.components['system_state']
        if system_state.is_running:
            system_state.set_running(False)
            if 'trading_thread' in st.session_state:
                st.session_state.trading_thread.join(timeout=10)
            st.success("交易系统已停止")
            try:
                st.experimental_rerun()
            except st.errors.RerunException:
                pass
        else:
            st.warning("交易系统未运行")

# ====================== 主应用函数 ======================
def run_app():
    """构建并运行Streamlit UI的主函数"""
    # 设置页面配置
    st.set_page_config(
        page_title="Oanda交易系统",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="📊",
        menu_items={
            'Get Help': 'https://github.com/your-repo',
            'Report a bug': "https://github.com/your-repo/issues",
        }
    )
    
    # 初始化主题设置
    if 'theme' not in st.session_state:
        st.session_state.theme = "light"
    
    # 标题区域
    st.title("🎯 Oanda实时交易仪表板")
    
    # 初始化系统组件
    if 'components' not in st.session_state:
        with st.spinner('正在初始化交易系统组件...'):
            st.session_state.components = initialize_system()

        if st.session_state.components is None:
            st.error("交易系统初始化失败，请检查日志")
            return

    components = st.session_state.components
    system_state = components['system_state']
    position_manager = components['position_manager']
    db_manager = components['db_manager']
    instrument_monitor = components['instrument_monitor']
    if 'order_manager' not in components:
        order_manager = OrderManager(
            client=components['oanda_client'],
            system_state=components['system_state'],
            position_manager=components['position_manager'],
            risk_manager=components['risk_manager'],
            db_manager=components['db_manager']
        )
        components['order_manager'] = order_manager
    else:
        order_manager = components['order_manager']
    
    # ====================== 侧边栏 ======================
    with st.sidebar:
        # 主题选择器
        st.subheader("主题设置")
        theme_choice = st.radio("选择主题", ["亮色", "暗色"], index=0 if st.session_state.theme == "light" else 1)
        st.session_state.theme = "light" if theme_choice == "亮色" else "dark"
        
        # 应用主题样式
        try:
            # 检查Streamlit版本是否支持原生主题（1.16.0+）
            st_version = st.__version__
            major, minor, patch = map(int, st_version.split('.'))
            
            if (major, minor) >= (1, 16):
                # 使用原生主题设置
                st._config.set_option("theme.base", st.session_state.theme)
            else:
                # 旧版本使用CSS覆盖
                if st.session_state.theme == "dark":
                    st.markdown(
                        """
                        <style>
                            .stApp { background-color: #1e1e1e; }
                            .css-18e3th9 { background-color: #1e1e1e; }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <style>
                            .stApp { background-color: #ffffff; }
                            .css-18e3th9 { background-color: #ffffff; }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
        except Exception as e:
            st.error(f"主题设置失败: {str(e)}")
            # 回退到亮色主题
            st.session_state.theme = "light"
            st.error("已回退到亮色主题")
        
        # 交易品种选择器
        st.subheader("交易品种")
        categorized_symbols = get_categorized_symbols()
        selected_symbols = []
        
        # 使用选项卡组织不同类别
        tabs = st.tabs(list(categorized_symbols.keys()))
        for idx, (category, symbols) in enumerate(categorized_symbols.items()):
            with tabs[idx]:
                for sym, display, _ in symbols:
                    if st.checkbox(f"{display} ({sym})", key=f"sym_{sym}"):
                        selected_symbols.append(sym)
        
        # 交易参数设置（使用扩展器）
        with st.expander("⚙️ 交易参数", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                risk_percentage = st.slider("风险系数 (%)", 0.1, 10.0, 1.0, step=0.1)
            with col2:
                stop_loss_multiplier = st.slider("止损系数", 1.0, 5.0, 2.0, step=0.1)
        
        # 三阶风险管理参数
        with st.expander("🛡️ 风险管理", expanded=True):
            st.subheader("账户级风险")
            account_risk = st.slider("最大账户风险 (%)", 1.0, 30.0, 10.0, step=0.5)
            
            st.subheader("品种级风险")
            symbol_risk = st.slider("单品种最大风险 (%)", 0.5, 15.0, 5.0, step=0.5)
            
            st.subheader("订单级风险")
            order_risk = st.slider("单订单最大风险 (%)", 0.1, 5.0, 1.0, step=0.1)
            
            # 风险计算器
            if 'account_summary' in st.session_state and st.session_state.account_summary:
                balance = float(st.session_state.account_summary['balance'])
                max_position_size = balance * order_risk / 100
                st.metric("最大持仓量", f"{max_position_size:.2f} {st.session_state.account_summary['currency']}")
        
        # 动态模型选择器
        st.subheader("模型选择")
        num_selected = len(selected_symbols)
        available_models = []
        model_pattern = re.compile(r'model_(\d+)\.pkl')
        
        # 扫描weights目录下的模型文件
        weights_dir = os.path.join(project_root, 'weights')
        if os.path.exists(weights_dir):
            for fname in os.listdir(weights_dir):
                match = model_pattern.match(fname)
                if match:
                    max_symbols = int(match.group(1))
                    if max_symbols >= num_selected:
                        available_models.append(fname)
        
        if available_models:
            st.session_state.selected_model = st.selectbox("选择交易模型", options=available_models, index=0)
            st.info(f"已选择模型: {st.session_state.selected_model} (支持最多 {model_pattern.match(st.session_state.selected_model).group(1)} 个品种)")
        else:
            st.warning("找不到匹配的模型，请选择更少的品种或添加新模型")
        
        # 交易控制按钮
        st.subheader("交易控制")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ 开始交易", type="primary", use_container_width=True):
                start_trading_system()
        with col2:
            if st.button("⏸️ 暂停交易", type="secondary", use_container_width=True):
                pause_trading()
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("⏹️ 停止交易", type="secondary", use_container_width=True):
                stop_trading_system()
        with col4:
            if st.button("🚨 一键平仓", type="secondary", use_container_width=True, 
                        help="立即平掉所有仓位"):
                stop_and_close_all()
        
        # 自动刷新设置
        st.subheader("系统设置")
        auto_refresh = st.checkbox("启用自动刷新", value=st.session_state.get('auto_refresh', True))
        st.session_state.auto_refresh = auto_refresh
        if auto_refresh:
            refresh_interval = st.slider("刷新间隔（秒）", 1, 30, st.session_state.get('refresh_interval', 5))
            st.session_state.refresh_interval = refresh_interval
    
    # ====================== 主界面 ======================
    # 账户信息面板
    st.subheader("💰 账户信息")
    try:
        st.session_state.account_summary = fetch_account_summary()
        if st.session_state.account_summary:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("账户余额", f"{st.session_state.account_summary['balance']} {st.session_state.account_summary['currency']}")
            with col2:
                st.metric("可用保证金", f"{st.session_state.account_summary['marginAvailable']} {st.session_state.account_summary['currency']}")
            with col3:
                st.metric("净值", f"{st.session_state.account_summary['NAV']} {st.session_state.account_summary['currency']}")
            
            # 显示持仓信息
            positions = position_manager.get_all_positions()
            if positions:
                st.subheader("📊 当前持仓")
                position_data = []
                for pos in positions:
                    units_long = int(pos['long']['units']) if 'long' in pos and 'units' in pos['long'] else 0
                    units_short = int(pos['short']['units']) if 'short' in pos and 'units' in pos['short'] else 0
                    
                    if units_long != 0 or units_short != 0:
                        position_data.append({
                            "品种": pos['instrument'],
                            "多仓": units_long,
                            "空仓": units_short,
                            "净仓": units_long - units_short
                        })
                
                if position_data:
                    st.dataframe(pd.DataFrame(position_data))
            else:
                st.info("当前没有持仓")
    except Exception as e:
        st.error(f"获取账户信息失败: {str(e)}")
    
    # 净值曲线
    st.subheader("📈 账户净值曲线")
    try:
        equity_history = fetch_equity_history(period="7D")
        if equity_history and 'changes' in equity_history:
            df_equity = pd.DataFrame(equity_history['changes'])
            df_equity['time'] = pd.to_datetime(df_equity['time'])
            df_equity = df_equity.set_index('time')
            
            # 绘制净值曲线
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=df_equity.index,
                y=df_equity['balance'],
                name="账户净值",
                line=dict(color='#4CAF50' if st.session_state.theme == "light" else '#81C784')
            ))
            
            fig_equity.update_layout(
                title="7日账户净值变化",
                xaxis_title="时间",
                yaxis_title="金额",
                height=300,
                template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white"
            )
            st.plotly_chart(fig_equity, use_container_width=True)
        else:
            st.warning("无法获取净值历史数据")
    except Exception as e:
        st.error(f"净值曲线生成失败: {str(e)}")
    
    # 交易记录
    st.subheader("📝 最近交易")
    try:
        transactions = fetch_recent_transactions(count=10)
        if transactions:
            trans_data = []
            for trans in transactions:
                trans_data.append({
                    "时间": trans['time'],
                    "类型": trans['type'],
                    "品种": trans['instrument'],
                    "数量": trans['units'],
                    "价格": trans['price'],
                    "盈亏": trans.get('pl', 'N/A')
                })
            st.dataframe(pd.DataFrame(trans_data))
        else:
            st.info("没有找到交易记录")
    except Exception as e:
        st.error(f"获取交易记录失败: {str(e)}")
    
    # 风险敞口热力图
    st.subheader("🔥 风险敞口分析")
    positions = position_manager.get_all_positions()
    if positions:
        exposure_data = []
        for pos in positions:
            instrument = pos['instrument']
            long_units = int(pos['long']['units']) if 'long' in pos and 'units' in pos['long'] else 0
            short_units = int(pos['short']['units']) if 'short' in pos and 'units' in pos['short'] else 0
            exposure = abs(long_units) + abs(short_units)
            exposure_data.append({"品种": instrument, "敞口": exposure})
        
        df_exposure = pd.DataFrame(exposure_data)
        
        # 绘制风险敞口热力图
        fig_heatmap = px.treemap(
            df_exposure, 
            path=['品种'], 
            values='敞口',
            color='敞口',
            color_continuous_scale='RdYlGn_r',
            height=400
        )
        fig_heatmap.update_layout(
            title="持仓风险分布热力图",
            margin=dict(t=40, l=0, r=0, b=0)
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("当前没有持仓，无风险敞口")
    
    # 多品种图表系统
    if selected_symbols:
        st.subheader("📊 品种图表")
        tabs = st.tabs([f"{sym}" for sym in selected_symbols])
        for i, symbol in enumerate(selected_symbols):
            with tabs[i]:
                # 获取K线数据
                oanda_client = OandaClient(
                    api_key=os.getenv("OANDA_API_KEY"),
                    account_id=os.getenv("OANDA_ACCOUNT_ID")
                )
                candles = oanda_client.get_candles(symbol, count=100, granularity="M15")
                
                if candles:
                    # 生成K线图
                    fig = generate_candlestick_chart(candles, symbol, st.session_state.theme)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 技术指标控制
                    with st.expander("技术指标设置", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            show_macd = st.checkbox("显示MACD", True, key=f"macd_{symbol}")
                        with col2:
                            show_rsi = st.checkbox("显示RSI", True, key=f"rsi_{symbol}")
                else:
                    st.warning(f"无法获取 {symbol} 的K线数据")
    else:
        st.info("请在左侧选择交易品种以显示图表")
    
    # 实时警报系统
    st.subheader("🚨 实时警报")
    try:
        alerts = instrument_monitor.get_alerts()
        if alerts:
            for alert in alerts:
                st.warning(f"{alert['symbol']} - {alert['message']} (时间: {alert['time']})")
        else:
            st.info("当前没有触发警报")
    except Exception as e:
        st.error(f"获取警报失败: {str(e)}")
    
    # 自动刷新逻辑
    if system_state.is_running and st.session_state.get('auto_refresh', True):
        time.sleep(st.session_state.get('refresh_interval', 5))
        try:
            st.experimental_rerun()
        except st.errors.RerunException:
            pass

def start_streamlit_app(components):
    """从main.py调用的启动UI函数"""
    if 'components' not in st.session_state:
        st.session_state.components = components
    
    run_app()

if __name__ == '__main__':
    run_app()
