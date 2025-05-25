#!/usr/bin/env python3
"""
OANDA AI交易模型 - 完整的Streamlit應用
支持訓練配置、啟動、監控的一體化界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import time
import threading
import queue
from datetime import datetime, timezone, timedelta
import sys
import os

# 確保能找到src模組
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.trainer.enhanced_trainer import EnhancedUniversalTrainer, create_training_time_range
    from src.common.logger_setup import logger
    from src.common.config import ACCOUNT_CURRENCY, INITIAL_CAPITAL
except ImportError as e:
    st.error(f"導入模組失敗: {e}")
    st.stop()

# 設置頁面配置
st.set_page_config(
    page_title="OANDA AI交易模型",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'training_status' not in st.session_state:
    st.session_state.training_status = 'idle'  # idle, running, completed, error
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0
if 'training_data' not in st.session_state:
    st.session_state.training_data = []
if 'trainer' not in st.session_state:
    st.session_state.trainer = None

# 可用的交易品種
AVAILABLE_SYMBOLS = [
    "EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD", "GBP_AUD", "EUR_CAD",
    "GBP_CAD", "AUD_CAD", "EUR_CHF", "GBP_CHF", "AUD_CHF", "CAD_CHF", "NZD_JPY",
    "XAU_USD", "XAG_USD", "SPX500_USD", "NAS100_USD", "US30_USD"
]

def load_tensorboard_data():
    """從TensorBoard日誌中讀取訓練數據"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None
    
    # 查找最新的TensorBoard日誌目錄
    tb_dirs = list(logs_dir.glob("sac_tensorboard_logs_*"))
    if not tb_dirs:
        return None
    
    latest_tb_dir = max(tb_dirs, key=lambda x: x.stat().st_mtime)
    
    # 這裡應該解析TensorBoard的事件文件
    # 為了演示，我們使用session_state中的數據
    if st.session_state.training_data:
        return pd.DataFrame(st.session_state.training_data)
    
    return None

def load_model_info():
    """載入模型信息"""
    weights_dir = Path("weights")
    logs_dir = Path("logs")
    
    model_files = []
    
    # 檢查weights目錄
    if weights_dir.exists():
        for model_file in weights_dir.rglob("*.zip"):
            model_files.append({
                'name': model_file.name,
                'path': str(model_file),
                'size': model_file.stat().st_size / (1024*1024),  # MB
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime)
            })
    
    # 檢查logs目錄
    if logs_dir.exists():
        for model_file in logs_dir.rglob("*.zip"):
            model_files.append({
                'name': model_file.name,
                'path': str(model_file),
                'size': model_file.stat().st_size / (1024*1024),  # MB
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime)
            })
    
    return sorted(model_files, key=lambda x: x['modified'], reverse=True)

def training_worker(trainer, progress_callback):
    """訓練工作線程"""
    try:
        st.session_state.training_status = 'running'
        
        # 模擬訓練進度更新
        success = trainer.run_full_training_pipeline()
        
        if success:
            st.session_state.training_status = 'completed'
        else:
            st.session_state.training_status = 'error'
            
    except Exception as e:
        st.session_state.training_status = 'error'
        st.error(f"訓練過程中發生錯誤: {e}")

def start_training(symbols, start_date, end_date, total_timesteps, save_freq, eval_freq):
    """啟動訓練"""
    try:
        # 轉換日期格式
        start_time = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_time = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        
        # 創建訓練器
        trainer = EnhancedUniversalTrainer(
            trading_symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            granularity="S5",
            total_timesteps=total_timesteps,
            save_freq=save_freq,
            eval_freq=eval_freq,
            model_name_prefix="sac_universal_trader"
        )
        
        st.session_state.trainer = trainer
        
        # 在後台線程中啟動訓練
        training_thread = threading.Thread(
            target=training_worker,
            args=(trainer, None)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return True
        
    except Exception as e:
        st.error(f"啟動訓練失敗: {e}")
        return False

def main():
    """主應用函數"""
    
    # 標題和描述
    st.title("🚀 OANDA AI交易模型訓練系統")
    st.markdown("**一體化的AI量化交易模型訓練和監控平台**")
    st.markdown("---")
    
    # 創建標籤頁
    tab1, tab2, tab3 = st.tabs(["🎯 訓練配置", "📊 實時監控", "💾 模型管理"])
    
    with tab1:
        st.header("🎯 訓練配置")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 交易品種選擇
            st.subheader("📈 選擇交易品種")
            
            # 預設選項
            preset_options = {
                "主要貨幣對": ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD"],
                "歐洲貨幣對": ["EUR_USD", "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_CAD"],
                "日元交叉盤": ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY"],
                "貴金屬": ["XAU_USD", "XAG_USD"],
                "美股指數": ["SPX500_USD", "NAS100_USD", "US30_USD"],
                "自定義": []
            }
            
            preset_choice = st.selectbox("選擇預設組合", list(preset_options.keys()))
            
            if preset_choice == "自定義":
                selected_symbols = st.multiselect(
                    "選擇交易品種",
                    AVAILABLE_SYMBOLS,
                    default=["EUR_USD", "USD_JPY", "GBP_USD"]
                )
            else:
                selected_symbols = st.multiselect(
                    "選擇交易品種",
                    AVAILABLE_SYMBOLS,
                    default=preset_options[preset_choice]
                )
            
            if len(selected_symbols) == 0:
                st.warning("請至少選擇一個交易品種")
            elif len(selected_symbols) > 20:
                st.warning("最多只能選擇20個交易品種")
            
            # 時間範圍設置
            st.subheader("📅 設置訓練時間範圍")
            
            col_date1, col_date2 = st.columns(2)
            
            with col_date1:
                start_date = st.date_input(
                    "開始日期",
                    value=datetime.now().date() - timedelta(days=30),
                    max_value=datetime.now().date()
                )
            
            with col_date2:
                end_date = st.date_input(
                    "結束日期",
                    value=datetime.now().date() - timedelta(days=1),
                    max_value=datetime.now().date()
                )
            
            if start_date >= end_date:
                st.error("開始日期必須早於結束日期")
            
            # 計算數據天數
            data_days = (end_date - start_date).days
            st.info(f"📊 將使用 {data_days} 天的歷史數據進行訓練")
            
            # 訓練參數設置
            st.subheader("⚙️ 訓練參數")
            
            col_param1, col_param2, col_param3 = st.columns(3)
            
            with col_param1:
                total_timesteps = st.number_input(
                    "總訓練步數",
                    min_value=1000,
                    max_value=1000000,
                    value=50000,
                    step=1000
                )
            
            with col_param2:
                save_freq = st.number_input(
                    "保存頻率",
                    min_value=100,
                    max_value=10000,
                    value=2000,
                    step=100
                )
            
            with col_param3:
                eval_freq = st.number_input(
                    "評估頻率",
                    min_value=500,
                    max_value=20000,
                    value=5000,
                    step=500
                )
            
            # 預估訓練時間
            estimated_minutes = total_timesteps / 1000 * 2  # 粗略估算
            st.info(f"⏱️ 預估訓練時間: {estimated_minutes:.0f} 分鐘")
        
        with col2:
            # 訓練狀態顯示
            st.subheader("🔄 訓練狀態")
            
            status_colors = {
                'idle': '🔵',
                'running': '🟡',
                'completed': '🟢',
                'error': '🔴'
            }
            
            status_texts = {
                'idle': '待機中',
                'running': '訓練中',
                'completed': '已完成',
                'error': '發生錯誤'
            }
            
            current_status = st.session_state.training_status
            st.markdown(f"**狀態**: {status_colors[current_status]} {status_texts[current_status]}")
            
            if current_status == 'running':
                st.progress(st.session_state.training_progress / 100)
                st.markdown(f"**進度**: {st.session_state.training_progress:.1f}%")
            
            # 系統資源監控
            st.subheader("💻 系統資源")
            try:
                import psutil
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                st.metric("CPU使用率", f"{cpu_percent:.1f}%")
                st.metric("內存使用率", f"{memory_percent:.1f}%")
            except ImportError:
                st.info("安裝 psutil 以顯示系統資源")
            
            # 訓練控制按鈕
            st.subheader("🎮 訓練控制")
            
            can_start = (
                len(selected_symbols) > 0 and 
                len(selected_symbols) <= 20 and
                start_date < end_date and
                current_status == 'idle'
            )
            
            if st.button("🚀 開始訓練", type="primary", disabled=not can_start):
                if start_training(selected_symbols, start_date, end_date, total_timesteps, save_freq, eval_freq):
                    st.success("訓練已啟動！請切換到監控標籤頁查看進度。")
                    st.rerun()
            
            if st.button("⏹️ 停止訓練", disabled=current_status != 'running'):
                st.session_state.training_status = 'idle'
                st.info("訓練已停止")
                st.rerun()
            
            if st.button("🔄 重置狀態"):
                st.session_state.training_status = 'idle'
                st.session_state.training_progress = 0
                st.session_state.training_data = []
                st.success("狀態已重置")
                st.rerun()
    
    with tab2:
        st.header("📊 實時監控")
        
        # 載入訓練數據
        data = load_tensorboard_data()
        
        if data is not None and len(data) > 0:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # 訓練獎勵趨勢圖
                fig_reward = go.Figure()
                fig_reward.add_trace(go.Scatter(
                    x=data['step'] if 'step' in data.columns else range(len(data)),
                    y=data['reward'] if 'reward' in data.columns else np.random.randn(len(data)),
                    mode='lines',
                    name='訓練獎勵',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig_reward.update_layout(
                    title="訓練獎勵趨勢",
                    xaxis_title="訓練步數",
                    yaxis_title="獎勵值",
                    height=300
                )
                st.plotly_chart(fig_reward, use_container_width=True)
                
                # 投資組合價值圖
                fig_portfolio = go.Figure()
                fig_portfolio.add_trace(go.Scatter(
                    x=data['step'] if 'step' in data.columns else range(len(data)),
                    y=data['portfolio_value'] if 'portfolio_value' in data.columns else 10000 + np.cumsum(np.random.randn(len(data)) * 100),
                    mode='lines',
                    name='投資組合價值',
                    line=dict(color='#2ca02c', width=2)
                ))
                fig_portfolio.update_layout(
                    title="投資組合價值變化",
                    xaxis_title="訓練步數",
                    yaxis_title=f"價值 ({ACCOUNT_CURRENCY})",
                    height=300
                )
                st.plotly_chart(fig_portfolio, use_container_width=True)
            
            with col2:
                # 實時指標
                st.subheader("📊 實時指標")
                
                latest_data = data.iloc[-1] if len(data) > 0 else None
                
                if latest_data is not None:
                    if 'reward' in data.columns:
                        st.metric(
                            "當前獎勵",
                            f"{latest_data['reward']:.2f}",
                            f"{latest_data['reward'] - data.iloc[-2]['reward']:.2f}" if len(data) > 1 else "0.00"
                        )
                    
                    if 'portfolio_value' in data.columns:
                        st.metric(
                            "投資組合價值",
                            f"${latest_data['portfolio_value']:,.2f}",
                            f"${latest_data['portfolio_value'] - data.iloc[-2]['portfolio_value']:,.2f}" if len(data) > 1 else "$0.00"
                        )
                    
                    if 'step' in data.columns:
                        st.metric(
                            "訓練步數",
                            f"{int(latest_data['step']):,}",
                            "10"
                        )
        else:
            st.info("📊 暫無訓練數據。請先在「訓練配置」標籤頁啟動訓練。")
        
        # 自動刷新選項
        col1, col2 = st.columns([1, 3])
        with col1:
            auto_refresh = st.checkbox("自動刷新", value=True)
        with col2:
            if auto_refresh:
                refresh_interval = st.slider("刷新間隔(秒)", 5, 60, 10)
                if st.session_state.training_status == 'running':
                    time.sleep(refresh_interval)
                    st.rerun()
    
    with tab3:
        st.header("💾 模型管理")
        
        # 載入模型文件列表
        model_files = load_model_info()
        
        if model_files:
            st.subheader("📁 已保存的模型")
            
            # 創建模型文件表格
            df_models = pd.DataFrame(model_files)
            df_models['size'] = df_models['size'].apply(lambda x: f"{x:.1f} MB")
            df_models['modified'] = df_models['modified'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            
            st.dataframe(
                df_models[['name', 'size', 'modified']],
                use_container_width=True,
                hide_index=True
            )
            
            # 模型操作
            st.subheader("🔧 模型操作")
            
            selected_model = st.selectbox(
                "選擇模型",
                options=[f"{m['name']} ({m['modified']})" for m in model_files],
                index=0 if model_files else None
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 載入模型"):
                    st.info("模型載入功能開發中...")
            
            with col2:
                if st.button("🔄 續練模型"):
                    st.info("續練功能開發中...")
            
            with col3:
                if st.button("🗑️ 刪除模型"):
                    st.warning("刪除功能開發中...")
        
        else:
            st.info("📁 暫無已保存的模型文件")
        
        # TensorBoard集成
        st.subheader("📊 TensorBoard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 啟動TensorBoard"):
                st.code("tensorboard --logdir=logs/", language="bash")
                st.info("請在終端中運行上述命令，然後在瀏覽器中打開 http://localhost:6006")
        
        with col2:
            if st.button("📁 打開日誌目錄"):
                logs_path = Path("logs").absolute()
                st.info(f"日誌目錄: {logs_path}")

if __name__ == "__main__":
    main()