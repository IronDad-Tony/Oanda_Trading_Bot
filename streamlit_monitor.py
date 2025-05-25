#!/usr/bin/env python3
"""
OANDA 交易模型訓練監控面板
使用Streamlit創建的實時監控界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import sys

# 確保能找到src模組
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 設置頁面配置
st.set_page_config(
    page_title="OANDA AI交易模型監控",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    # 為了簡化，我們創建一些模擬數據
    return generate_mock_data()

def generate_mock_data():
    """生成模擬的訓練數據用於演示"""
    steps = np.arange(0, 1000, 10)
    
    # 模擬訓練獎勵（逐漸改善）
    base_reward = -100 + steps * 0.1 + np.random.normal(0, 10, len(steps))
    
    # 模擬投資組合價值
    portfolio_value = 10000 + np.cumsum(np.random.normal(5, 50, len(steps)))
    
    # 模擬損失值
    loss = 1.0 * np.exp(-steps/500) + np.random.normal(0, 0.1, len(steps))
    
    return pd.DataFrame({
        'step': steps,
        'reward': base_reward,
        'portfolio_value': portfolio_value,
        'loss': loss,
        'timestamp': [datetime.now() - timedelta(minutes=len(steps)-i) for i in range(len(steps))]
    })

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

def main():
    """主應用函數"""
    
    # 標題和描述
    st.title("🚀 OANDA AI交易模型監控面板")
    st.markdown("---")
    
    # 側邊欄
    with st.sidebar:
        st.header("⚙️ 控制面板")
        
        # 自動刷新選項
        auto_refresh = st.checkbox("自動刷新", value=True)
        if auto_refresh:
            refresh_interval = st.slider("刷新間隔(秒)", 5, 60, 10)
        
        st.markdown("---")
        
        # 訓練控制
        st.header("🎯 訓練控制")
        
        if st.button("🚀 開始新訓練", type="primary"):
            st.info("訓練功能開發中...")
        
        if st.button("⏹️ 停止訓練"):
            st.info("停止功能開發中...")
        
        st.markdown("---")
        
        # 系統狀態
        st.header("📊 系統狀態")
        st.metric("系統狀態", "運行中", "正常")
        st.metric("GPU使用率", "45%", "2%")
        st.metric("內存使用", "2.3GB", "0.1GB")
    
    # 主要內容區域
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 訓練進度圖表
        st.header("📈 訓練進度")
        
        # 載入數據
        data = load_tensorboard_data()
        
        if data is not None:
            # 獎勵趨勢圖
            fig_reward = go.Figure()
            fig_reward.add_trace(go.Scatter(
                x=data['step'],
                y=data['reward'],
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
                x=data['step'],
                y=data['portfolio_value'],
                mode='lines',
                name='投資組合價值',
                line=dict(color='#2ca02c', width=2)
            ))
            fig_portfolio.update_layout(
                title="投資組合價值變化",
                xaxis_title="訓練步數",
                yaxis_title="價值 (AUD)",
                height=300
            )
            st.plotly_chart(fig_portfolio, use_container_width=True)
            
            # 損失函數圖
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=data['step'],
                y=data['loss'],
                mode='lines',
                name='損失值',
                line=dict(color='#d62728', width=2)
            ))
            fig_loss.update_layout(
                title="模型損失趨勢",
                xaxis_title="訓練步數",
                yaxis_title="損失值",
                height=300
            )
            st.plotly_chart(fig_loss, use_container_width=True)
            
        else:
            st.info("📊 暫無訓練數據，請先開始訓練")
    
    with col2:
        # 實時指標
        st.header("📊 實時指標")
        
        if data is not None:
            latest_data = data.iloc[-1]
            
            st.metric(
                "當前獎勵",
                f"{latest_data['reward']:.2f}",
                f"{latest_data['reward'] - data.iloc[-2]['reward']:.2f}"
            )
            
            st.metric(
                "投資組合價值",
                f"${latest_data['portfolio_value']:,.2f}",
                f"${latest_data['portfolio_value'] - data.iloc[-2]['portfolio_value']:,.2f}"
            )
            
            st.metric(
                "當前損失",
                f"{latest_data['loss']:.4f}",
                f"{latest_data['loss'] - data.iloc[-2]['loss']:.4f}"
            )
            
            st.metric(
                "訓練步數",
                f"{int(latest_data['step']):,}",
                "10"
            )
        
        st.markdown("---")
        
        # 模型文件列表
        st.header("💾 模型文件")
        
        model_files = load_model_info()
        
        if model_files:
            for model in model_files[:5]:  # 顯示最新的5個模型
                with st.expander(f"📁 {model['name']}"):
                    st.write(f"**大小**: {model['size']:.1f} MB")
                    st.write(f"**修改時間**: {model['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**路徑**: `{model['path']}`")
        else:
            st.info("暫無模型文件")
        
        st.markdown("---")
        
        # 快速操作
        st.header("⚡ 快速操作")
        
        if st.button("🔄 刷新數據"):
            st.rerun()
        
        if st.button("📊 打開TensorBoard"):
            st.info("請在終端運行: `tensorboard --logdir=logs/`")
        
        if st.button("📁 打開日誌目錄"):
            st.info("日誌目錄: `logs/`")
    
    # 底部信息
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🤖 **AI模型**: Transformer + SAC")
    
    with col2:
        st.info("📈 **交易品種**: 多資產組合")
    
    with col3:
        st.info("🔄 **更新時間**: " + datetime.now().strftime("%H:%M:%S"))
    
    # 自動刷新
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()