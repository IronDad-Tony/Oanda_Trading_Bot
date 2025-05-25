#!/usr/bin/env python3
"""
完整的UI測試演示頁面
展示所有圖表和功能的預期輸出效果
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
from collections import deque

# 設置頁面配置
st.set_page_config(
    page_title="OANDA AI交易模型 - 完整演示",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 模擬配置
ACCOUNT_CURRENCY = "USD"
INITIAL_CAPITAL = 100000

class MockSharedDataManager:
    """模擬共享數據管理器"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.training_status = 'idle'
        self.training_progress = 0
        self.training_error = None
        self.stop_requested = False
        
        # 使用deque作為線程安全的序列
        self.metrics_queue = deque(maxlen=1000)
        self.trade_queue = deque(maxlen=5000)
        
        # 當前統計數據
        self.current_metrics = {
            'step': 0,
            'reward': 0.0,
            'portfolio_value': INITIAL_CAPITAL,
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'l2_norm': 0.0,
            'grad_norm': 0.0,
            'timestamp': datetime.now()
        }
        
        # 交易品種統計
        self.symbol_stats = {}
        
        # 初始化測試數據
        self._generate_test_data()
    
    def _generate_test_data(self):
        """生成測試數據"""
        np.random.seed(42)
        
        # 生成50個訓練指標數據點
        for i in range(50):
            step = i * 1000
            reward = -1.0 + (i * 0.05) + np.random.randn() * 0.3
            
            if i == 0:
                portfolio_value = INITIAL_CAPITAL
            else:
                return_rate = reward * 0.001
                portfolio_value = self.current_metrics['portfolio_value'] * (1 + return_rate + np.random.normal(0, 0.005))
            
            actor_loss = max(0, 0.3 * np.exp(-i/10) + np.random.normal(0, 0.05))
            critic_loss = max(0, 0.5 * np.exp(-i/8) + np.random.normal(0, 0.08))
            l2_norm = max(0, 10 + np.sin(i/5) + np.random.normal(0, 0.3))
            grad_norm = max(0, 0.8 * np.exp(-i/15) + np.random.normal(0, 0.1))
            
            metric = {
                'step': step,
                'reward': reward,
                'portfolio_value': portfolio_value,
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'l2_norm': l2_norm,
                'grad_norm': grad_norm,
                'timestamp': datetime.now() - timedelta(minutes=(50-i)*2)
            }
            
            self.metrics_queue.append(metric)
            self.current_metrics = metric.copy()
        
        # 生成交易統計數據
        symbols = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CAD']
        for symbol in symbols:
            num_trades = np.random.randint(20, 100)
            returns = np.random.normal(0.1, 2.0, num_trades)
            
            wins = sum(1 for r in returns if r > 0)
            losses = sum(1 for r in returns if r < 0)
            total_profit = sum(returns)
            
            win_rate = (wins / num_trades) * 100 if num_trades > 0 else 0
            avg_return = np.mean(returns)
            max_return = np.max(returns)
            max_loss = np.min(returns)
            returns_std = np.std(returns)
            sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0
            
            self.symbol_stats[symbol] = {
                'trades': num_trades,
                'total_profit': total_profit,
                'wins': wins,
                'losses': losses,
                'returns': returns.tolist(),
                'win_rate': win_rate,
                'avg_return': avg_return,
                'max_return': max_return,
                'max_loss': max_loss,
                'sharpe_ratio': sharpe_ratio
            }
            
            # 生成一些交易記錄
            for j in range(min(10, num_trades)):
                trade = {
                    'symbol': symbol,
                    'action': np.random.choice(['buy', 'sell']),
                    'price': 1.0 + np.random.normal(0, 0.01),
                    'quantity': np.random.randint(1000, 10000),
                    'profit_loss': returns[j],
                    'timestamp': datetime.now() - timedelta(minutes=np.random.randint(1, 120))
                }
                self.trade_queue.append(trade)
def update_training_status(self, status, progress=None, error=None):
        """更新訓練狀態"""
        with self.lock:
            self.training_status = status
            if progress is not None:
                self.training_progress = progress
            if error is not None:
                self.training_error = error
    
    def request_stop(self):
        """請求停止訓練"""
        with self.lock:
            self.stop_requested = True
    
    def is_stop_requested(self):
        """檢查是否請求停止"""
        with self.lock:
            return self.stop_requested
    
    def reset_stop_flag(self):
        """重置停止標誌"""
        with self.lock:
            self.stop_requested = False
    
    def get_latest_metrics(self, count=100):
        """獲取最新的訓練指標"""
        with self.lock:
            return list(self.metrics_queue)[-count:] if self.metrics_queue else []
    
    def get_latest_trades(self, count=100):
        """獲取最新的交易記錄"""
        with self.lock:
            return list(self.trade_queue)[-count:] if self.trade_queue else []
    
    def get_current_status(self):
        """獲取當前狀態"""
        with self.lock:
            return {
                'status': self.training_status,
                'progress': self.training_progress,
                'error': self.training_error,
                'current_metrics': self.current_metrics.copy(),
                'symbol_stats': {k: v.copy() for k, v in self.symbol_stats.items()}
            }

# 初始化模擬數據管理器
if 'mock_shared_manager' not in st.session_state:
    st.session_state.mock_shared_manager = MockSharedDataManager()

def simulate_training_process():
    """模擬訓練過程"""
    shared_manager = st.session_state.mock_shared_manager
    
    # 模擬訓練狀態變化
    for progress in range(0, 101, 10):
        shared_manager.update_training_status('running', progress)
        time.sleep(0.1)
        
        if shared_manager.is_stop_requested():
            break
    
    if not shared_manager.is_stop_requested():
        shared_manager.update_training_status('completed', 100)
    else:
        shared_manager.update_training_status('idle', 0)

def main():
    """主應用函數"""
    
    # 標題和描述
    st.title("🚀 OANDA AI交易模型 - 完整UI演示")
    st.markdown("**展示所有圖表和功能的預期輸出效果**")
    st.markdown("---")
    
    # 側邊欄控制
    with st.sidebar:
        st.header("🎮 演示控制")
        
        # 訓練狀態控制
        shared_manager = st.session_state.mock_shared_manager
        current_status = shared_manager.get_current_status()
        
        st.subheader("訓練狀態模擬")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 開始訓練", disabled=current_status['status'] == 'running'):
                shared_manager.reset_stop_flag()
                # 在後台線程中模擬訓練
                training_thread = threading.Thread(target=simulate_training_process)
                training_thread.daemon = True
                training_thread.start()
                st.rerun()
        
        with col2:
            if st.button("⏹️ 停止訓練", disabled=current_status['status'] != 'running'):
                shared_manager.request_stop()
                st.rerun()
        
        # 顯示當前狀態
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
        
        st.markdown(f"**當前狀態**: {status_colors[current_status['status']]} {status_texts[current_status['status']]}")
        
        if current_status['status'] == 'running':
            st.progress(current_status['progress'] / 100)
            st.markdown(f"**進度**: {current_status['progress']:.1f}%")
        
        # 數據刷新控制
        st.subheader("數據刷新")
        auto_refresh = st.checkbox("自動刷新", value=True)
        if auto_refresh:
            refresh_interval = st.slider("刷新間隔(秒)", 1, 10, 3)
        
        if st.button("🔄 手動刷新"):
            st.rerun()
    
    # 創建標籤頁
    tab1, tab2, tab3 = st.tabs(["🎯 訓練配置", "📊 實時監控", "💾 模型管理"])
    
    with tab1:
        st.header("🎯 訓練配置演示")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 交易品種選擇演示
            st.subheader("📈 選擇交易品種")
            
            preset_options = {
                "主要貨幣對": ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD"],
                "歐洲貨幣對": ["EUR_USD", "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_CAD"],
                "日元交叉盤": ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY"],
            }
            
            preset_choice = st.selectbox("選擇預設組合", list(preset_options.keys()))
            selected_symbols = st.multiselect(
                "選擇交易品種",
                ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD"],
                default=preset_options[preset_choice]
            )
            
            # 時間範圍設置演示
            st.subheader("📅 設置訓練時間範圍")
            
            col_date1, col_date2 = st.columns(2)
            
            with col_date1:
                start_date = st.date_input(
                    "開始日期",
                    value=datetime.now().date() - timedelta(days=30)
                )
            
            with col_date2:
                end_date = st.date_input(
                    "結束日期",
                    value=datetime.now().date() - timedelta(days=1)
                )
            
            data_days = (end_date - start_date).days
            st.info(f"📊 將使用 {data_days} 天的歷史數據進行訓練")
            
            # 訓練參數設置演示
            st.subheader("⚙️ 訓練參數")
            
            col_param1, col_param2, col_param3 = st.columns(3)
            
            with col_param1:
                total_timesteps = st.number_input("總訓練步數", min_value=1000, max_value=1000000, value=50000, step=1000)
            
            with col_param2:
                save_freq = st.number_input("保存頻率", min_value=100, max_value=10000, value=2000, step=100)
            
            with col_param3:
                eval_freq = st.number_input("評估頻率", min_value=500, max_value=20000, value=5000, step=500)
            
            estimated_minutes = total_timesteps / 1000 * 2
            st.info(f"⏱️ 預估訓練時間: {estimated_minutes:.0f} 分鐘")
        
        with col2:
            # 訓練狀態顯示演示
            st.subheader("🔄 訓練狀態")
            
            current_status = shared_manager.get_current_status()['status']
            current_progress = shared_manager.get_current_status()['progress']
            
            st.markdown(f"**狀態**: {status_colors[current_status]} {status_texts[current_status]}")
            
            if current_status == 'running':
                st.progress(current_progress / 100)
                st.markdown(f"**進度**: {current_progress:.1f}%")
            
            # 系統資源監控演示
            st.subheader("💻 系統資源")
            try:
                import psutil
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                st.metric("CPU使用率", f"{cpu_percent:.1f}%")
                st.metric("內存使用率", f"{memory_percent:.1f}%")
            except ImportError:
                # 模擬數據
                st.metric("CPU使用率", "45.2%")
                st.metric("內存使用率", "67.8%")
            
            # 訓練控制按鈕演示
            st.subheader("🎮 訓練控制")
            
            can_start = current_status in ['idle', 'completed', 'error']
            can_stop = current_status == 'running'
            
            st.button("🚀 開始訓練", type="primary", disabled=not can_start)
            st.button("⏹️ 停止訓練", disabled=not can_stop)
            st.button("🔄 重置")