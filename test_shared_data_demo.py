#!/usr/bin/env python3
"""
共享數據管理器測試頁面
展示使用共享序列進行實時數據同步的完整功能
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
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import sys
import os

# 確保能找到src模組
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 基本配置
ACCOUNT_CURRENCY = "USD"
INITIAL_CAPITAL = 100000

# 設置頁面配置
st.set_page_config(
    page_title="共享數據管理器測試",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SharedTrainingDataManager:
    """共享訓練數據管理器 - 使用線程安全的數據結構"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.training_status = 'idle'
        self.training_progress = 0
        self.training_error = None
        self.stop_requested = False
        
        # 使用deque作為線程安全的序列
        self.metrics_queue = deque(maxlen=1000)  # 最多保存1000個數據點
        self.trade_queue = deque(maxlen=5000)    # 最多保存5000筆交易記錄
        
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
    
    def add_training_metric(self, step, reward, portfolio_value, actor_loss=None, 
                           critic_loss=None, l2_norm=None, grad_norm=None):
        """添加訓練指標"""
        metric = {
            'step': step,
            'reward': reward,
            'portfolio_value': portfolio_value,
            'actor_loss': actor_loss or 0.0,
            'critic_loss': critic_loss or 0.0,
            'l2_norm': l2_norm or 0.0,
            'grad_norm': grad_norm or 0.0,
            'timestamp': datetime.now()
        }
        
        with self.lock:
            self.metrics_queue.append(metric)
            self.current_metrics = metric.copy()
    
    def add_trade_record(self, symbol, action, price, quantity, profit_loss, timestamp=None):
        """添加交易記錄"""
        trade = {
            'symbol': symbol,
            'action': action,  # 'buy', 'sell', 'hold'
            'price': price,
            'quantity': quantity,
            'profit_loss': profit_loss,
            'timestamp': timestamp or datetime.now()
        }
        
        with self.lock:
            self.trade_queue.append(trade)
            
            # 更新交易品種統計
            if symbol not in self.symbol_stats:
                self.symbol_stats[symbol] = {
                    'trades': 0,
                    'total_profit': 0.0,
                    'wins': 0,
                    'losses': 0,
                    'returns': []
                }
            
            stats = self.symbol_stats[symbol]
            stats['trades'] += 1
            stats['total_profit'] += profit_loss
            stats['returns'].append(profit_loss)
            
            if profit_loss > 0:
                stats['wins'] += 1
            elif profit_loss < 0:
                stats['losses'] += 1
    
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
    
    def clear_data(self):
        """清除所有數據"""
        with self.lock:
            self.metrics_queue.clear()
            self.trade_queue.clear()
            self.symbol_stats.clear()
            self.training_status = 'idle'
            self.training_progress = 0
            self.training_error = None
            self.stop_requested = False
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

# 創建全局共享數據管理器實例
if 'shared_data_manager' not in st.session_state:
    st.session_state.shared_data_manager = SharedTrainingDataManager()

def simulate_real_training(shared_manager, symbols, total_steps):
    """模擬真實訓練過程，展示共享數據管理器的功能"""
    import numpy as np
    
    try:
        st.write("🚀 開始模擬真實訓練過程...")
        
        # 模擬訓練步數
        num_steps = min(100, total_steps // 500)  # 最多100步，每步代表500個實際步數
        
        for step in range(num_steps):
            # 檢查停止信號
            if shared_manager.is_stop_requested():
                st.write(f"⏹️ 收到停止信號，在第{step}步停止訓練")
                break
            
            # 更新進度
            progress = (step + 1) / num_steps * 100
            shared_manager.update_training_status('running', progress)
            
            # 每步添加訓練指標
            current_step = step * 500
            
            # 生成獎勵值（逐漸改善的趨勢）
            base_reward = -2.0 + (step * 0.04) + np.random.randn() * 0.4
            
            # 計算投資組合價值
            if step == 0:
                portfolio_value = INITIAL_CAPITAL
            else:
                return_rate = base_reward * 0.0008
                portfolio_value = shared_manager.current_metrics['portfolio_value'] * (1 + return_rate + np.random.normal(0, 0.003))
            
            # 生成損失數據
            actor_loss = max(0, 0.4 * np.exp(-step/12) + np.random.normal(0, 0.06))
            critic_loss = max(0, 0.6 * np.exp(-step/10) + np.random.normal(0, 0.09))
            
            # 生成範數數據
            l2_norm = max(0, 12 + np.sin(step/6) * 1.5 + np.random.normal(0, 0.4))
            grad_norm = max(0, 1.2 * np.exp(-step/18) + np.random.normal(0, 0.12))
            
            # 添加訓練指標到共享管理器
            shared_manager.add_training_metric(
                step=current_step,
                reward=base_reward,
                portfolio_value=portfolio_value,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                l2_norm=l2_norm,
                grad_norm=grad_norm
            )
            
            # 模擬交易記錄
            if step % 2 == 0:  # 每2步生成一些交易
                for symbol in symbols[:4]:  # 只對前4個品種生成交易
                    if np.random.random() > 0.6:  # 40%概率生成交易
                        action = np.random.choice(['buy', 'sell'], p=[0.6, 0.4])
                        price = 1.0 + np.random.normal(0, 0.015)
                        quantity = np.random.randint(1000, 15000)
                        profit_loss = np.random.normal(0.15, 2.5)
                        
                        shared_manager.add_trade_record(
                            symbol=symbol,
                            action=action,
                            price=price,
                            quantity=quantity,
                            profit_loss=profit_loss
                        )
            
            # 顯示進度信息
            if step % 10 == 0:
                st.write(f"📊 訓練步驟 {current_step}: 獎勵={base_reward:.3f}, 淨值={portfolio_value:.2f}")
            
            time.sleep(0.5)  # 模擬訓練時間
        
        # 訓練完成
        shared_manager.update_training_status('completed', 100)
        st.write("✅ 模擬訓練完成！")
        return True
        
    except Exception as e:
        st.error(f"❌ 模擬訓練過程中發生錯誤: {e}")
        shared_manager.update_training_status('error', error=str(e))
        return False

def main():
    """主應用函數"""
    
    # 標題和描述
    st.title("🧪 共享數據管理器測試頁面")
    st.markdown("**展示使用共享序列進行實時數據同步的完整功能**")
    st.markdown("---")
    
    # 獲取共享數據管理器
    shared_manager = st.session_state.shared_data_manager
    current_status = shared_manager.get_current_status()
    
    # 創建標籤頁
    tab1, tab2, tab3 = st.tabs(["🎮 控制面板", "📊 實時監控", "📋 數據詳情"])
    
    with tab1:
        st.header("🎮 訓練控制面板")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 測試配置")
            
            # 交易品種選擇
            symbols = st.multiselect(
                "選擇測試交易品種",
                ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD", "XAU_USD"],
                default=["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD"]
            )
            
            # 訓練步數
            total_steps = st.number_input(
                "總訓練步數",
                min_value=1000,
                max_value=100000,
                value=25000,
                step=1000
            )
            
            st.info(f"將模擬 {min(100, total_steps // 500)} 個訓練步驟，每步代表500個實際步數")
        
        with col2:
            st.subheader("🔄 當前狀態")
            
            # 狀態顯示
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
            
            status = current_status['status']
            st.markdown(f"**狀態**: {status_colors[status]} {status_texts[status]}")
            
            if status == 'running':
                st.progress(current_status['progress'] / 100)
                st.markdown(f"**進度**: {current_status['progress']:.1f}%")
            elif status == 'error' and current_status['error']:
                st.error(f"錯誤: {current_status['error']}")
            
            # 控制按鈕
            st.subheader("🎮 控制")
            
            can_start = status in ['idle', 'completed', 'error'] and len(symbols) > 0
            can_stop = status == 'running'
            
            if st.button("🚀 開始測試", disabled=not can_start, type="primary"):
                if symbols:
                    shared_manager.clear_data()
                    shared_manager.reset_stop_flag()
                    shared_manager.update_training_status('running', 0)
                    
                    # 在新線程中運行模擬訓練
                    training_thread = threading.Thread(
                        target=simulate_real_training,
                        args=(shared_manager, symbols, total_steps)
                    )
                    training_thread.daemon = True
                    training_thread.start()
                    
                    st.success("🚀 測試已開始！請切換到監控標籤頁查看實時數據。")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("請至少選擇一個交易品種")
            
            if st.button("⏹️ 停止測試", disabled=not can_stop):
                shared_manager.request_stop()
                st.success("⏹️ 已發送停止信號")
                time.sleep(1)
                st.rerun()
            
            if st.button("🔄 重置數據"):
                shared_manager.clear_data()
                st.success("🔄 數據已重置")
                time.sleep(0.5)
                st.rerun()
    
    with tab2:
        st.header("📊 實時監控")
        
        # 從共享數據管理器獲取最新數據
        latest_metrics = shared_manager.get_latest_metrics(1000)
        latest_trades = shared_manager.get_latest_trades(1000)
        
        if latest_metrics:
            # 構建圖表數據
            steps = [m['step'] for m in latest_metrics]
            rewards = [m['reward'] for m in latest_metrics]
            portfolio_values = [m['portfolio_value'] for m in latest_metrics]
            actor_losses = [m['actor_loss'] for m in latest_metrics]
            critic_losses = [m['critic_loss'] for m in latest_metrics]
            l2_norms = [m['l2_norm'] for m in latest_metrics]
            grad_norms = [m['grad_norm'] for m in latest_metrics]
            timestamps = [m['timestamp'] for m in latest_metrics]
            
            # 創建圖表
            col1, col2 = st.columns(2)
            
            with col1:
                # 獎勵趨勢圖
                fig_reward = go.Figure()
                fig_reward.add_trace(go.Scatter(
                    x=steps,
                    y=rewards,
                    mode='lines+markers',
                    name='訓練獎勵',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig_reward.update_layout(
                    title="實時訓練獎勵趨勢",
                    xaxis_title="訓練步數",
                    yaxis_title="獎勵值",
                    height=350
                )
                st.plotly_chart(fig_reward, use_container_width=True)
                
                # 損失函數圖
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=steps,
                    y=actor_losses,
                    mode='lines',
                    name='Actor Loss',
                    line=dict(color='#d62728', width=2)
                ))
                fig_loss.add_trace(go.Scatter(
                    x=steps,
                    y=critic_losses,
                    mode='lines',
                    name='Critic Loss',
                    line=dict(color='#9467bd', width=2)
                ))
                
                fig_loss.update_layout(
                    title="實時損失函數變化",
                    xaxis_title="訓練步數",
                    yaxis_title="損失值",
                    height=350
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                # 投資組合淨值圖
                fig_portfolio = go.Figure()
                fig_portfolio.add_trace(go.Scatter(
                    x=steps,
                    y=portfolio_values,
                    mode='lines+markers',
                    name='投資組合淨值',
                    line=dict(color='#2ca02c', width=2)
                ))
                
                fig_portfolio.add_hline(
                    y=INITIAL_CAPITAL,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"初始資本: {INITIAL_CAPITAL:,}"
                )
                
                fig_portfolio.update_layout(
                    title="實時投資組合淨值",
                    xaxis_title="訓練步數",
                    yaxis_title=f"淨值 ({ACCOUNT_CURRENCY})",
                    height=350
                )
                st.plotly_chart(fig_portfolio, use_container_width=True)
                
                # 範數監控圖
                fig_norm = go.Figure()
                fig_norm.add_trace(go.Scatter(
                    x=steps,
                    y=l2_norms,
                    mode='lines',
                    name='L2 Norm',
                    line=dict(color='#ff7f0e', width=2)
                ))
                fig_norm.add_trace(go.Scatter(
                    x=steps,
                    y=grad_norms,
                    mode='lines',
                    name='Gradient Norm',
                    line=dict(color='#2ca02c', width=2)
                ))
                
                fig_norm.update_layout(
                    title="實時模型範數監控",
                    xaxis_title="訓練步數",
                    yaxis_title="範數值",
                    height=350
                )
                st.plotly_chart(fig_norm, use_container_width=True)
            
            # 實時指標
            st.subheader("📊 實時指標")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                latest_metrics_data = latest_metrics[-1]
                st.metric(
                    "當前步數",
                    f"{latest_metrics_data['step']:,}",
                    f"+{latest_metrics_data['step'] - latest_metrics[-2]['step']:,}" if len(latest_metrics) > 1 else "+0"
                )
            
            with col2:
                st.metric(
                    "當前獎勵",
                    f"{latest_metrics_data['reward']:.3f}",
                    f"{latest_metrics_data['reward'] - latest_metrics[-2]['reward']:.3f}" if len(latest_metrics) > 1 else "0.000"
                )
            
            with col3:
                current_value = latest_metrics_data['portfolio_value']
                roi = ((current_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
                st.metric(
                    "投資回報率",
                    f"{roi:.2f}%",
                    f"{roi - ((latest_metrics[-2]['portfolio_value'] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100:.2f}%" if len(latest_metrics) > 1 else "0.00%"
                )
            
            with col4:
                if len(timestamps) > 0:
                    duration = (timestamps[-1] - timestamps[0]).total_seconds()
                    minutes = int(duration // 60)
                    seconds = int(duration % 60)
                    st.metric(
                        "運行時間",
                        f"{minutes}m {seconds}s"
                    )
            
            # 交易統計
            if current_status['symbol_stats']:
                st.subheader("📊 交易統計")
                
                stats_data = []
                for symbol, stats in current_status['symbol_stats'].items():
                    win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
                    avg_return = np.mean(stats['returns']) if stats['returns'] else 0
                    
                    stats_data.append({
                        '交易品種': symbol,
                        '交易次數': stats['trades'],
                        '勝率': f"{win_rate:.1f}%",
                        '平均收益': f"{avg_return:.2f}%",
                        '總收益': f"{stats['total_profit']:.2f}%"
                    })
                
                if stats_data:
                    df_stats = pd.DataFrame(stats_data)
                    st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        else:
            st.info("📊 暫無實時數據。請在控制面板啟動測試。")
        
        # 自動刷新
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            auto_refresh = st.checkbox("自動刷新", value=True)
        with col2:
            if st.button("🔄 手動刷新"):
                st.rerun()
        
        if auto_refresh and current_status['status'] == 'running':
            time.sleep(2)
            st.rerun()
    
    with tab3:
        st.header("📋 數據詳情")
        
        # 顯示共享數據管理器的詳細信息
        st.subheader("🔍 共享數據管理器狀態")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**基本狀態:**")
            st.json({
                "訓練狀態": current_status['status'],
                "訓練進度": f"{current_status['progress']:.1f}%",
                "錯誤信息": current_status['error'],
                "停止請求": shared_manager.is_stop_requested()
            })
            
            st.markdown("**當前指標:**")
            current_metrics = current_status['current_metrics']
            st.json({
                "步數": current_metrics['step'],
                "獎勵": round(current_metrics['reward'], 4),
                "投資組合價值": round(current_metrics['portfolio_value'], 2),
                "Actor損失": round(current_metrics['actor_loss'], 4),
                "Critic損失": round(current_metrics['critic_loss'], 4),
                "L2範數": round(current_metrics['l2_norm'], 4),
                "梯度範數": round(current_metrics['grad_norm'], 4)
            })
        
        with col2:
            st.markdown("**數據隊列狀態:**")
            st.json({
                "指標數據點數": len(shared_manager.metrics_queue),
                "交易記錄數": len(shared_manager.trade_queue),
                "交易品種數": len(shared_manager.symbol_stats)
            })
            
            if current_status['symbol_stats']:
                st.markdown("**交易品種統計:**")
                st.json(current_status['symbol_stats'])
        
        # 原始數據表格
        if shared_manager.metrics_queue:
            st.subheader("📊 原始指標數據")
            
            # 轉換為DataFrame
            metrics_data = []
            for metric in list(shared_manager.metrics_queue)[-20:]:  # 顯示最近20條
                metrics_data.append({
                    '時間戳': metric['timestamp'].strftime('%H:%M:%S'),
                    '步數': metric['step'],
                    '獎勵': round(metric['reward'], 4),
                    '投資組合價值': round(metric['portfolio_value'], 2),
                    'Actor損失': round(metric['actor_loss'], 4),
                    'Critic損失': round(metric['critic_loss'], 4),
                    'L2範數': round(metric['l2_norm'], 4),
                    '梯度範數': round(metric['grad_norm'], 4)
                })
            
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        if shared_manager.trade_queue:
            st.subheader("💼 原始交易數據")
            
            # 轉換為DataFrame
            trade_data = []
            for trade in list(shared_manager.trade_queue)[-20:]:  # 顯示最近20條
                trade_data.append({
                    '時間戳': trade['timestamp'].strftime('%H:%M:%S'),
                    '交易品種': trade['symbol'],
                    '動作': trade['action'],
                    '價格': round(trade['price'], 5),
                    '數量': trade['quantity'],
                    '盈虧': round(trade['profit_loss'], 2)
                })
            
            df_trades = pd.DataFrame(trade_data)
            st.dataframe(df_trades, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()