#!/usr/bin/env python3
"""
測試Streamlit監控頁面的數據收集和顯示功能
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# 模擬訓練數據更新
def simulate_training_data():
    """模擬訓練過程中的數據更新"""
    
    # 初始化session state
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = {
            'steps': [],
            'rewards': [],
            'portfolio_values': [],
            'losses': [],
            'norms': [],
            'symbol_stats': {},
            'timestamps': []
        }
    
    # 模擬數據生成
    current_step = len(st.session_state.training_metrics['steps'])
    
    # 添加新的數據點
    st.session_state.training_metrics['steps'].append(current_step * 100)
    st.session_state.training_metrics['timestamps'].append(datetime.now())
    
    # 模擬獎勵（帶有一些噪音的上升趨勢）
    base_reward = -0.5 + (current_step * 0.1) + np.random.randn() * 0.2
    st.session_state.training_metrics['rewards'].append(base_reward)
    
    # 模擬投資組合價值（從10000開始，有波動的增長）
    if current_step == 0:
        portfolio_value = 10000
    else:
        last_value = st.session_state.training_metrics['portfolio_values'][-1]
        portfolio_value = last_value * (1 + np.random.randn() * 0.01 + 0.001)
    st.session_state.training_metrics['portfolio_values'].append(portfolio_value)
    
    # 模擬損失數據
    st.session_state.training_metrics['losses'].append({
        'actor_loss': abs(np.random.randn() * 0.1),
        'critic_loss': abs(np.random.randn() * 0.2)
    })
    
    # 模擬範數數據
    st.session_state.training_metrics['norms'].append({
        'l2_norm': 10 + np.random.randn() * 0.5,
        'grad_norm': 0.5 + abs(np.random.randn() * 0.1)
    })
    
    # 模擬symbol統計
    symbols = ['EUR_USD', 'USD_JPY', 'GBP_USD']
    for symbol in symbols:
        if symbol not in st.session_state.training_metrics['symbol_stats']:
            st.session_state.training_metrics['symbol_stats'][symbol] = {
                'trades': 0,
                'win_rate': 50,
                'avg_return': 0,
                'max_return': 0,
                'max_loss': 0,
                'sharpe_ratio': 0,
                'returns': []
            }
        
        # 隨機更新統計
        stats = st.session_state.training_metrics['symbol_stats'][symbol]
        if np.random.rand() > 0.7:  # 30%機率進行交易
            stats['trades'] += 1
            trade_return = np.random.randn() * 2
            stats['returns'].append(trade_return)
            
            # 更新統計指標
            returns = stats['returns']
            if returns:
                wins = sum(1 for r in returns if r > 0)
                stats['win_rate'] = (wins / len(returns)) * 100
                stats['avg_return'] = np.mean(returns)
                stats['max_return'] = max(returns) if returns else 0
                stats['max_loss'] = min(returns) if returns else 0
                
                if len(returns) > 1:
                    returns_std = np.std(returns)
                    if returns_std > 0:
                        stats['sharpe_ratio'] = stats['avg_return'] / returns_std

def main():
    st.title("📊 Streamlit監控頁面測試")
    
    # 控制面板
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎲 生成測試數據"):
            simulate_training_data()
            st.success("已添加新的測試數據點")
    
    with col2:
        if st.button("🔄 清除數據"):
            if 'training_metrics' in st.session_state:
                st.session_state.training_metrics = {
                    'steps': [],
                    'rewards': [],
                    'portfolio_values': [],
                    'losses': [],
                    'norms': [],
                    'symbol_stats': {},
                    'timestamps': []
                }
            st.success("數據已清除")
    
    with col3:
        auto_generate = st.checkbox("自動生成數據")
    
    # 顯示當前數據狀態
    if 'training_metrics' in st.session_state:
        metrics = st.session_state.training_metrics
        
        st.markdown("---")
        st.subheader("📈 數據統計")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("數據點數量", len(metrics['steps']))
        
        with col2:
            if metrics['rewards']:
                st.metric("最新獎勵", f"{metrics['rewards'][-1]:.3f}")
        
        with col3:
            if metrics['portfolio_values']:
                st.metric("最新淨值", f"{metrics['portfolio_values'][-1]:.2f}")
        
        with col4:
            st.metric("交易品種數", len(metrics['symbol_stats']))
        
        # 顯示圖表
        st.markdown("---")
        st.subheader("📈 訓練監控圖表")
        
        # 創建兩列來顯示圖表
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # 淨值變化圖
            if metrics['portfolio_values'] and metrics['steps']:
                fig_portfolio = go.Figure()
                fig_portfolio.add_trace(go.Scatter(
                    x=metrics['steps'],
                    y=metrics['portfolio_values'],
                    mode='lines',
                    name='投資組合淨值',
                    line=dict(color='#2ca02c', width=2)
                ))
                
                # 添加初始資本線
                fig_portfolio.add_hline(
                    y=10000,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="初始資本: 10,000"
                )
                
                fig_portfolio.update_layout(
                    title="投資組合淨值變化",
                    xaxis_title="訓練步數",
                    yaxis_title="淨值",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_portfolio, use_container_width=True)
            else:
                st.info("等待淨值數據...")
        
        with chart_col2:
            # 訓練獎勵圖
            if metrics['rewards'] and metrics['steps']:
                fig_reward = go.Figure()
                fig_reward.add_trace(go.Scatter(
                    x=metrics['steps'],
                    y=metrics['rewards'],
                    mode='lines',
                    name='訓練獎勵',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # 添加移動平均線
                if len(metrics['rewards']) > 10:
                    window_size = min(20, len(metrics['rewards']) // 3)
                    ma_rewards = pd.Series(metrics['rewards']).rolling(window=window_size).mean()
                    fig_reward.add_trace(go.Scatter(
                        x=metrics['steps'],
                        y=ma_rewards,
                        mode='lines',
                        name=f'{window_size}步移動平均',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))
                
                fig_reward.update_layout(
                    title="訓練獎勵趨勢",
                    xaxis_title="訓練步數",
                    yaxis_title="獎勵值",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_reward, use_container_width=True)
            else:
                st.info("等待獎勵數據...")
        
        # 額外的圖表：損失函數和範數
        if metrics['losses'] or metrics['norms']:
            st.markdown("---")
            st.subheader("🔬 模型診斷圖表")
            
            diag_col1, diag_col2 = st.columns(2)
            
            with diag_col1:
                # 損失函數圖
                if metrics['losses'] and metrics['steps']:
                    fig_loss = go.Figure()
                    
                    actor_losses = [l.get('actor_loss', 0) for l in metrics['losses']]
                    critic_losses = [l.get('critic_loss', 0) for l in metrics['losses']]
                    
                    if actor_losses:
                        fig_loss.add_trace(go.Scatter(
                            x=metrics['steps'][:len(actor_losses)],
                            y=actor_losses,
                            mode='lines',
                            name='Actor Loss',
                            line=dict(color='#d62728', width=2)
                        ))
                    
                    if critic_losses:
                        fig_loss.add_trace(go.Scatter(
                            x=metrics['steps'][:len(critic_losses)],
                            y=critic_losses,
                            mode='lines',
                            name='Critic Loss',
                            line=dict(color='#9467bd', width=2)
                        ))
                    
                    fig_loss.update_layout(
                        title="損失函數變化",
                        xaxis_title="訓練步數",
                        yaxis_title="損失值",
                        height=350
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
            
            with diag_col2:
                # 範數圖
                if metrics['norms'] and metrics['steps']:
                    fig_norm = go.Figure()
                    
                    l2_norms = [n.get('l2_norm', 0) for n in metrics['norms']]
                    grad_norms = [n.get('grad_norm', 0) for n in metrics['norms']]
                    
                    if l2_norms:
                        fig_norm.add_trace(go.Scatter(
                            x=metrics['steps'][:len(l2_norms)],
                            y=l2_norms,
                            mode='lines',
                            name='L2 Norm',
                            line=dict(color='#ff7f0e', width=2)
                        ))
                    
                    if grad_norms:
                        fig_norm.add_trace(go.Scatter(
                            x=metrics['steps'][:len(grad_norms)],
                            y=grad_norms,
                            mode='lines',
                            name='Gradient Norm',
                            line=dict(color='#2ca02c', width=2),
                            yaxis='y2'
                        ))
                    
                    fig_norm.update_layout(
                        title="模型範數監控",
                        xaxis_title="訓練步數",
                        yaxis_title="L2 Norm",
                        yaxis2=dict(
                            title="Gradient Norm",
                            overlaying='y',
                            side='right'
                        ),
                        height=350
                    )
                    st.plotly_chart(fig_norm, use_container_width=True)
        
        # 顯示symbol統計
        if metrics['symbol_stats']:
            st.markdown("---")
            st.subheader("📊 交易統計")
            
            stats_data = []
            for symbol, stats in metrics['symbol_stats'].items():
                stats_data.append({
                    '交易品種': symbol,
                    '交易次數': stats['trades'],
                    '勝率': f"{stats['win_rate']:.1f}%",
                    '平均收益': f"{stats['avg_return']:.2f}%",
                    '夏普比率': f"{stats['sharpe_ratio']:.2f}"
                })
            
            if stats_data:
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, use_container_width=True)
    
    # 自動生成數據
    if auto_generate:
        time.sleep(1)
        simulate_training_data()
        st.rerun()
    
    # 使用說明
    with st.expander("ℹ️ 使用說明"):
        st.markdown("""
        這個測試頁面用於驗證Streamlit監控功能的數據收集和顯示。
        
        **功能說明：**
        - **生成測試數據**：模擬訓練過程中的數據更新
        - **清除數據**：重置所有測試數據
        - **自動生成數據**：每秒自動生成新的數據點
        
        **測試內容：**
        1. 訓練步數、獎勵、淨值的收集
        2. 損失函數和模型範數的記錄
        3. 各交易品種的統計信息
        4. 數據的實時更新和顯示
        
        這些數據結構與實際訓練過程中使用的相同，可以用來驗證監控頁面的功能是否正常。
        """)

if __name__ == "__main__":
    main()