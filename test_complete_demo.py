#!/usr/bin/env python3
"""
完整的Streamlit監控頁面測試演示
展示所有圖表的預期輸出效果
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys

# 確保能找到src模組
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.common.config import ACCOUNT_CURRENCY, INITIAL_CAPITAL
except ImportError:
    # 如果無法導入，使用默認值
    ACCOUNT_CURRENCY = "AUD"
    INITIAL_CAPITAL = 100000

# 設置頁面配置
st.set_page_config(
    page_title="AI交易監控演示",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_comprehensive_test_data():
    """生成完整的測試數據用於演示所有圖表功能"""
    np.random.seed(42)  # 確保可重現的結果
    
    # 生成100個數據點
    num_points = 100
    
    # 生成訓練步數
    steps = list(range(0, num_points * 100, 100))
    
    # 生成獎勵數據（逐漸改善的趨勢，帶有合理的波動）
    rewards = []
    base_reward = -2.5
    for i in range(num_points):
        # 添加學習趨勢
        trend = i * 0.04  # 逐漸改善
        # 添加週期性波動（模擬市場週期）
        cycle = 0.3 * np.sin(i / 10) 
        # 添加隨機噪聲
        noise = np.random.normal(0, 0.4)
        reward = base_reward + trend + cycle + noise
        rewards.append(reward)
    
    # 生成投資組合淨值（基於獎勵計算）
    portfolio_values = []
    current_value = INITIAL_CAPITAL
    
    for i, reward in enumerate(rewards):
        # 將獎勵轉換為收益率
        return_rate = reward * 0.0008  # 縮放因子
        # 添加市場波動
        market_volatility = np.random.normal(0, 0.003)
        current_value *= (1 + return_rate + market_volatility)
        # 確保不會變成負數
        current_value = max(current_value, INITIAL_CAPITAL * 0.5)
        portfolio_values.append(current_value)
    
    # 生成損失數據（應該逐漸下降）
    losses = []
    for i in range(num_points):
        # Actor損失：策略網絡損失
        actor_loss = 0.8 * np.exp(-i/25) + 0.1 + np.random.normal(0, 0.05)
        # Critic損失：價值網絡損失
        critic_loss = 1.2 * np.exp(-i/20) + 0.15 + np.random.normal(0, 0.08)
        losses.append({
            'actor_loss': max(0.01, actor_loss),
            'critic_loss': max(0.01, critic_loss)
        })
    
    # 生成範數數據
    norms = []
    for i in range(num_points):
        # L2範數：模型參數範數
        l2_norm = 12 + 3 * np.sin(i/15) + np.random.normal(0, 0.5)
        # 梯度範數：應該逐漸減小
        grad_norm = 2.0 * np.exp(-i/40) + 0.2 + np.random.normal(0, 0.1)
        norms.append({
            'l2_norm': max(0.1, l2_norm),
            'grad_norm': max(0.01, grad_norm)
        })
    
    # 生成時間戳
    start_time = datetime.now() - timedelta(hours=5)
    timestamps = [start_time + timedelta(minutes=i*3) for i in range(num_points)]
    
    # 生成交易品種統計
    symbols = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CAD', 'EUR_GBP', 'GBP_JPY']
    symbol_stats = {}
    
    for i, symbol in enumerate(symbols):
        # 不同品種有不同的特性
        base_performance = 0.5 + (i * 0.3)  # 不同品種的基礎表現
        num_trades = np.random.randint(50, 200)
        
        # 生成收益分佈
        returns = np.random.normal(base_performance, 1.5 + i*0.2, num_trades)
        
        wins = sum(1 for r in returns if r > 0)
        win_rate = (wins / num_trades) * 100
        avg_return = np.mean(returns)
        max_return = np.max(returns)
        max_loss = np.min(returns)
        
        returns_std = np.std(returns)
        sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0
        
        symbol_stats[symbol] = {
            'trades': num_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'max_return': max_return,
            'max_loss': max_loss,
            'sharpe_ratio': sharpe_ratio,
            'returns': returns.tolist()
        }
    
    return {
        'steps': steps,
        'rewards': rewards,
        'portfolio_values': portfolio_values,
        'losses': losses,
        'norms': norms,
        'symbol_stats': symbol_stats,
        'timestamps': timestamps
    }

def main():
    """主演示函數"""
    
    st.title("📊 AI交易監控系統 - 完整演示")
    st.markdown("**展示所有圖表和監控功能的預期效果**")
    st.markdown("---")
    
    # 生成測試數據
    if st.button("🎲 重新生成測試數據", type="primary"):
        st.session_state.demo_data = generate_comprehensive_test_data()
        st.success("✅ 測試數據已重新生成！")
        st.rerun()
    
    if 'demo_data' not in st.session_state:
        st.session_state.demo_data = generate_comprehensive_test_data()
    
    metrics = st.session_state.demo_data
    
    # 創建標籤頁
    tab1, tab2, tab3 = st.tabs(["📈 主要指標", "📊 交易統計", "🔬 模型診斷"])
    
    with tab1:
        st.header("📈 主要指標監控")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 訓練獎勵趨勢圖
            fig_reward = go.Figure()
            fig_reward.add_trace(go.Scatter(
                x=metrics['steps'],
                y=metrics['rewards'],
                mode='lines',
                name='訓練獎勵',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # 添加移動平均線
            window_size = 20
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
            
            # 投資組合淨值變化圖
            fig_portfolio = go.Figure()
            fig_portfolio.add_trace(go.Scatter(
                x=metrics['steps'],
                y=metrics['portfolio_values'],
                mode='lines',
                name='投資組合淨值',
                line=dict(color='#2ca02c', width=2),
                fill='tonexty'
            ))
            
            # 添加初始資本線
            fig_portfolio.add_hline(
                y=INITIAL_CAPITAL,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"初始資本: {INITIAL_CAPITAL:,.0f} {ACCOUNT_CURRENCY}"
            )
            
            fig_portfolio.update_layout(
                title="投資組合淨值變化",
                xaxis_title="訓練步數",
                yaxis_title=f"淨值 ({ACCOUNT_CURRENCY})",
                height=400
            )
            st.plotly_chart(fig_portfolio, use_container_width=True)
        
        with col2:
            # 實時指標
            st.subheader("📊 實時指標")
            
            latest_idx = -1
            
            # 當前步數
            st.metric(
                "訓練步數",
                f"{metrics['steps'][latest_idx]:,}",
                f"+{metrics['steps'][latest_idx] - metrics['steps'][latest_idx-1]:,}"
            )
            
            # 當前獎勵
            st.metric(
                "當前獎勵",
                f"{metrics['rewards'][latest_idx]:.2f}",
                f"{metrics['rewards'][latest_idx] - metrics['rewards'][latest_idx-1]:.2f}"
            )
            
            # 投資組合淨值
            current_value = metrics['portfolio_values'][latest_idx]
            value_change = current_value - metrics['portfolio_values'][latest_idx-1]
            roi = ((current_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            
            st.metric(
                "投資組合淨值",
                f"{ACCOUNT_CURRENCY} {current_value:,.2f}",
                f"{value_change:+,.2f}"
            )
            
            st.metric(
                "投資回報率",
                f"{roi:.2f}%",
                f"{roi - ((metrics['portfolio_values'][latest_idx-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100:.2f}%"
            )
            
            # 訓練時長
            duration = (metrics['timestamps'][-1] - metrics['timestamps'][0]).total_seconds()
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            st.metric(
                "訓練時長",
                f"{hours}h {minutes}m"
            )
            
            # 平均獎勵
            avg_reward = np.mean(metrics['rewards'][-20:])  # 最近20步的平均
            st.metric(
                "近期平均獎勵",
                f"{avg_reward:.3f}"
            )
    
    with tab2:
        st.header("📊 交易統計分析")
        
        # 交易統計表
        stats_data = []
        for symbol, stats in metrics['symbol_stats'].items():
            stats_data.append({
                '交易品種': symbol,
                '交易次數': stats['trades'],
                '勝率': f"{stats['win_rate']:.1f}%",
                '平均收益': f"{stats['avg_return']:.2f}%",
                '最大收益': f"{stats['max_return']:.2f}%",
                '最大虧損': f"{stats['max_loss']:.2f}%",
                '夏普比率': f"{stats['sharpe_ratio']:.2f}"
            })
        
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        # 交易品種收益分佈圖
        fig_returns = go.Figure()
        for symbol, stats in metrics['symbol_stats'].items():
            fig_returns.add_trace(go.Box(
                y=stats['returns'],
                name=symbol,
                boxpoints='outliers'
            ))
        
        fig_returns.update_layout(
            title="各交易品種收益分佈",
            yaxis_title="收益率 (%)",
            height=500
        )
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # 品種表現對比
        col1, col2 = st.columns(2)
        
        with col1:
            # 勝率對比
            symbols = list(metrics['symbol_stats'].keys())
            win_rates = [metrics['symbol_stats'][s]['win_rate'] for s in symbols]
            
            fig_winrate = go.Figure(data=[
                go.Bar(x=symbols, y=win_rates, marker_color='lightblue')
            ])
            fig_winrate.update_layout(
                title="各品種勝率對比",
                yaxis_title="勝率 (%)",
                height=350
            )
            st.plotly_chart(fig_winrate, use_container_width=True)
        
        with col2:
            # 夏普比率對比
            sharpe_ratios = [metrics['symbol_stats'][s]['sharpe_ratio'] for s in symbols]
            
            fig_sharpe = go.Figure(data=[
                go.Bar(x=symbols, y=sharpe_ratios, marker_color='lightgreen')
            ])
            fig_sharpe.update_layout(
                title="各品種夏普比率對比",
                yaxis_title="夏普比率",
                height=350
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)
    
    with tab3:
        st.header("🔬 模型診斷分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 損失函數圖
            fig_loss = go.Figure()
            
            actor_losses = [l['actor_loss'] for l in metrics['losses']]
            critic_losses = [l['critic_loss'] for l in metrics['losses']]
            
            fig_loss.add_trace(go.Scatter(
                x=metrics['steps'],
                y=actor_losses,
                mode='lines',
                name='Actor Loss',
                line=dict(color='#d62728', width=2)
            ))
            
            fig_loss.add_trace(go.Scatter(
                x=metrics['steps'],
                y=critic_losses,
                mode='lines',
                name='Critic Loss',
                line=dict(color='#9467bd', width=2)
            ))
            
            fig_loss.update_layout(
                title="損失函數變化",
                xaxis_title="訓練步數",
                yaxis_title="損失值",
                height=400
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            # 模型範數圖
            fig_norm = go.Figure()
            
            l2_norms = [n['l2_norm'] for n in metrics['norms']]
            grad_norms = [n['grad_norm'] for n in metrics['norms']]
            
            fig_norm.add_trace(go.Scatter(
                x=metrics['steps'],
                y=l2_norms,
                mode='lines',
                name='L2 Norm',
                line=dict(color='#ff7f0e', width=2)
            ))
            
            fig_norm.add_trace(go.Scatter(
                x=metrics['steps'],
                y=grad_norms,
                mode='lines',
                name='Gradient Norm',
                line=dict(color='#2ca02c', width=2)
            ))
            
            fig_norm.update_layout(
                title="模型範數監控",
                xaxis_title="訓練步數",
                yaxis_title="範數值",
                height=400
            )
            st.plotly_chart(fig_norm, use_container_width=True)
        
        # 訓練穩定性指標
        st.subheader("📊 訓練穩定性分析")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            reward_std = np.std(metrics['rewards'][-50:])
            st.metric("獎勵標準差", f"{reward_std:.3f}")
        
        with col2:
            avg_actor_loss = np.mean([l['actor_loss'] for l in metrics['losses'][-50:]])
            st.metric("平均Actor Loss", f"{avg_actor_loss:.4f}")
        
        with col3:
            avg_grad_norm = np.mean([n['grad_norm'] for n in metrics['norms'][-50:]])
            st.metric("平均梯度範數", f"{avg_grad_norm:.4f}")
        
        with col4:
            final_roi = ((metrics['portfolio_values'][-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            st.metric("最終回報率", f"{final_roi:.2f}%")
        
        # 學習曲線分析
        st.subheader("📈 學習曲線分析")
        
        # 計算移動統計
        window = 10
        reward_ma = pd.Series(metrics['rewards']).rolling(window).mean()
        reward_std_ma = pd.Series(metrics['rewards']).rolling(window).std()
        
        fig_learning = go.Figure()
        
        # 獎勵移動平均
        fig_learning.add_trace(go.Scatter(
            x=metrics['steps'],
            y=reward_ma,
            mode='lines',
            name='獎勵移動平均',
            line=dict(color='blue', width=2)
        ))
        
        # 添加標準差帶
        upper_bound = reward_ma + reward_std_ma
        lower_bound = reward_ma - reward_std_ma
        
        fig_learning.add_trace(go.Scatter(
            x=metrics['steps'],
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig_learning.add_trace(go.Scatter(
            x=metrics['steps'],
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            name='標準差範圍',
            showlegend=True
        ))
        
        fig_learning.update_layout(
            title="學習穩定性分析",
            xaxis_title="訓練步數",
            yaxis_title="獎勵值",
            height=400
        )
        st.plotly_chart(fig_learning, use_container_width=True)
    
    # 總結信息
    st.markdown("---")
    st.subheader("📋 演示總結")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **✅ 主要指標功能：**
        - 訓練獎勵趨勢圖
        - 投資組合淨值變化
        - 實時指標面板
        - 移動平均線分析
        """)
    
    with col2:
        st.markdown("""
        **✅ 交易統計功能：**
        - 品種統計表格
        - 收益分佈箱線圖
        - 勝率對比圖
        - 夏普比率分析
        """)
    
    with col3:
        st.markdown("""
        **✅ 模型診斷功能：**
        - 損失函數監控
        - 模型範數分析
        - 訓練穩定性指標
        - 學習曲線分析
        """)
    
    st.success("🎉 所有圖表功能均正常運作！這展示了監控系統的完整功能。")

if __name__ == "__main__":
    main()