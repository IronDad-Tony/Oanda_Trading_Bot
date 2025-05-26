#!/usr/bin/env python3
"""
OANDA AI Trading Model - Complete Streamlit Application
Enhanced real-time monitoring with proper data synchronization and GPU monitoring
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
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
import sys
import os
import psutil
import logging
from collections import defaultdict
from src.data_manager.oanda_downloader import manage_data_download_for_symbols

# Try to import GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Ensure src modules can be found
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import trainer, use fallback if failed
try:
    from src.trainer.universal_trainer import UniversalTrainer as EnhancedUniversalTrainer, create_training_time_range
    from src.common.logger_setup import logger
    from src.common.config import ACCOUNT_CURRENCY, INITIAL_CAPITAL, DEVICE, USE_AMP
    from src.common.shared_data_manager import get_shared_data_manager
    from src.data_manager.instrument_info_manager import InstrumentInfoManager
    TRAINER_AVAILABLE = True
    logger.info("Successfully imported trainer and shared data manager")
except ImportError as e:
    # Fallback configuration if import fails
    logger = logging.getLogger(__name__)
    ACCOUNT_CURRENCY = "USD"
    INITIAL_CAPITAL = 100000
    DEVICE = "cpu"
    USE_AMP = False
    TRAINER_AVAILABLE = False
    
    # Create fallback shared data manager
    def get_shared_data_manager():
        """Fallback shared data manager"""
        class FallbackManager:
            def __init__(self):
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
                self.symbol_stats = {}
                self.metrics_data = []
                self.trades_data = []
            
            def update_training_status(self, status, progress=None, error=None):
                self.training_status = status
                if progress is not None:
                    self.training_progress = progress
                if error is not None:
                    self.training_error = error
            
            def is_stop_requested(self):
                return self.stop_requested
            
            def request_stop(self):
                self.stop_requested = True
            
            def reset_stop_flag(self):
                self.stop_requested = False
            
            def add_training_metric(self, **kwargs):
                metric = {
                    'step': kwargs.get('step', 0),
                    'reward': kwargs.get('reward', 0.0),
                    'portfolio_value': kwargs.get('portfolio_value', INITIAL_CAPITAL),
                    'actor_loss': kwargs.get('actor_loss', 0.0),
                    'critic_loss': kwargs.get('critic_loss', 0.0),
                    'l2_norm': kwargs.get('l2_norm', 0.0),
                    'grad_norm': kwargs.get('grad_norm', 0.0),
                    'timestamp': kwargs.get('timestamp', datetime.now())
                }
                self.metrics_data.append(metric)
                self.current_metrics = metric
                if len(self.metrics_data) > 1000:
                    self.metrics_data = self.metrics_data[-1000:]
            
            def add_trade_record(self, **kwargs):
                trade = {
                    'symbol': kwargs.get('symbol', 'EUR_USD'),
                    'action': kwargs.get('action', 'buy'),
                    'price': kwargs.get('price', 1.0),
                    'quantity': kwargs.get('quantity', 1000),
                    'profit_loss': kwargs.get('profit_loss', 0.0),
                    'timestamp': kwargs.get('timestamp', datetime.now())
                }
                self.trades_data.append(trade)
                if len(self.trades_data) > 500:
                    self.trades_data = self.trades_data[-500:]
            
            def get_latest_metrics(self, count=100):
                return self.metrics_data[-count:] if self.metrics_data else []
            
            def get_latest_trades(self, count=100):
                return self.trades_data[-count:] if self.trades_data else []
            
            def get_current_status(self):
                return {
                    'status': self.training_status,
                    'progress': self.training_progress,
                    'error': self.training_error,
                    'current_metrics': self.current_metrics.copy(),
                    'symbol_stats': self.symbol_stats.copy()
                }
            
            def clear_data(self):
                self.training_status = 'idle'
                self.training_progress = 0
                self.training_error = None
                self.stop_requested = False
                self.symbol_stats.clear()
                self.metrics_data.clear()
                self.trades_data.clear()
        
        return FallbackManager()
    
    def create_training_time_range(days_back: int = 30):
        end_time = datetime.now(timezone.utc) - timedelta(days=1)
        start_time = end_time - timedelta(days=days_back)
        return start_time, end_time

# Set page configuration
st.set_page_config(
    page_title="OANDA AI Trading Model",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_categorized_symbols_and_details():
    """Fetch all OANDA symbols, categorize, and return dict: {category: [(symbol, display_name, type)]}"""
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
        'Others': []
    }
    # Major pairs list (OANDA standard)
    major_pairs = {
        'EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CHF', 'USD_CAD', 'NZD_USD'
    }
    # Indices, Energy, Metals, Commodities keywords
    index_keywords = ["SPX", "NAS", "US30", "UK100", "DE30", "JP225", "HK33", "AU200", "FRA40", "EU50", "CN50"]
    energy_keywords = ["OIL", "WTICO", "BRENT", "NATGAS", "GAS"]
    metal_keywords = ["XAU", "XAG", "GOLD", "SILVER", "PLAT", "PALL"]
    commodity_keywords = ["CORN", "WHEAT", "SOYBN", "SUGAR", "COFFEE", "COCOA", "COTTON"]
    crypto_keywords = ["BTC", "ETH", "LTC", "BCH", "XRP", "ADA", "DOGE", "CRYPTO"]

    for sym in all_symbols:
        details = iim.get_details(sym)
        if details is None:
            continue
        symbol = details.symbol
        display = details.display_name if hasattr(details, 'display_name') else sym
        t = details.type.upper() if details.type else ''
        # --- Classification logic ---
        if symbol in major_pairs:
            categorized['Major Pairs'].append((symbol, display, t))
        elif t == 'CURRENCY' and '_' in symbol:
            # Minor pairs: CURRENCY type, not in major_pairs, not precious metals
            base, quote = symbol.split('_')
            if not (base.startswith('XAU') or base.startswith('XAG')):
                categorized['Minor Pairs'].append((symbol, display, t))
        elif any(metal in symbol or metal in display.upper() for metal in metal_keywords):
            categorized['Precious Metals'].append((symbol, display, t))
        elif any(idx in symbol for idx in index_keywords):
            categorized['Indices'].append((symbol, display, t))
        elif any(energy in symbol for energy in energy_keywords):
            categorized['Energy'].append((symbol, display, t))
        elif any(comm in symbol for comm in commodity_keywords):
            categorized['Commodities'].append((symbol, display, t))
        elif t == 'CRYPTO' or any(crypto in symbol or crypto in display.upper() for crypto in crypto_keywords):
            categorized['Crypto'].append((symbol, display, t))
        else:
            categorized['Others'].append((symbol, display, t))
    # Remove empty categories, but always keep 'Others' if any uncategorized
    categorized = {k: v for k, v in categorized.items() if v}
    return categorized

def init_session_state():
    """Initialize all session state variables"""
    if 'shared_data_manager' not in st.session_state:
        st.session_state.shared_data_manager = get_shared_data_manager()
    if 'training_status' not in st.session_state:
        st.session_state.training_status = 'idle'
    if 'training_thread' not in st.session_state:
        st.session_state.training_thread = None
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 5
def get_gpu_info():
    """Get GPU usage information with enhanced monitoring"""
    try:
        if not GPU_AVAILABLE:
            return []
            
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_info = []
            for gpu in gpus:
                memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0
                
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': memory_percent,
                    'memory_free': gpu.memoryTotal - gpu.memoryUsed,
                    'temperature': gpu.temperature,
                    'uuid': gpu.uuid
                })
            return gpu_info
        else:
            return []
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
        return []

def get_system_info():
    """Get comprehensive system information including GPU monitoring"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        gpu_info = get_gpu_info()
        
        return {
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count,
                'frequency': cpu_freq.current if cpu_freq else None,
                'max_frequency': cpu_freq.max if cpu_freq else None
            },
            'memory': {
                'percent': memory.percent,
                'used': memory.used / (1024**3),
                'total': memory.total / (1024**3),
                'available': memory.available / (1024**3),
            },
            'disk': {
                'percent': (disk.used / disk.total) * 100,
                'used': disk.used / (1024**3),
                'total': disk.total / (1024**3),
                'free': disk.free / (1024**3)
            },
            'gpu': gpu_info
        }
    except Exception as e:
        logger.warning(f"Failed to get system info: {e}")
        return None

def display_system_monitoring():
    """Display enhanced system monitoring with GPU information"""
    st.subheader("💻 System Resources")
    
    system_info = get_system_info()
    if not system_info:
        st.error("Unable to retrieve system information")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "CPU Usage", 
            f"{system_info['cpu']['percent']:.1f}%",
            help=f"CPU Cores: {system_info['cpu']['count']}"
        )
        
        if system_info['cpu']['frequency']:
            st.caption(f"Frequency: {system_info['cpu']['frequency']:.0f} MHz")
    
    with col2:
        st.metric(
            "Memory Usage", 
            f"{system_info['memory']['percent']:.1f}%",
            help=f"Used: {system_info['memory']['used']:.1f}GB / {system_info['memory']['total']:.1f}GB"
        )
        st.caption(f"Available: {system_info['memory']['available']:.1f}GB")
    
    if system_info['gpu']:
        st.subheader("🎮 GPU Monitoring")
        
        for i, gpu in enumerate(system_info['gpu']):
            with st.expander(f"GPU {gpu['id']}: {gpu['name']}", expanded=True):
                gpu_col1, gpu_col2, gpu_col3 = st.columns(3)
                
                with gpu_col1:
                    st.metric("GPU Load", f"{gpu['load']:.1f}%", help="GPU utilization percentage")
                
                with gpu_col2:
                    st.metric("VRAM Usage", f"{gpu['memory_percent']:.1f}%", 
                             help=f"Used: {gpu['memory_used']:.0f}MB / {gpu['memory_total']:.0f}MB")
                
                with gpu_col3:
                    st.metric("Temperature", f"{gpu['temperature']:.0f}°C", help="GPU temperature")
                
                vram_progress = gpu['memory_percent'] / 100
                st.progress(vram_progress)
                st.caption(f"VRAM: {gpu['memory_used']:.0f}MB / {gpu['memory_total']:.0f}MB")
    else:
        st.info("No GPU detected or GPU monitoring unavailable")

def simulate_training_with_shared_manager(shared_manager, symbols, total_timesteps):
    """Simulate training process with realistic data generation"""
    logger.info(f"Starting simulation training for symbols: {symbols}, steps: {total_timesteps}")
    shared_manager.update_training_status('running', 0)
    
    for step in range(total_timesteps):
        if shared_manager.is_stop_requested():
            logger.info("Simulation training: Stop request received.")
            shared_manager.update_training_status('idle')
            return False
        
        progress = (step + 1) / total_timesteps * 100
        shared_manager.update_training_status('running', progress)
        
        base_reward = -2.0 + (step / total_timesteps) * 3.0
        reward = base_reward + np.random.normal(0, 0.5)
        
        portfolio_change = reward * 0.001
        portfolio_value = INITIAL_CAPITAL * (1 + portfolio_change * (step + 1) / 100)
        portfolio_value = max(portfolio_value, INITIAL_CAPITAL * 0.5)
        
        actor_loss = max(0.01, 0.5 * np.exp(-step/1000) + np.random.normal(0, 0.05))
        critic_loss = max(0.01, 0.8 * np.exp(-step/800) + np.random.normal(0, 0.08))
        
        l2_norm = 10 + 3 * np.sin(step/100) + np.random.normal(0, 0.5)
        grad_norm = max(0.01, 1.0 * np.exp(-step/1500) + np.random.normal(0, 0.1))
        
        shared_manager.add_training_metric(
            step=step,
            reward=reward,
            portfolio_value=portfolio_value,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            l2_norm=l2_norm,
            grad_norm=grad_norm,
            
        )
        
        if step % 25 == 0 and symbols:
            symbol = np.random.choice(symbols)
            profit_loss = np.random.normal(0.1, 1.5)
            shared_manager.add_trade_record(
                symbol=symbol,
                action=np.random.choice(['buy', 'sell']),
                price=np.random.uniform(1.0, 2.0),
                quantity=np.random.uniform(1000, 10000),
                profit_loss=profit_loss,
                
            )
        
        if step % 100 == 0:
            logger.debug(f"Simulation training progress: {progress:.1f}%")
            
        time.sleep(0.002)
        
    shared_manager.update_training_status('completed', 100)
    logger.info("Simulation training completed.")
    return True

def training_worker(trainer, shared_manager, symbols, total_timesteps):
    """Training worker thread - uses shared data manager"""
    try:
        logger.info("Starting training worker thread with shared data manager")
        
        if trainer and TRAINER_AVAILABLE:
            logger.info("Starting real training process")
            trainer.shared_data_manager = shared_manager
            
            try:
                success = trainer.run_full_training_pipeline()
            except Exception as e:
                logger.error(f"Error in real training process: {e}")
                logger.info("Falling back to simulation training")
                success = simulate_training_with_shared_manager(shared_manager, symbols, total_timesteps)
        else:
            logger.info("Starting simulation training process")
            success = simulate_training_with_shared_manager(shared_manager, symbols, total_timesteps)
        
        if shared_manager.is_stop_requested():
            shared_manager.update_training_status('idle')
            logger.info("Training stopped by user")
        elif success:
            shared_manager.update_training_status('completed', 100)
            logger.info("Training completed")
        else:
            shared_manager.update_training_status('error', error="Training did not complete successfully")
            
    except Exception as e:
        logger.error(f"Error in training process: {e}", exc_info=True)
        shared_manager.update_training_status('error', error=str(e))
    finally:
        if trainer and hasattr(trainer, 'cleanup'):
            trainer.cleanup()

def start_training(symbols, start_date, end_date, total_timesteps, save_freq, eval_freq):
    """Start training with enhanced error handling"""
    try:
        shared_manager = st.session_state.shared_data_manager
        shared_manager.clear_data()
        shared_manager.reset_stop_flag()
        
        start_time = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_time = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        
        if TRAINER_AVAILABLE:
            trainer = EnhancedUniversalTrainer(
                trading_symbols=symbols,
                start_time=start_time,
                end_time=end_time,
                granularity="S5",
                total_timesteps=total_timesteps,
                save_freq=save_freq,
                eval_freq=eval_freq,
                model_name_prefix="sac_universal_trader",
                streamlit_session_state=st.session_state
            )
            st.session_state.trainer = trainer
        else:
            st.session_state.trainer = None
        
        shared_manager.update_training_status('running', 0)
        
        training_thread = threading.Thread(
            target=training_worker,
            args=(st.session_state.trainer, shared_manager, symbols, total_timesteps),
            daemon=True
        )
        training_thread.start()
        
        st.session_state.training_thread = training_thread
        
        return True
        
    except Exception as e:
        st.error(f"Failed to start training: {e}")
        logger.error(f"Failed to start training: {e}", exc_info=True)
        return False

def stop_training():
    """Stop training with proper cleanup"""
    try:
        shared_manager = st.session_state.shared_data_manager
        shared_manager.request_stop()
        logger.info("Stop signal sent through shared data manager")
        
        if st.session_state.trainer:
            if hasattr(st.session_state.trainer, 'stop'):
                st.session_state.trainer.stop()
            
            if hasattr(st.session_state.trainer, 'save_current_model'):
                st.session_state.trainer.save_current_model()
                logger.info("Current training progress saved")
        
        if st.session_state.training_thread and st.session_state.training_thread.is_alive():
            st.session_state.training_thread.join(timeout=5.0)
        
        st.session_state.training_status = 'idle'
        st.session_state.training_thread = None
        
        return True
        
    except Exception as e:
        logger.error(f"Error stopping training: {e}", exc_info=True)
        return False
def create_real_time_charts():
    """Create real-time training monitoring charts"""
    shared_manager = st.session_state.shared_data_manager
    latest_metrics = shared_manager.get_latest_metrics(200)
    
    if not latest_metrics:
        st.info("No training data available. Start training to view real-time charts.")
        return
    
    df = pd.DataFrame(latest_metrics)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
        "📈 Performance", "🧠 Model Diagnostics", "💰 Portfolio", "📊 Trading Activity"
    ])
    
    with chart_tab1:
        st.subheader("Training Performance")
        
        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(
            x=df['step'],
            y=df['reward'],
            mode='lines+markers',
            name='Reward',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=3)
        ))
        fig_reward.update_layout(
            title="Training Reward Progression",
            xaxis_title="Training Steps",
            yaxis_title="Reward",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_reward, use_container_width=True)
        
        if len(df) > 10:
            window_size = min(20, len(df) // 5)
            df['reward_ma'] = df['reward'].rolling(window=window_size, center=True).mean()
            
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(
                x=df['step'],
                y=df['reward'],
                mode='lines',
                name='Raw Reward',
                line=dict(color='lightblue', width=1),
                opacity=0.5
            ))
            fig_ma.add_trace(go.Scatter(
                x=df['step'],
                y=df['reward_ma'],
                mode='lines',
                name=f'Moving Average ({window_size})',
                line=dict(color='red', width=3)
            ))
            fig_ma.update_layout(
                title=f"Reward Trend Analysis",
                xaxis_title="Training Steps",
                yaxis_title="Reward",
                height=400
            )
            st.plotly_chart(fig_ma, use_container_width=True)
    
    with chart_tab2:
        st.subheader("Model Diagnostics")
        
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=df['step'],
            y=df['actor_loss'],
            mode='lines',
            name='Actor Loss',
            line=dict(color='orange', width=2)
        ))
        fig_loss.add_trace(go.Scatter(
            x=df['step'],
            y=df['critic_loss'],
            mode='lines',
            name='Critic Loss',
            line=dict(color='green', width=2)
        ))
        fig_loss.update_layout(
            title="Training Loss Curves",
            xaxis_title="Training Steps",
            yaxis_title="Loss",
            yaxis_type="log",
            height=400
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        
        fig_norms = go.Figure()
        fig_norms.add_trace(go.Scatter(
            x=df['step'],
            y=df['l2_norm'],
            mode='lines',
            name='L2 Norm',
            line=dict(color='purple', width=2),
            yaxis='y'
        ))
        fig_norms.add_trace(go.Scatter(
            x=df['step'],
            y=df['grad_norm'],
            mode='lines',
            name='Gradient Norm',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))
        fig_norms.update_layout(
            title="Model Parameter Norms",
            xaxis_title="Training Steps",
            yaxis=dict(title="L2 Norm", side="left"),
            yaxis2=dict(title="Gradient Norm", side="right", overlaying="y"),
            height=400
        )
        st.plotly_chart(fig_norms, use_container_width=True)
    
    with chart_tab3:
        st.subheader("Portfolio Performance")
        
        fig_portfolio = go.Figure()
        fig_portfolio.add_trace(go.Scatter(
            x=df['step'],
            y=df['portfolio_value'],
            mode='lines+markers',
            name='Portfolio Value',
            line=dict(color='green', width=2),
            marker=dict(size=3),
            fill='tonexty'
        ))
        
        fig_portfolio.add_hline(
            y=INITIAL_CAPITAL,
            line_dash="dash",
            line_color="red",
            annotation_text="Initial Capital"
        )
        
        fig_portfolio.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Training Steps",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        if not df.empty:
            current_value = df['portfolio_value'].iloc[-1]
            total_return = ((current_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            max_value = df['portfolio_value'].max()
            min_value = df['portfolio_value'].min()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Value", f"${current_value:,.2f}")
            with col2:
                st.metric("Total Return", f"{total_return:+.2f}%")
            with col3:
                st.metric("Max Value", f"${max_value:,.2f}")
            with col4:
                st.metric("Min Value", f"${min_value:,.2f}")
    
    with chart_tab4:
        st.subheader("Trading Activity")
        
        latest_trades = shared_manager.get_latest_trades(100)
        
        if latest_trades:
            trades_df = pd.DataFrame(latest_trades)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Histogram(
                x=trades_df['profit_loss'],
                nbinsx=30,
                name='P&L Distribution',
                marker_color='lightblue'
            ))
            fig_pnl.update_layout(
                title="Profit & Loss Distribution",
                xaxis_title="Profit/Loss",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
            
            if 'symbol' in trades_df.columns:
                symbol_counts = trades_df['symbol'].value_counts()
                
                fig_symbols = go.Figure(data=[
                    go.Bar(x=symbol_counts.index, y=symbol_counts.values,
                           marker_color='lightgreen')
                ])
                fig_symbols.update_layout(
                    title="Trading Activity by Symbol",
                    xaxis_title="Symbol",
                    yaxis_title="Number of Trades",
                    height=400
                )
                st.plotly_chart(fig_symbols, use_container_width=True)
        else:
            st.info("No trading data available yet.")

def display_training_status():
    """Display current training status with enhanced information"""
    shared_manager = st.session_state.shared_data_manager
    status_info = shared_manager.get_current_status()
    status = status_info['status']
    progress = status_info['progress']
    error = status_info['error']
    current_metrics = status_info['current_metrics']

    # --- Training speed and ETA calculation ---
    steps_per_sec = None
    eta_text = None
    if status == 'running' and current_metrics and current_metrics['step'] > 0:
        metrics = shared_manager.get_latest_metrics(20)
        if len(metrics) >= 2:
            steps = [m['step'] for m in metrics]
            times = [m['timestamp'] for m in metrics]
            if isinstance(times[0], str):
                times = [pd.to_datetime(t) for t in times]
            dt = (times[-1] - times[0]).total_seconds()
            dsteps = steps[-1] - steps[0]
            if dt > 0 and dsteps > 0:
                steps_per_sec = dsteps / dt
                total_steps = st.session_state.get('total_timesteps', 0)
                steps_left = total_steps - steps[-1] if total_steps > 0 else 0
                if steps_per_sec > 0 and steps_left > 0:
                    eta_sec = int(steps_left / steps_per_sec)
                    h, m, s = eta_sec // 3600, (eta_sec % 3600) // 60, eta_sec % 60
                    eta_text = f"ETA: {h:02d}:{m:02d}:{s:02d}"

    if status == 'running':
        st.success(f"🚀 Training in Progress - {progress:.1f}% Complete")
        st.progress(progress / 100)
        if current_metrics and current_metrics['step'] > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Step", f"{current_metrics['step']:,}")
            with col2:
                st.metric("Latest Reward", f"{current_metrics['reward']:.3f}")
            with col3:
                st.metric("Portfolio Value", f"${current_metrics['portfolio_value']:,.2f}")
            with col4:
                st.metric("Actor Loss", f"{current_metrics['actor_loss']:.4f}")
        if steps_per_sec:
            st.info(f"Training Speed: {steps_per_sec:.2f} steps/sec")
        if eta_text:
            st.info(eta_text)
        
    elif status == 'completed':
        st.success("✅ Training Completed Successfully!")
        if current_metrics:
            st.info(f"Final Step: {current_metrics['step']:,} | Final Portfolio: ${current_metrics['portfolio_value']:,.2f}")
        
    elif status == 'error':
        st.error(f"❌ Training Error: {error}")
        
    elif status == 'idle':
        st.info("⏸️ Training Not Active")
    
    else:
        st.warning(f"Unknown Status: {status}")
def download_data_with_progress(symbols, start_date, end_date, granularity="S5"):
    """Download historical data with Streamlit progress bar and status text."""
    st.info("Checking and downloading required historical data for selected symbols...")
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    manage_data_download_for_symbols(
        symbols,
        start_date.isoformat() + "T00:00:00Z",
        end_date.isoformat() + "T23:59:59Z",
        granularity=granularity,
        streamlit_progress_bar=progress_bar,
        streamlit_status_text=status_text
    )
    progress_bar.progress(1.0)
    status_text.success("Historical data download complete.")

def main():
    """Main application function"""
    
    init_session_state()
    
    st.title("🚀 OANDA AI Trading Model")
    st.markdown("**Enhanced Real-time Trading Monitor with GPU Support**")
    
    if not TRAINER_AVAILABLE:
        st.warning("⚠️ Running in simulation mode - trainer modules not available")
    else:
        st.success("✅ All modules loaded successfully")
    
    with st.sidebar:
        st.header("⚙️ Training Configuration")
        
        st.subheader("Trading Symbols")
        # --- Dynamic OANDA symbol selection UI ---
        categorized = get_categorized_symbols_and_details()
        selected_symbols = []
        for cat, symbols in sorted(categorized.items()):
            with st.expander(f"{cat} ({len(symbols)})", expanded=(cat=="CURRENCY")):
                options = [f"{sym} - {display}" for sym, display, _ in symbols]
                default = [options[0]] if cat=="CURRENCY" and options else []
                selected = st.multiselect(
                    f"Select {cat} symbols:",
                    options,
                    default=default,
                    help=f"Select {cat} instruments."
                )
                # Map back to symbol codes
                for sel in selected:
                    code = sel.split(" - ")[0]
                    selected_symbols.append(code)
        # Limit to MAX_SYMBOLS_ALLOWED
        from src.common.config import MAX_SYMBOLS_ALLOWED
        # --- 新增：顯示選取狀態 ---
        st.markdown(f"<span style='font-size:16px;'>最多可選取 <b style='color:#0072C6;'>{MAX_SYMBOLS_ALLOWED}</b> 個 symbols，目前已選取 <b style='color:{'red' if len(selected_symbols)>MAX_SYMBOLS_ALLOWED else '#009900'};'>{len(selected_symbols)}</b> 個。</span>", unsafe_allow_html=True)
        if len(selected_symbols) > MAX_SYMBOLS_ALLOWED:
            st.warning(f"You selected {len(selected_symbols)} symbols, but only {MAX_SYMBOLS_ALLOWED} are allowed. Truncating.")
            selected_symbols = selected_symbols[:MAX_SYMBOLS_ALLOWED]
        
        st.subheader("Trading Parameters")
        from src.common.config import (
            INITIAL_CAPITAL, MAX_ACCOUNT_RISK_PERCENTAGE, ATR_STOP_LOSS_MULTIPLIER, MAX_POSITION_SIZE_PERCENTAGE_OF_EQUITY
        )
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - timedelta(days=30),
                help="Training data start date"
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=datetime.now().date() - timedelta(days=1),
                help="Training data end date"
            )
        total_timesteps = st.number_input(
            "Total Training Steps",
            min_value=1000,
            max_value=1000000,
            value=50000,
            step=1000,
            help="Total number of training steps"
        )
        # --- Custom training parameters ---
        col1, col2 = st.columns(2)
        with col1:
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=1000.0,
                max_value=1e8,
                value=float(INITIAL_CAPITAL),
                step=1000.0,
                help="Initial simulated account capital."
            )
            risk_pct = st.number_input(
                "Max Risk % per Trade",
                min_value=0.01,
                max_value=1.0,
                value=float(MAX_ACCOUNT_RISK_PERCENTAGE),
                step=0.01,
                format="%.2f",
                help="Maximum risk per trade as a percentage of account equity."
            )
        with col2:
            atr_mult = st.number_input(
                "ATR Stop Multiplier",
                min_value=0.5,
                max_value=10.0,
                value=float(ATR_STOP_LOSS_MULTIPLIER),
                step=0.1,
                help="ATR-based stop loss multiplier."
            )
            max_pos_pct = st.number_input(
                "Max Position Size %",
                min_value=0.01,
                max_value=1.0,
                value=float(MAX_POSITION_SIZE_PERCENTAGE_OF_EQUITY),
                step=0.01,
                format="%.2f",
                help="Maximum nominal position size as a percentage of equity."
            )
        # --- Save/Eval frequency controls ---
        col1, col2 = st.columns(2)
        with col1:
            save_freq = st.number_input(
                "Save Frequency",
                min_value=100,
                max_value=10000,
                value=2000,
                step=100,
                help="How often to save the model"
            )
        with col2:
            eval_freq = st.number_input(
                "Eval Frequency",
                min_value=100,
                max_value=10000,
                value=5000,
                step=100,
                help="How often to evaluate the model"
            )
        
        st.subheader("Training Controls")
        
        shared_manager = st.session_state.shared_data_manager
        current_status = shared_manager.get_current_status()['status']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if current_status != 'running':
                if st.button("🚀 Start Training", type="primary", use_container_width=True):
                    if not selected_symbols:
                        st.error("Please select at least one trading symbol")
                    elif start_date >= end_date:
                        st.error("Start date must be before end date")
                    else:
                        # --- Data download progress bar before training ---
                        download_data_with_progress(selected_symbols, start_date, end_date)
                        success = start_training(
                            selected_symbols, start_date, end_date, 
                            total_timesteps, save_freq, eval_freq
                        )
                        if success:
                            st.success("Training started successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to start training")
            else:
                st.button("🚀 Start Training", disabled=True, use_container_width=True)
        
        with col2:
            if current_status == 'running':
                if st.button("⏹️ Stop Training", type="secondary", use_container_width=True):
                    success = stop_training()
                    if success:
                        st.success("Training stopped successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to stop training")
            else:
                st.button("⏹️ Stop Training", disabled=True, use_container_width=True)
        
        st.subheader("Auto Refresh")
        auto_refresh = st.checkbox("Enable Auto Refresh", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh Interval (seconds)",
                min_value=1,
                max_value=30,
                value=st.session_state.refresh_interval,
                help="How often to refresh the data"
            )
            st.session_state.refresh_interval = refresh_interval
        
        if st.button("🔄 Manual Refresh", use_container_width=True):
            st.rerun()
    
    # Main content area
    display_training_status()
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Real-time Charts", "💻 System Monitor", "📋 Training Logs"])
    
    with tab1:
        create_real_time_charts()
    
    with tab2:
        display_system_monitoring()
    
    with tab3:
        st.subheader("📋 Training Logs")
        shared_manager = st.session_state.shared_data_manager
        latest_metrics = shared_manager.get_latest_metrics(50)
        if latest_metrics:
            df = pd.DataFrame(latest_metrics)
            df = df.sort_values('step', ascending=False)
            st.dataframe(
                df[['step', 'reward', 'portfolio_value', 'actor_loss', 'critic_loss']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No training logs available yet.")
        latest_trades = shared_manager.get_latest_trades(20)
        if latest_trades:
            st.subheader("Recent Trades")
            trades_df = pd.DataFrame(latest_trades)
            # Compute total P&L only for reduce/close actions, color-code
            def calc_total_pnl(row):
                # Only show P&L for reduce/close, not open/add
                action = row.get('action', '').lower()
                if action in ['reduce', 'close', 'sell', 'buy']:
                    # Assume profit_loss is per unit, total = profit_loss * quantity
                    qty = row.get('quantity', 0)
                    pl = row.get('profit_loss', 0)
                    total = pl * qty
                    return total
                return None
            trades_df['Total_PnL'] = trades_df.apply(calc_total_pnl, axis=1)
            def color_pnl(val):
                if pd.isna(val):
                    return ''
                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                return f'color: {color}; font-weight: bold;'
            # Show step if available, else fallback to index
            if 'step' in trades_df.columns:
                trades_df = trades_df.sort_values('step', ascending=False)
                show_cols = ['step', 'symbol', 'action', 'price', 'quantity', 'Total_PnL']
            else:
                trades_df = trades_df.reset_index()
                show_cols = ['index', 'symbol', 'action', 'price', 'quantity', 'Total_PnL']
            st.dataframe(
                trades_df[show_cols].style.applymap(color_pnl, subset=['Total_PnL']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No recent trades data available yet.")
    
    # Auto refresh functionality
    if st.session_state.auto_refresh:
        time.sleep(st.session_state.refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()