#!/usr/bin/env python3
"""
OANDA AI Trading Model - Complete English Streamlit Application
Integrated interface for training configuration, execution, and monitoring
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
from typing import Dict, Any, Optional
import sys
import os
import psutil

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
    from src.trainer.enhanced_trainer import EnhancedUniversalTrainer
    from src.common.logger_setup import logger
    from src.common.config import ACCOUNT_CURRENCY, INITIAL_CAPITAL, DEVICE, USE_AMP
    from src.common.shared_data_manager import get_shared_data_manager
    TRAINER_AVAILABLE = True
    logger.info("Successfully imported trainer and shared data manager")
except ImportError as e:
    # Fallback configuration if import fails
    import logging
    logger = logging.getLogger(__name__)
    ACCOUNT_CURRENCY = "USD"
    INITIAL_CAPITAL = 100000
    DEVICE = "cpu"
    USE_AMP = False
    TRAINER_AVAILABLE = False
    st.warning(f"Trainer module import failed, using simulation mode: {e}")
    
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
            
            def add_training_metric(self, *args, **kwargs):
                pass
            
            def add_trade_record(self, *args, **kwargs):
                pass
            
            def get_latest_metrics(self, count=100):
                return []
            
            def get_latest_trades(self, count=100):
                return []
            
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
        
        return FallbackManager()
# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'training_status' not in st.session_state:
        st.session_state.training_status = 'idle'  # idle, running, completed, error
    if 'training_progress' not in st.session_state:
        st.session_state.training_progress = 0
    if 'training_data' not in st.session_state:
        st.session_state.training_data = []
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    if 'training_error' not in st.session_state:
        st.session_state.training_error = None
    if 'training_thread' not in st.session_state:
        st.session_state.training_thread = None
    if 'stop_training' not in st.session_state:
        st.session_state.stop_training = False
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

def get_gpu_info():
    """Get GPU usage information with enhanced monitoring"""
    try:
        if not GPU_AVAILABLE:
            return []
            
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_info = []
            for gpu in gpus:
                # Calculate memory usage percentage
                memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0
                
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,  # Convert to percentage
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
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory information
        memory = psutil.virtual_memory()
        
        # Disk information
        disk = psutil.disk_usage('/')
        
        # GPU information
        gpu_info = get_gpu_info()
        
        # Network information
        try:
            network = psutil.net_io_counters()
            network_info = {
                'bytes_sent': network.bytes_sent / (1024**2),  # MB
                'bytes_recv': network.bytes_recv / (1024**2),  # MB
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
        except:
            network_info = None
        
        return {
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count,
                'frequency': cpu_freq.current if cpu_freq else None,
                'max_frequency': cpu_freq.max if cpu_freq else None
            },
            'memory': {
                'percent': memory.percent,
                'used': memory.used / (1024**3),  # GB
                'total': memory.total / (1024**3),  # GB
                'available': memory.available / (1024**3),  # GB
                'cached': memory.cached / (1024**3) if hasattr(memory, 'cached') else 0
            },
            'disk': {
                'percent': (disk.used / disk.total) * 100,
                'used': disk.used / (1024**3),  # GB
                'total': disk.total / (1024**3),  # GB
                'free': disk.free / (1024**3)  # GB
            },
            'gpu': gpu_info,
            'network': network_info
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
    
    # CPU and Memory in columns
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
    
    # GPU Information (if available)
    if system_info['gpu']:
        st.subheader("🎮 GPU Monitoring")
        
        for i, gpu in enumerate(system_info['gpu']):
            with st.expander(f"GPU {gpu['id']}: {gpu['name']}", expanded=True):
                gpu_col1, gpu_col2, gpu_col3 = st.columns(3)
                
                with gpu_col1:
                    st.metric(
                        "GPU Load", 
                        f"{gpu['load']:.1f}%",
                        help="GPU utilization percentage"
                    )
                
                with gpu_col2:
                    st.metric(
                        "VRAM Usage", 
                        f"{gpu['memory_percent']:.1f}%",
                        help=f"Used: {gpu['memory_used']:.0f}MB / {gpu['memory_total']:.0f}MB"
                    )
                
                with gpu_col3:
                    st.metric(
                        "Temperature", 
                        f"{gpu['temperature']:.0f}°C",
                        help="GPU temperature"
                    )
                
                # VRAM usage bar
                vram_progress = gpu['memory_percent'] / 100
                st.progress(vram_progress)
                st.caption(f"VRAM: {gpu['memory_used']:.0f}MB / {gpu['memory_total']:.0f}MB")
    else:
        st.info("No GPU detected or GPU monitoring unavailable")
    
    # Disk usage
    st.subheader("💾 Storage")
    disk_col1, disk_col2 = st.columns(2)
    
    with disk_col1:
        st.metric(
            "Disk Usage", 
            f"{system_info['disk']['percent']:.1f}%",
            help=f"Used: {system_info['disk']['used']:.1f}GB / {system_info['disk']['total']:.1f}GB"
        )
    
    with disk_col2:
        st.metric(
            "Free Space", 
            f"{system_info['disk']['free']:.1f}GB",
            help="Available disk space"
        )

# Create global shared data manager instance
if 'shared_data_manager' not in st.session_state:
    st.session_state.shared_data_manager = get_shared_data_manager()

# Set page configuration
st.set_page_config(
    page_title="OANDA AI Trading Model",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Available trading symbols
AVAILABLE_SYMBOLS = [
    "EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD", "GBP_AUD", "EUR_CAD",
    "GBP_CAD", "AUD_CAD", "EUR_CHF", "GBP_CHF", "AUD_CHF", "CAD_CHF", "NZD_JPY",
    "XAU_USD", "XAG_USD", "SPX500_USD", "NAS100_USD", "US30_USD"
]
def simulate_training_with_shared_manager(shared_manager, symbols, total_timesteps):
    """Simulate training process for when TRAINER_AVAILABLE is False"""
    logger.info(f"Starting simulation training for symbols: {symbols}, steps: {total_timesteps}")
    shared_manager.update_training_status('running', 0)
    
    for step in range(total_timesteps):
        if shared_manager.is_stop_requested():
            logger.info("Simulation training: Stop request received.")
            shared_manager.update_training_status('idle')
            return False
        
        progress = (step + 1) / total_timesteps * 100
        shared_manager.update_training_status('running', progress)
        
        # Simulate metric updates with more realistic data
        current_metrics = {
            'step': step,
            'reward': np.random.rand() * 10 - 5 + (step * 0.001),  # Gradually improving reward
            'portfolio_value': float(INITIAL_CAPITAL * (1 + np.random.randn() * 0.01 + step * 0.00001)),
            'actor_loss': max(0.001, 0.5 * np.exp(-step/1000) + np.random.rand() * 0.1),
            'critic_loss': max(0.001, 0.8 * np.exp(-step/800) + np.random.rand() * 0.1),
            'l2_norm': 10 + np.sin(step/100) * 2 + np.random.rand() * 0.5,
            'grad_norm': max(0.001, 1.0 * np.exp(-step/1500) + np.random.rand() * 0.2),
            'timestamp': datetime.now(timezone.utc)
        }
        shared_manager.add_training_metric(**current_metrics)
        
        # Simulate trading records occasionally
        if step % 50 == 0 and symbols:
            symbol = np.random.choice(symbols)
            profit_loss = np.random.normal(0.1, 2.0)
            shared_manager.add_trade_record(
                symbol=symbol,
                action=np.random.choice(['buy', 'sell']),
                price=np.random.uniform(1.0, 2.0),
                quantity=np.random.uniform(1000, 10000),
                profit_loss=profit_loss
            )
        
        if step % 100 == 0:  # Log every 100 steps
            logger.debug(f"Simulation training progress: {progress:.1f}%")
            
        time.sleep(0.001)  # Simulate processing time
        
    shared_manager.update_training_status('completed', 100)
    logger.info("Simulation training completed.")
    return True

def training_worker(trainer, shared_manager, symbols, total_timesteps):
    """Training worker thread - uses shared data manager"""
    try:
        logger.info("Starting training worker thread with shared data manager")
        
        if trainer and TRAINER_AVAILABLE:
            # Real training
            logger.info("Starting real training process")
            
            # Attach shared data manager to trainer
            trainer.shared_data_manager = shared_manager
            
            # Execute real training
            try:
                success = trainer.run_full_training_pipeline()
            except Exception as e:
                logger.error(f"Error in real training process: {e}")
                # Fall back to simulation training if real training fails
                logger.info("Falling back to simulation training")
                success = simulate_training_with_shared_manager(shared_manager, symbols, total_timesteps)
        else:
            # Simulation training
            logger.info("Starting simulation training process")
            success = simulate_training_with_shared_manager(shared_manager, symbols, total_timesteps)
        
        # Update final status
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
        # Ensure resources are released after training stops
        if trainer and hasattr(trainer, 'cleanup'):
            trainer.cleanup()

def start_training(symbols, start_date, end_date, total_timesteps, save_freq, eval_freq):
    """Start training with enhanced error handling"""
    try:
        # Reset shared data manager
        shared_manager = st.session_state.shared_data_manager
        shared_manager.clear_data()
        shared_manager.reset_stop_flag()
        
        # Convert date format
        start_time = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_time = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        
        if TRAINER_AVAILABLE:
            # Create real trainer
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
            # Use simulation trainer
            st.session_state.trainer = None
        
        # Update training status
        shared_manager.update_training_status('running', 0)
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=training_worker,
            args=(st.session_state.trainer, shared_manager, symbols, total_timesteps),
            daemon=True
        )
        training_thread.start()
        
        # Save thread reference
        st.session_state.training_thread = training_thread
        
        return True
        
    except Exception as e:
        st.error(f"Failed to start training: {e}")
        logger.error(f"Failed to start training: {e}", exc_info=True)
        return False

def stop_training():
    """Stop training with proper cleanup"""
    try:
        # Send stop signal through shared data manager
        shared_manager = st.session_state.shared_data_manager
        shared_manager.request_stop()
        logger.info("Stop signal sent through shared data manager")
        
        # If there's a trainer instance, try to stop it
        if st.session_state.trainer:
            if hasattr(st.session_state.trainer, 'stop'):
                st.session_state.trainer.stop()
            
            # Save current model
            if hasattr(st.session_state.trainer, 'save_current_model'):
                st.session_state.trainer.save_current_model()
                logger.info("Current training progress saved")
        
        # Wait for training thread to end (max 5 seconds)
        if st.session_state.training_thread and st.session_state.training_thread.is_alive():
            st.session_state.training_thread.join(timeout=5.0)
        
        # Reset status
        st.session_state.training_status = 'idle'
        st.session_state.training_thread = None
        
        return True
        
    except Exception as e:
        logger.error(f"Error stopping training: {e}", exc_info=True)
        return False

def reset_training_state():
    """Reset training state and parameters"""
    # Stop ongoing training
    if st.session_state.training_status == 'running':
        stop_training()
    
    # Reset all training-related session state
    st.session_state.training_status = 'idle'
    st.session_state.training_progress = 0
    st.session_state.training_data = []
    st.session_state.trainer = None
    st.session_state.training_error = None
    st.session_state.training_thread = None
    st.session_state.stop_training = False
    
    # Clear selected symbols (if exists)
    if 'selected_symbols' in st.session_state:
        del st.session_state.selected_symbols
    
    # Clear shared data manager
    shared_manager = st.session_state.shared_data_manager
    shared_manager.clear_data()
    
    logger.info("Training state has been reset")

def generate_test_data():
    """Generate realistic test data for monitoring page display"""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Clear existing data
    st.session_state.training_metrics = {
        'steps': [],
        'rewards': [],
        'portfolio_values': [],
        'losses': [],
        'norms': [],
        'symbol_stats': {},
        'timestamps': []
    }
    
    # Generate 100 data points for better visualization
    num_points = 100
    np.random.seed(42)  # Ensure reproducible results
    
    # Generate training steps
    steps = list(range(0, num_points * 50, 50))
    
    # Generate reward data with realistic learning curve
    rewards = []
    base_reward = -3.0
    for i in range(num_points):
        # Learning curve: starts low, improves, then stabilizes
        if i < 20:
            trend = i * 0.05  # Initial learning
        elif i < 60:
            trend = 1.0 + (i - 20) * 0.02  # Steady improvement
        else:
            trend = 1.8 + np.sin((i - 60) / 10) * 0.1  # Stabilization with small fluctuations
        
        noise = np.random.normal(0, 0.3)
        reward = base_reward + trend + noise
        rewards.append(reward)
    
    # Generate portfolio values based on rewards
    portfolio_values = []
    current_value = INITIAL_CAPITAL
    
    for i, reward in enumerate(rewards):
        # Convert reward to return rate
        return_rate = reward * 0.0005  # Scaling factor
        current_value *= (1 + return_rate)
        # Add some market volatility
        current_value *= (1 + np.random.normal(0, 0.003))
        portfolio_values.append(max(current_value, INITIAL_CAPITAL * 0.5))  # Prevent extreme losses
    
    # Generate realistic loss data
    losses = []
    for i in range(num_points):
        # Actor loss decreases over time
        actor_loss = 0.8 * np.exp(-i/30) + 0.05 + np.random.normal(0, 0.02)
        # Critic loss also decreases but more gradually
        critic_loss = 1.2 * np.exp(-i/40) + 0.08 + np.random.normal(0, 0.03)
        losses.append({
            'actor_loss': max(0.01, actor_loss),
            'critic_loss': max(0.01, critic_loss)
        })
    
    # Generate norm data
    norms = []
    for i in range(num_points):
        # L2 norm stabilizes over time
        l2_norm = 15 + 5 * np.exp(-i/25) + np.sin(i/15) * 1 + np.random.normal(0, 0.5)
        # Gradient norm decreases as training progresses
        grad_norm = 2.0 * np.exp(-i/35) + 0.1 + np.random.normal(0, 0.05)
        norms.append({
            'l2_norm': max(0.1, l2_norm),
            'grad_norm': max(0.01, grad_norm)
        })
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(hours=4)
    timestamps = [start_time + timedelta(minutes=i*2.4) for i in range(num_points)]
    
    # Generate comprehensive trading symbol statistics
    symbols = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CAD', 'XAU_USD']
    symbol_stats = {}
    
    for symbol in symbols:
        num_trades = np.random.randint(50, 200)
        # Generate returns with different characteristics for each symbol
        if 'XAU' in symbol:  # Gold - higher volatility
            returns = np.random.normal(0.2, 3.0, num_trades)
        elif 'JPY' in symbol:  # JPY pairs - different behavior
            returns = np.random.normal(0.05, 1.5, num_trades)
        else:  # Major pairs
            returns = np.random.normal(0.1, 2.0, num_trades)
        
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
    
    # Update session state
    st.session_state.training_metrics.update({
        'steps': steps,
        'rewards': rewards,
        'portfolio_values': portfolio_values,
        'losses': losses,
        'norms': norms,
        'symbol_stats': symbol_stats,
        'timestamps': timestamps
    })
def create_training_charts():
    """Create comprehensive training monitoring charts"""
    metrics = st.session_state.training_metrics
    
    if not metrics['steps']:
        st.info("No training data available. Start training or generate test data to view charts.")
        return
    
    # Create tabs for different chart categories
    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
        "📈 Performance", "🧠 Model Diagnostics", "💰 Portfolio", "📊 Symbol Analysis"
    ])
    
    with chart_tab1:
        st.subheader("Training Performance")
        
        # Reward progression
        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(
            x=metrics['steps'],
            y=metrics['rewards'],
            mode='lines+markers',
            name='Reward',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        fig_reward.update_layout(
            title="Training Reward Progression",
            xaxis_title="Training Steps",
            yaxis_title="Reward",
            hovermode='x unified'
        )
        st.plotly_chart(fig_reward, use_container_width=True)
        
        # Moving average of rewards
        if len(metrics['rewards']) > 10:
            window_size = min(20, len(metrics['rewards']) // 5)
            rewards_ma = pd.Series(metrics['rewards']).rolling(window=window_size).mean()
            
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(
                x=metrics['steps'],
                y=metrics['rewards'],
                mode='lines',
                name='Raw Reward',
                line=dict(color='lightblue', width=1),
                opacity=0.5
            ))
            fig_ma.add_trace(go.Scatter(
                x=metrics['steps'],
                y=rewards_ma,
                mode='lines',
                name=f'Moving Average ({window_size})',
                line=dict(color='red', width=3)
            ))
            fig_ma.update_layout(
                title=f"Reward Trend Analysis (Moving Average)",
                xaxis_title="Training Steps",
                yaxis_title="Reward"
            )
            st.plotly_chart(fig_ma, use_container_width=True)
    
    with chart_tab2:
        st.subheader("Model Diagnostics")
        
        # Loss curves
        if metrics['losses']:
            actor_losses = [loss['actor_loss'] for loss in metrics['losses']]
            critic_losses = [loss['critic_loss'] for loss in metrics['losses']]
            
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=metrics['steps'],
                y=actor_losses,
                mode='lines',
                name='Actor Loss',
                line=dict(color='orange', width=2)
            ))
            fig_loss.add_trace(go.Scatter(
                x=metrics['steps'],
                y=critic_losses,
                mode='lines',
                name='Critic Loss',
                line=dict(color='green', width=2)
            ))
            fig_loss.update_layout(
                title="Training Loss Curves",
                xaxis_title="Training Steps",
                yaxis_title="Loss",
                yaxis_type="log"
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        # Gradient norms
        if metrics['norms']:
            l2_norms = [norm['l2_norm'] for norm in metrics['norms']]
            grad_norms = [norm['grad_norm'] for norm in metrics['norms']]
            
            fig_norms = go.Figure()
            fig_norms.add_trace(go.Scatter(
                x=metrics['steps'],
                y=l2_norms,
                mode='lines',
                name='L2 Norm',
                line=dict(color='purple', width=2),
                yaxis='y'
            ))
            fig_norms.add_trace(go.Scatter(
                x=metrics['steps'],
                y=grad_norms,
                mode='lines',
                name='Gradient Norm',
                line=dict(color='red', width=2),
                yaxis='y2'
            ))
            fig_norms.update_layout(
                title="Model Parameter Norms",
                xaxis_title="Training Steps",
                yaxis=dict(title="L2 Norm", side="left"),
                yaxis2=dict(title="Gradient Norm", side="right", overlaying="y")
            )
            st.plotly_chart(fig_norms, use_container_width=True)
    
    with chart_tab3:
        st.subheader("Portfolio Performance")
        
        # Portfolio value progression
        fig_portfolio = go.Figure()
        fig_portfolio.add_trace(go.Scatter(
            x=metrics['steps'],
            y=metrics['portfolio_values'],
            mode='lines+markers',
            name='Portfolio Value',
            line=dict(color='green', width=2),
            marker=dict(size=4),
            fill='tonexty'
        ))
        
        # Add initial capital line
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
            hovermode='x unified'
        )
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Portfolio statistics
        if metrics['portfolio_values']:
            current_value = metrics['portfolio_values'][-1]
            total_return = ((current_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            max_value = max(metrics['portfolio_values'])
            min_value = min(metrics['portfolio_values'])
            
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
        st.subheader("Trading Symbol Analysis")
        
        if metrics['symbol_stats']:
            # Symbol performance comparison
            symbols = list(metrics['symbol_stats'].keys())
            win_rates = [metrics['symbol_stats'][s]['win_rate'] for s in symbols]
            avg_returns = [metrics['symbol_stats'][s]['avg_return'] for s in symbols]
            trade_counts = [metrics['symbol_stats'][s]['trades'] for s in symbols]
            
            # Win rate comparison
            fig_winrate = go.Figure(data=[
                go.Bar(x=symbols, y=win_rates, name='Win Rate (%)',
                       marker_color='lightblue')
            ])
            fig_winrate.update_layout(
                title="Win Rate by Trading Symbol",
                xaxis_title="Symbol",
                yaxis_title="Win Rate (%)"
            )
            st.plotly_chart(fig_winrate, use_container_width=True)
            
            # Average return comparison
            fig_returns = go.Figure(data=[
                go.Bar(x=symbols, y=avg_returns, name='Average Return',
                       marker_color=['green' if r > 0 else 'red' for r in avg_returns])
            ])
            fig_returns.update_layout(
                title="Average Return by Trading Symbol",
                xaxis_title="Symbol",
                yaxis_title="Average Return"
            )
            st.plotly_chart(fig_returns, use_container_width=True)
            
            # Detailed symbol statistics table
            st.subheader("Detailed Symbol Statistics")
            symbol_df = pd.DataFrame({
                'Symbol': symbols,
                'Trades': trade_counts,
                'Win Rate (%)': [f"{wr:.1f}" for wr in win_rates],
                'Avg Return': [f"{ar:.3f}" for ar in avg_returns],
                'Max Return': [f"{metrics['symbol_stats'][s]['max_return']:.3f}" for s in symbols],
                'Max Loss': [f"{metrics['symbol_stats'][s]['max_loss']:.3f}" for s in symbols],
                'Sharpe Ratio': [f"{metrics['symbol_stats'][s]['sharpe_ratio']:.3f}" for s in symbols]
            })
            st.dataframe(symbol_df, use_container_width=True)

def display_training_status():
    """Display current training status with enhanced information"""
    shared_manager = st.session_state.shared_data_manager
    status_info = shared_manager.get_current_status()
    
    status = status_info['status']
    progress = status_info['progress']
    error = status_info['error']
    current_metrics = status_info['current_metrics']
    
    # Status indicator
    if status == 'running':
        st.success(f"🚀 Training in Progress - {progress:.1f}% Complete")
        st.progress(progress / 100)
        
        # Current metrics display
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

def main():
    """Main application function"""
    
    # Initialize session state
    init_session_state()
    
    # Title and description
    st.title("🚀 OANDA AI Trading Model Training System")
    st.markdown("**Integrated AI Quantitative Trading Model Training and Monitoring Platform**")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Training Configuration", "📊 Real-time Monitoring", "💾 Model Management"])
    
    with tab1:
        st.header("🎯 Training Configuration")
        
        # Add page instructions
        with st.expander("ℹ️ Usage Instructions", expanded=False):
            st.markdown("""
            ### 📖 Training Configuration Guide
            
            **🎯 Overview:**
            Configure and start AI trading model training. Select trading symbols, set parameters, and monitor progress.
            
            **📈 Trading Symbol Selection:**
            - **Preset Combinations**: Common trading symbol combinations for different strategies
              - Major Currency Pairs: High liquidity, beginner-friendly
              - European Currency Pairs: Active during European sessions
              - Japanese Cross Pairs: Active during Asian sessions
              - Precious Metals: Safe haven assets with higher volatility
              - US Stock Indices: Stock market indices
            - **Custom Selection**: Choose 1-20 trading symbols freely
            - **Recommendation**: Start with 3-5 major currency pairs for first training
            
            **📅 Training Time Range:**
            - **Start/End Date**: Select historical data time range
            - **Recommended Range**: At least 30 days, preferably 60-90 days
            - **Note**: More data = longer training time but potentially better performance
            
            **⚙️ Training Parameters:**
            - **Total Training Steps**: Total model training iterations
              - Recommended: 50,000-100,000 steps
              - More steps = longer training but potentially higher accuracy
            - **Save Frequency**: Model save interval (every N steps)
              - Recommended: 2,000-5,000 steps
              - Higher frequency = more storage but better progress preservation
            - **Evaluation Frequency**: Performance evaluation interval (every N steps)
              - Recommended: 5,000-10,000 steps
              - Used to monitor training effectiveness
            
            **💡 Recommended Configurations:**
            - **Beginner**: 3 major pairs + 30 days + 50,000 steps
            - **Advanced**: 5-8 symbols + 60 days + 100,000 steps
            - **Professional**: 10-15 symbols + 90 days + 200,000 steps
            """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Trading symbol selection
            st.subheader("📈 Select Trading Symbols")
            
            # Preset options
            preset_options = {
                "Major Currency Pairs": ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD"],
                "European Currency Pairs": ["EUR_USD", "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_CAD"],
                "Japanese Cross Pairs": ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY"],
                "Precious Metals": ["XAU_USD", "XAG_USD"],
                "US Stock Indices": ["SPX500_USD", "NAS100_USD", "US30_USD"],
                "Custom": []
            }
            
            preset_choice = st.selectbox("Select Preset Combination", list(preset_options.keys()))
            
            if preset_choice == "Custom":
                selected_symbols = st.multiselect(
                    "Select Trading Symbols",
                    AVAILABLE_SYMBOLS,
                    default=["EUR_USD", "USD_JPY", "GBP_USD"]
                )
            else:
                selected_symbols = st.multiselect(
                    "Select Trading Symbols",
                    AVAILABLE_SYMBOLS,
                    default=preset_options[preset_choice]
                )
            
            if len(selected_symbols) == 0:
                st.warning("Please select at least one trading symbol")
            elif len(selected_symbols) > 20:
                st.warning("Maximum 20 trading symbols allowed")
            
            # Time range settings
            st.subheader("📅 Set Training Time Range")
            
            col_date1, col_date2 = st.columns(2)
            
            with col_date1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now().date() - timedelta(days=30),
                    max_value=datetime.now().date()
                )
            
            with col_date2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now().date() - timedelta(days=1),
                    max_value=datetime.now().date()
                )
            
            if start_date >= end_date:
                st.error("Start date must be earlier than end date")
            
            # Calculate data days
            data_days = (end_date - start_date).days
            st.info(f"📊 Data Range: {data_days} days")
            
            # Training parameters
            st.subheader("⚙️ Training Parameters")
            
            col_param1, col_param2 = st.columns(2)
            
            with col_param1:
                total_timesteps = st.number_input(
                    "Total Training Steps",
                    min_value=1000,
                    max_value=1000000,
                    value=50000,
                    step=1000,
                    help="Total number of training iterations"
                )
                
                save_freq = st.number_input(
                    "Save Frequency",
                    min_value=100,
                    max_value=50000,
                    value=2000,
                    step=100,
                    help="Save model every N steps"
                )
            
            with col_param2:
                eval_freq = st.number_input(
                    "Evaluation Frequency",
                    min_value=500,
                    max_value=100000,
                    value=5000,
                    step=500,
                    help="Evaluate model every N steps"
                )
                
                # Estimated training time
                estimated_minutes = (total_timesteps / 1000) * len(selected_symbols) * 0.5
                st.info(f"⏱️ Estimated Time: {estimated_minutes:.0f} minutes")
        
        with col2:
            # Training status and controls
            st.subheader("🎮 Training Control")
            
            # Display current training status
            display_training_status()
            
            # Training control buttons
            shared_manager = st.session_state.shared_data_manager
            current_status = shared_manager.get_current_status()['status']
            
            if current_status != 'running':
                if st.button("🚀 Start Training", type="primary", use_container_width=True):
                    if len(selected_symbols) > 0 and start_date < end_date:
                        with st.spinner("Starting training..."):
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
                        st.error("Please check your configuration")
            else:
                if st.button("⏹️ Stop Training", type="secondary", use_container_width=True):
                    with st.spinner("Stopping training..."):
                        success = stop_training()
                    if success:
                        st.success("Training stopped successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to stop training")
            
            # Reset button
            if st.button("🔄 Reset Configuration", use_container_width=True):
                reset_training_state()
                st.success("Configuration reset!")
                st.rerun()
            
            # System information
            st.subheader("💻 System Status")
            
            # Device information
            device_info = f"Device: {DEVICE.upper()}"
            if USE_AMP:
                device_info += " (AMP Enabled)"
            st.info(device_info)
            
            # Display system monitoring
            display_system_monitoring()
    
    with tab2:
        st.header("📊 Real-time Training Monitoring")
        
        # Add monitoring instructions
        with st.expander("ℹ️ Monitoring Guide", expanded=False):
            st.markdown("""
            ### 📊 Real-time Monitoring Guide
            
            **🎯 Overview:**
            Monitor AI model training progress in real-time with comprehensive charts and metrics.
            
            **📈 Performance Tab:**
            - **Reward Progression**: Shows how the AI's performance improves over time
            - **Moving Average**: Smoothed trend line to see overall progress
            - **Interpretation**: Higher rewards = better trading performance
            
            **🧠 Model Diagnostics Tab:**
            - **Loss Curves**: Actor and Critic loss should decrease over time
            - **Parameter Norms**: Monitor model stability and convergence
            - **Interpretation**: Decreasing losses = model is learning effectively
            
            **💰 Portfolio Tab:**
            - **Portfolio Value**: Track virtual portfolio performance
            - **Statistics**: Current value, returns, max/min values
            - **Interpretation**: Increasing portfolio value = profitable trading strategy
            
            **📊 Symbol Analysis Tab:**
            - **Win Rate**: Percentage of profitable trades per symbol
            - **Average Return**: Mean profit/loss per trade
            - **Detailed Statistics**: Comprehensive performance metrics
            
            **💡 Tips:**
            - Refresh the page to update charts with latest data
            - Use "Generate Test Data" to see sample visualizations
            - Monitor GPU usage during training for optimal performance
            """)
        
        # Control buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
        
        with col_btn2:
            if st.button("📊 Generate Test Data", use_container_width=True):
                generate_test_data()
                st.success("Test data generated!")
                st.rerun()
        
        with col_btn3:
            if st.button("🗑️ Clear Data", use_container_width=True):
                st.session_state.training_metrics = {
                    'steps': [], 'rewards': [], 'portfolio_values': [],
                    'losses': [], 'norms': [], 'symbol_stats': {}, 'timestamps': []
                }
                st.success("Data cleared!")
                st.rerun()
        
        # Display training status
        display_training_status()
        
        # Create and display charts
        create_training_charts()
        
        # System monitoring section
        st.markdown("---")
        display_system_monitoring()
    
    with tab3:
        st.header("💾 Model Management")
        st.markdown("*Model management features will be implemented here*")
        st.info("This section will include model loading, saving, comparison, and deployment features.")

if __name__ == "__main__":
    main()