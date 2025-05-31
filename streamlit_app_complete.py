#!/usr/bin/env python3
"""
OANDA AI Trading Model - Complete Streamlit Application
Enhanced real-time monitoring with proper data synchronization and GPU monitoring
"""

# Fix PyTorch Streamlit compatibility issue
import os
import sys

# Set environment variables before importing torch
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'

# Suppress TensorFlow warnings more aggressively
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
os.environ['TF_DISABLE_DEPRECATED_WARNINGS'] = '1'  # Disable deprecated warnings
os.environ['TF_ENABLE_DEPRECATED_WARNINGS'] = '0'  # Disable deprecated warnings

# Additional warnings suppression
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*Session state does not function.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*tf.reset_default_graph.*')

# Fix for PyTorch 2.7.0+ Streamlit compatibility
try:
    import torch
    # Disable torch._classes module path inspection by Streamlit
    if hasattr(torch, '_classes'):
        torch._classes.__path__ = []
except:
    pass

import streamlit as st

# --- BEGINNING OF SCRIPT - 路徑設置 ---
import sys
from pathlib import Path

# 確保項目模組可以被找到 - 統一路徑設置
def setup_project_path():
    """Set up the Python path for the project.""" # Translated
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

project_root = setup_project_path()
# --- END OF 路徑設置 ---

from src.common.logger_setup import logger # This will run logger_setup.py

# --- Session State Initialization Flag ---
# Moved to init_session_state() function

# Disable file watcher to prevent high CPU usage on file change
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
import psutil
import logging
from collections import defaultdict
from src.data_manager.oanda_downloader import manage_data_download_for_symbols # This import should now be more reliable
import asyncio # Add asyncio import
import re # Ensure re is imported at the top

# Try to import GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Suppress TensorFlow warnings immediately after imports
try:
    import tensorflow as tf
    # Disable TensorFlow deprecated warnings using the new API
    tf.get_logger().setLevel('ERROR')
    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # Additional warning suppression
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
except ImportError:
    pass
except Exception:
    pass

# Try to import trainer, use fallback if failed
try:
    from src.trainer.universal_trainer import UniversalTrainer as EnhancedUniversalTrainer, create_training_time_range
    # logger is already imported globally
    from src.common.config import ACCOUNT_CURRENCY, INITIAL_CAPITAL, DEVICE, USE_AMP, MAX_ACCOUNT_RISK_PERCENTAGE, ATR_STOP_LOSS_MULTIPLIER, MAX_POSITION_SIZE_PERCENTAGE_OF_EQUITY, MAX_SYMBOLS_ALLOWED
    from src.common.shared_data_manager import get_shared_data_manager
    from src.data_manager.instrument_info_manager import InstrumentInfoManager
    TRAINER_AVAILABLE = True
    # Removed logger.info("Successfully imported trainer and shared data manager") to reduce noise on re-runs
except ImportError as e:
    logger.error(f"Failed to import trainer modules: {e}", exc_info=True)
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
                    'timestamp': kwargs.get('timestamp', datetime.now())                }
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
                    'training_step': kwargs.get('training_step', 0),  # Added: training step
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

# Note: The 'Others' category has been renamed to 'Bonds' as requested by the user
# since all symbols in that category were identified as bond instruments.

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
        'Bonds': []
    }
    # Major pairs list (OANDA standard)
    major_pairs = {
        'EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CHF', 'USD_CAD', 'NZD_USD'
    }
    
    # 特定符号的明确分类映射 (根据用户要求调整)
    # 以下符号按照用户指定分类：
    # - 指数类：NL25_EUR(荷兰AEX), CH20_CHF(瑞士SMI), ESPIX_EUR(西班牙IBEX), 
    #          US2000_USD(美国小盘股), CHINAH_HKD(中国H股), SG30_SGD(新加坡海峡时报), FR40_EUR(法国CAC40)
    # - 能源类：BCO_USD(布伦特原油)
    # - 贵金属类：XCU_USD(铜)
    specific_classifications = {
        'NL25_EUR': 'Indices',      # 荷兰AEX指数
        'CH20_CHF': 'Indices',      # 瑞士SMI指数
        'ESPIX_EUR': 'Indices',     # 西班牙IBEX指数
        'BCO_USD': 'Energy',        # 布伦特原油
        'XCU_USD': 'Precious Metals', # 铜 - 归类到贵金属
        'US2000_USD': 'Indices',    # 美国小盘股指数
        'CHINAH_HKD': 'Indices',    # 中国H股指数
        'SG30_SGD': 'Indices',      # 新加坡海峡时报指数
        'FR40_EUR': 'Indices'       # 法国CAC40指数
    }
    
    # Indices, Energy, Metals, Commodities keywords
    index_keywords = ["SPX", "NAS", "US30", "UK100", "DE30", "JP225", "HK33", "AU200", "FRA40", "EU50", "CN50", 
                     "NL25", "CH20", "ESPIX", "US2000", "CHINAH", "SG30", "FR40"]
    energy_keywords = ["OIL", "WTICO", "BRENT", "NATGAS", "GAS", "BCO"]
    metal_keywords = ["XAU", "XAG", "GOLD", "SILVER", "PLAT", "PALL", "XCU"]
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
        # 首先检查特定符号的明确分类
        if symbol in specific_classifications:
            category = specific_classifications[symbol]
            categorized[category].append((symbol, display, t))
        elif symbol in major_pairs:
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
            categorized['Bonds'].append((symbol, display, t))
    # Remove empty categories, but always keep 'Bonds' if any uncategorized
    categorized = {k: v for k, v in categorized.items() if v}
    return categorized

def init_session_state():
    """Initialize all session state variables if they don't exist."""
    # Only initialize if we're running in Streamlit context
    if not hasattr(st, 'session_state'):
        return
    
    # Set app_initialized flag for first-time initialization
    if 'app_initialized' not in st.session_state:
        logger.info("Streamlit App: First time initialization of session state.")
        st.session_state.app_initialized = True
        # Other truly one-time initializations for the entire session can go here.
    
    # Initialize shared data manager
    if 'shared_data_manager' not in st.session_state:
        st.session_state.shared_data_manager = get_shared_data_manager()
        logger.info("Initialized shared_data_manager in session state.")
    
    # training_status should reflect the source of truth, which is shared_data_manager
    # Set a default if shared_data_manager hasn't populated it yet.
    if 'training_status' not in st.session_state:
        if hasattr(st.session_state, 'shared_data_manager') and st.session_state.shared_data_manager:
            st.session_state.training_status = st.session_state.shared_data_manager.get_current_status().get('status', 'idle')
        else:
            st.session_state.training_status = 'idle' # Absolute fallback
    
    if 'training_thread' not in st.session_state:
        st.session_state.training_thread = None
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 5
    if 'total_timesteps' not in st.session_state: # For ETA calculation
        st.session_state.total_timesteps = 0

    # 新增：UI 優化相關的 session state
    if 'last_data_update' not in st.session_state:
        st.session_state.last_data_update = 0
    if 'update_cache' not in st.session_state:
        st.session_state.update_cache = {}
    if 'chart_display_mode' not in st.session_state:
        st.session_state.chart_display_mode = 'full'  # 'full', 'lite', 'minimal'

    # Flag to indicate this function has run for the current session setup
    st.session_state.session_state_initialized = True
    logger.debug("Session state initialized/verified.")

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
            actor_loss=actor_loss,            critic_loss=critic_loss,
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
                training_step=step  # 添加訓練步數參數
            )
        
        if step % 100 == 0:
            logger.debug(f"Simulation training progress: {progress:.1f}%")
            
        time.sleep(0.002)
        
    shared_manager.update_training_status('completed', 100)
    logger.info("Simulation training completed.")
    return True

def training_worker(trainer_instance, shared_manager, symbols, total_timesteps): # Renamed arg for clarity
    """Training worker thread - uses shared data manager"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        logger.info(f"Training worker thread started. Trainer instance: {trainer_instance}, Symbols: {symbols}, Steps: {total_timesteps}")
        
        success = False  # 初始化 success 變量
        if trainer_instance and TRAINER_AVAILABLE:
            logger.info("Starting real training process with trainer instance.")
            # Ensure the trainer uses the correct shared manager
            if hasattr(trainer_instance, 'set_shared_data_manager'):
                 trainer_instance.set_shared_data_manager(shared_manager)
            elif hasattr(trainer_instance, 'shared_data_manager'): # Or assign directly if it's a public attribute
                 trainer_instance.shared_data_manager = shared_manager
            else:
                 logger.warning("Trainer instance does not have a method/attribute to set shared_data_manager.")

            success = trainer_instance.run_full_training_pipeline()
        else:
            logger.info("No valid trainer instance or TRAINER_NOT_AVAILABLE. Starting simulation training process.")
            success = simulate_training_with_shared_manager(shared_manager, symbols, total_timesteps)
        
        if shared_manager.is_stop_requested():
            shared_manager.update_training_status('idle')
            logger.info("Training worker: Stop request processed. Status set to idle.")
        elif success:
            shared_manager.update_training_status('completed', 100)
            logger.info("Training worker: Process completed successfully.")
        else:
            # If success is False but no specific error was set by the trainer, set a generic one.
            if shared_manager.get_current_status()['status'] != 'error':
                 shared_manager.update_training_status('error', error="Training did not complete successfully (trainer returned False).")
            logger.warning("Training worker: Process did not complete successfully.")
            
    except Exception as e:
        logger.error(f"Error in training worker: {e}", exc_info=True)
        shared_manager.update_training_status('error', error=str(e))
    finally:
        logger.info("Training worker: Finalizing.")
        if trainer_instance and hasattr(trainer_instance, 'cleanup'):
            logger.info("Training worker: Calling trainer.cleanup()")
            trainer_instance.cleanup()
        loop.close()
        logger.info("Training worker: Asyncio loop closed. Thread finishing.")

def start_training(symbols, start_date, end_date, total_timesteps, save_freq, eval_freq):
    """Start training with enhanced error handling and session state management."""
    
    # Ensure session state is initialized (though main() should handle this first)
    if not st.session_state.get('session_state_initialized', False):
        init_session_state()

    shared_manager = st.session_state.shared_data_manager
    
    if st.session_state.get('training_thread') is not None and st.session_state.training_thread.is_alive():
        st.warning("Training is already in progress.")
        logger.warning("Attempted to start training while a training thread is already active.")
        return False

    logger.info(f"Attempting to start new training session for symbols: {symbols}, total_timesteps: {total_timesteps}")
    shared_manager.clear_data() 
    shared_manager.reset_stop_flag()
    # Reset session-specific step tracking
    st.session_state.initial_global_step_of_session = None
    st.session_state.first_metric_received = False
    logger.info("Initialized initial_global_step_of_session and first_metric_received for new session.")

    shared_manager.update_training_status('starting', 0)

    try:
        start_time_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_time_dt = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        
        trainer_instance_for_thread = None 
        if TRAINER_AVAILABLE:
            logger.info("TRAINER_AVAILABLE is True. Creating new EnhancedUniversalTrainer instance.")
            # Create a new trainer instance for this training session
            current_trainer = EnhancedUniversalTrainer(
                trading_symbols=symbols,
                start_time=start_time_dt,
                end_time=end_time_dt,
                granularity="S5", 
                total_timesteps=total_timesteps,
                save_freq=save_freq,
                eval_freq=eval_freq,
                model_name_prefix="sac_universal_trader",
                # The trainer should ideally get the shared_manager via a method or init arg
                # For now, we pass it to the worker, which can set it if the trainer supports it.
            )
            st.session_state.trainer = current_trainer # Store the new trainer in session state
            trainer_instance_for_thread = current_trainer
            logger.info(f"New EnhancedUniversalTrainer instance created and stored in session_state.trainer: {current_trainer}")
        else:
            st.session_state.trainer = None 
            logger.warning("TRAINER_AVAILABLE is False. Real training will not occur; simulation will be used by worker.")
            trainer_instance_for_thread = None # Pass None to worker for simulation

        training_thread = threading.Thread(
            target=training_worker,
            args=(trainer_instance_for_thread, shared_manager, symbols, total_timesteps),
            daemon=True
        )
        training_thread.start()
        st.session_state.training_thread = training_thread 
        
        shared_manager.update_training_status('running', 0) 
        logger.info(f"Training thread started. Thread ID: {training_thread.ident}. Trainer for thread: {trainer_instance_for_thread}")
        st.session_state.total_timesteps = total_timesteps 
        return True
        
    except Exception as e:
        st.error(f"Failed to start training: {e}")
        logger.error(f"Critical error during training setup: {e}", exc_info=True)
        shared_manager.update_training_status('error', error=f"Setup failed: {str(e)}")
        
        # Cleanup attempt for trainer if it was created
        if st.session_state.get('trainer') is not None:
             if hasattr(st.session_state.trainer, 'cleanup'):
                 try:
                     st.session_state.trainer.cleanup()
                     logger.info("Cleaned up trainer instance after setup failure.")
                 except Exception as cleanup_e:
                     logger.error(f"Error during trainer cleanup after setup failure: {cleanup_e}", exc_info=True)
        st.session_state.trainer = None # Ensure it's cleared
        st.session_state.training_thread = None # Ensure thread is cleared
        return False

def stop_training():
    """Stop training with proper cleanup and session state management."""
    logger.info("Attempting to stop training via stop_training function.")
    
    if not st.session_state.get('session_state_initialized', False):
        init_session_state() # Should not be strictly necessary if main() calls it, but safe

    shared_manager = st.session_state.shared_data_manager
    shared_manager.request_stop() 
    logger.info("Stop request sent to shared_data_manager.")

    trainer_to_stop = st.session_state.get('trainer') 
    if trainer_to_stop:
        logger.info(f"Found trainer instance in session state to stop: {trainer_to_stop}")
        # Prefer a method in trainer that handles its own stop logic, including saving
        if hasattr(trainer_to_stop, 'stop_training_process'): 
            logger.info("Calling trainer.stop_training_process() if available.")
            try:
                trainer_to_stop.stop_training_process() # This method should ideally handle saving model etc.
            except Exception as e_stop_trainer:
                logger.error(f"Error calling trainer.stop_training_process(): {e_stop_trainer}", exc_info=True)
        elif hasattr(trainer_to_stop, 'stop'): # Fallback
             logger.info("Calling trainer.stop() as fallback.")
             try:
                 trainer_to_stop.stop()
             except Exception as e_stop_trainer_fallback:
                 logger.error(f"Error calling trainer.stop() fallback: {e_stop_trainer_fallback}", exc_info=True)
        
        # Explicit save call if not handled by stop_training_process or if needed as a guarantee
        if hasattr(trainer_to_stop, 'save_current_model'):
            logger.info("Attempting to save current model via trainer.save_current_model().")
            try:
                trainer_to_stop.save_current_model()
                logger.info("Model save explicitly called and completed.")
            except Exception as e_save:
                logger.error(f"Error calling trainer.save_current_model(): {e_save}", exc_info=True)
    else:
        logger.warning("No trainer instance found in session state during stop_training call.")

    training_thread_to_join = st.session_state.get('training_thread')
    if training_thread_to_join and training_thread_to_join.is_alive():
        logger.info(f"Training thread {training_thread_to_join.ident} is alive. Attempting to join with 10s timeout.")
        training_thread_to_join.join(timeout=10.0) 
        if training_thread_to_join.is_alive():
            logger.warning(f"Training thread {training_thread_to_join.ident} did not join after 10 seconds. It might be stuck or unresponsive.")
        else:
            logger.info(f"Training thread {training_thread_to_join.ident} successfully joined.")
    elif training_thread_to_join:
        logger.info(f"Training thread {training_thread_to_join.ident} was found but is not alive.")
    else:
        logger.info("No active training thread found to join.")
    
    current_sm_status = shared_manager.get_current_status()
    if current_sm_status['status'] not in ['error', 'completed']:
        shared_manager.update_training_status('idle') # Set to idle if not already error/completed
        logger.info("Shared manager status updated to 'idle' after stop sequence.")
    else:
        logger.info(f"Shared manager status is '{current_sm_status['status']}', not changing to 'idle'.")
    
    # 清除所有共享數據，包括訓練指標、交易記錄和圖表數據
    shared_manager.clear_data()
    logger.info("All shared training data, metrics, and trade records have been cleared.")
    
    # 清理訓練相關的 session state 變量，確保 UI 完全重置到初始狀態
    training_session_keys_to_reset = [
        'training_thread',
        'trainer', 
        'first_metric_received',
        'initial_global_step_of_session',
        'total_timesteps'
    ]
    
    for key in training_session_keys_to_reset:
        if key in st.session_state:
            old_value = st.session_state[key]
            st.session_state[key] = None if key in ['training_thread', 'trainer'] else False if key == 'first_metric_received' else 0
            logger.info(f"Reset session state key '{key}' from {old_value} to {st.session_state[key]}")

    # 特別處理 trainer 的 cleanup
    if st.session_state.get('trainer') is not None: # Double check after reset
        if hasattr(st.session_state.trainer, 'cleanup'):
            logger.info("Calling trainer.cleanup() as part of stop_training finalization.")
            try:
                st.session_state.trainer.cleanup()
            except Exception as e_cleanup:
                logger.error(f"Error during trainer.cleanup() in stop_training: {e_cleanup}", exc_info=True)
        st.session_state.trainer = None # Ensure it's cleared
        logger.info("Trainer instance cleared from session state.")
    
    logger.info("stop_training function execution completed with full UI state reset.")
    return True # Assume success in signaling stop, actual stop depends on thread.

def create_real_time_charts():
    """Create real-time training monitoring charts with smart updating"""
    shared_manager = st.session_state.shared_data_manager
    update_manager = get_update_manager()
    
    # 更新訓練狀態到智能管理器
    current_status = shared_manager.get_current_status()
    update_manager.update_training_state(current_status.get('status', 'idle'))
    
    # 檢查是否需要更新圖表
    chart_mode = st.session_state.get('chart_display_mode', 'full')
    
    # 根據模式決定數據量
    if chart_mode == 'minimal':
        max_points = 50
    elif chart_mode == 'lite':
        max_points = 100
    else:
        max_points = 500
    
    all_metrics = shared_manager.get_latest_metrics(max_points)
    
    if not all_metrics:
        st.info("No training data available. Start training to view real-time charts.")
        return
    
    df = pd.DataFrame(all_metrics)
      # Determine initial global step for session-relative plotting
    initial_global_step = st.session_state.get('initial_global_step_of_session')
    if initial_global_step is None: # If not captured yet (e.g., very early, or no training started)
        # Try to infer from the data - use the minimum step as the session start
        if 'step' in df.columns and not df.empty:
            initial_global_step = df['step'].min()
            st.session_state['initial_global_step_of_session'] = initial_global_step
        else:
            initial_global_step = 0


    x_axis_column = 'step' # Default to global step
    if 'step' in df.columns and not df.empty:
        df['session_step'] = df['step'] - initial_global_step
        x_axis_column = 'session_step' # Prefer session step for plotting
        # Ensure session_step is non-negative, can happen if initial_global_step captured late
        df['session_step'] = df['session_step'].clip(lower=0)


    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure required columns are present and handle potential NaN from initialization
    for col in ['actor_loss', 'critic_loss', 'l2_norm', 'grad_norm']:
        if col not in df.columns:
            df[col] = np.nan # Add column with NaNs if missing
        # else:
            # df[col] = pd.to_numeric(df[col], errors='coerce') # Ensure numeric, coerce errors to NaN

    # Filter out rows where essential data for plotting might be NaN, 
    # but be careful not to filter too aggressively if NaNs are expected (e.g. initial steps)
    # For losses and norms, we want to plot them even if they start as NaN and then get values.
    # Plotly handles NaNs by skipping those points, which is usually desired.

    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
        "📈 Performance", "🧠 Model Diagnostics", "💰 Portfolio", "📊 Trading Activity"
    ])
    
    with chart_tab1:
        st.subheader("Training Performance")
        
        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(
            x=df[x_axis_column], # Use determined x-axis column
            y=df['reward'],
            mode='lines+markers',
            name='Reward',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=3)
        ))
        fig_reward.update_layout(
            title="Training Reward Progression",
            xaxis_title="Training Steps (Session)", # X-axis label updated
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
                x=df[x_axis_column], # Use determined x-axis column
                y=df['reward'],
                mode='lines',
                name='Raw Reward',
                line=dict(color='lightblue', width=1),
                opacity=0.5
            ))
            fig_ma.add_trace(go.Scatter(
                x=df[x_axis_column], # Use determined x-axis column
                y=df['reward_ma'],
                mode='lines',
                name=f'Moving Average ({window_size})',
                line=dict(color='red', width=3)
            ))
            fig_ma.update_layout(
                title=f"Reward Trend Analysis",
                xaxis_title="Training Steps (Session)", # X-axis label updated
                yaxis_title="Reward",
                height=400
            )
            st.plotly_chart(fig_ma, use_container_width=True)
    
    with chart_tab2:
        st.subheader("Model Diagnostics")
        
        fig_loss = go.Figure()
        if 'actor_loss' in df.columns and df['actor_loss'].notna().any():
            fig_loss.add_trace(go.Scatter(
                x=df[x_axis_column], # Use determined x-axis column
                y=df['actor_loss'],
                mode='lines',
                name='Actor Loss',
                line=dict(color='orange', width=2)
            ))
        if 'critic_loss' in df.columns and df['critic_loss'].notna().any():
            fig_loss.add_trace(go.Scatter(
                x=df[x_axis_column], # Use determined x-axis column
                y=df['critic_loss'],
                mode='lines',
                name='Critic Loss',
                line=dict(color='green', width=2)
            ))
        fig_loss.update_layout(
            title="Training Loss Curves",
            xaxis_title="Training Steps (Session)", # X-axis label updated
            yaxis_title="Loss",
            yaxis_type="log", # Consider making this conditional or providing a toggle if losses can be zero/negative
            height=400
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # 添加Transformer範數圖表
        fig_transformer_norms = go.Figure()
        if 'l2_norm' in df.columns and df['l2_norm'].notna().any():
            fig_transformer_norms.add_trace(go.Scatter(
                x=df[x_axis_column], # Use determined x-axis column
                y=df['l2_norm'],
                mode='lines',
                name='Transformer L2 Norm',
                line=dict(color='purple', width=2)
            ))
        if 'grad_norm' in df.columns and df['grad_norm'].notna().any():
            fig_transformer_norms.add_trace(go.Scatter(
                x=df[x_axis_column], # Use determined x-axis column
                y=df['grad_norm'],
                mode='lines',
                name='Transformer Gradient Norm',
                line=dict(color='red', width=2)
            ))
        fig_transformer_norms.update_layout(
            title="Transformer Model Norms",
            xaxis_title="Training Steps (Session)", # X-axis label updated
            yaxis_title="Norm Value",
            height=400
        )
        st.plotly_chart(fig_transformer_norms, use_container_width=True)
    
    with chart_tab3:
        st.subheader("Portfolio Performance")
        
        fig_portfolio = go.Figure()
        fig_portfolio.add_trace(go.Scatter(
            x=df[x_axis_column], # Use determined x-axis column
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
            xaxis_title="Training Steps (Session)", # X-axis label updated
            yaxis_title=f"Portfolio Value ({ACCOUNT_CURRENCY})", # Using ACCOUNT_CURRENCY
            hovermode='x unified',            height=400
        )
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        if not df.empty:
            current_value = df['portfolio_value'].iloc[-1]
            total_return = ((current_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            max_value = df['portfolio_value'].max()
            min_value = df['portfolio_value'].min()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Value", f"{ACCOUNT_CURRENCY} {current_value:,.2f}") # Using ACCOUNT_CURRENCY
            with col2:
                st.metric("Total Return", f"{total_return:+.2f}%")
            with col3:
                st.metric("Max Value", f"{ACCOUNT_CURRENCY} {max_value:,.2f}") # Using ACCOUNT_CURRENCY
            with col4:
                st.metric("Min Value", f"{ACCOUNT_CURRENCY} {min_value:,.2f}") # Using ACCOUNT_CURRENCY
    
    with chart_tab4:
        st.subheader("Trading Activity")
        
        # Use get_all_trades() instead of get_latest_trades()
        all_trades = shared_manager.get_all_trades()
        
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            if 'timestamp' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
              # Create sub-tabs for different trading activity views
            trade_tab1, trade_tab2, trade_tab3 = st.tabs([
                "📈 Trades Over Time", "📊 P&L Distribution", "🎯 Symbol Activity"
            ])
            
            with trade_tab1:
                st.subheader("Trades Over Training Steps")
                # Check if we have training_step data
                if 'training_step' in trades_df.columns and trades_df['training_step'].notna().any():
                    # Convert training steps to session-relative if needed
                    initial_global_step = st.session_state.get('initial_global_step_of_session')
                    
                    # Debug information
                    # st.write(f"Debug: initial_global_step from session_state: {initial_global_step}")
                    # st.write(f"Debug: training_step range in trades: {trades_df['training_step'].min()} - {trades_df['training_step'].max()}")
                    
                    if initial_global_step is None or initial_global_step == 0:
                        # Use the minimum training step from trades as session start
                        initial_global_step = trades_df['training_step'].min()
                        st.session_state['initial_global_step_of_session'] = initial_global_step
                        # st.write(f"Debug: Set initial_global_step to trades minimum: {initial_global_step}")
                    
                    # Calculate session steps with proper handling of edge cases
                    trades_df['session_step'] = trades_df['training_step'] - initial_global_step
                    trades_df['session_step'] = trades_df['session_step'].clip(lower=0)
                    
                    # Additional debug information
                    # st.write(f"Debug: session_step range: {trades_df['session_step'].min()} - {trades_df['session_step'].max()}")
                    
                    # If all session_steps are still 0, use raw training steps
                    if trades_df['session_step'].max() == 0 and trades_df['training_step'].max() > 0:
                        # st.warning("Session steps are all 0, using raw training steps instead")
                        trades_df['session_step'] = trades_df['training_step']
                    
                    # 檢查是否有足夠的數據繪製圖表
                    if len(trades_df) < 2:
                        st.info("⏳ Waiting for more trade data to display timeline...")
                    else:
                        # Create scatter plot of trades over training steps
                        fig_trades_timeline = go.Figure()
                    
                    # Color code by action type
                    buy_trades = trades_df[trades_df['action'].str.startswith('Long', na=False)]
                    sell_trades = trades_df[trades_df['action'].str.startswith('Short', na=False)]
                    
                    if not buy_trades.empty:
                        fig_trades_timeline.add_trace(go.Scatter(
                            x=buy_trades['session_step'],
                            y=buy_trades['profit_loss'] if 'profit_loss' in buy_trades.columns else [0] * len(buy_trades),
                            mode='markers',
                            name='Buy Trades',
                            marker=dict(
                                color='green',
                                size=8,
                                symbol='triangle-up'
                            ),
                            text=buy_trades['symbol'] + '<br>Price: $' + buy_trades['price'].round(4).astype(str) +
                                 '<br>Quantity: ' + buy_trades['quantity'].round(2).astype(str),
                            hovertemplate='<b>%{text}</b><br>' +
                                        'Training Step: %{x}<br>' +
                                        f'P&L: {ACCOUNT_CURRENCY} %{{y:.2f}}<extra></extra>' # Using ACCOUNT_CURRENCY
                        ))
                    
                    if not sell_trades.empty:
                        fig_trades_timeline.add_trace(go.Scatter(
                            x=sell_trades['session_step'],
                            y=sell_trades['profit_loss'] if 'profit_loss' in sell_trades.columns else [0] * len(sell_trades),
                            mode='markers',
                            name='Sell Trades',
                            marker=dict(
                                color='red',
                                size=8,
                                symbol='triangle-down'
                            ),
                            text=sell_trades['symbol'] + '<br>Price: $' + sell_trades['price'].round(4).astype(str) +
                                 '<br>Quantity: ' + sell_trades['quantity'].round(2).astype(str),
                            hovertemplate='<b>%{text}</b><br>' +
                                        'Training Step: %{x}<br>' +
                                        f'P&L: {ACCOUNT_CURRENCY} %{{y:.2f}}<extra></extra>' # Using ACCOUNT_CURRENCY
                        ))
                        
                        # Add horizontal line at zero P&L
                        fig_trades_timeline.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        
                        fig_trades_timeline.update_layout(
                            title="Trade Execution Timeline (by Training Steps)",
                            xaxis_title="Training Steps (Session)",
                            yaxis_title=f"Profit/Loss per Trade ({ACCOUNT_CURRENCY})", # Using ACCOUNT_CURRENCY
                            hovermode='closest',
                            height=500,
                            showlegend=True
                        )
                        st.plotly_chart(fig_trades_timeline, use_container_width=True)
                    
                    # Cumulative P&L over training steps
                    if 'profit_loss' in trades_df.columns:
                        trades_df_sorted = trades_df.sort_values('session_step')
                        trades_df_sorted['cumulative_pnl'] = trades_df_sorted['profit_loss'].cumsum()
                        
                        fig_cumulative = go.Figure()
                        fig_cumulative.add_trace(go.Scatter(
                            x=trades_df_sorted['session_step'],
                            y=trades_df_sorted['cumulative_pnl'],
                            mode='lines+markers',
                            name='Cumulative P&L',
                            line=dict(color='blue', width=2),
                            marker=dict(size=4),
                            fill='tonexty'
                        ))
                        
                        fig_cumulative.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        
                        fig_cumulative.update_layout(
                            title="Cumulative Trading P&L Over Training Steps",
                            xaxis_title="Training Steps (Session)",
                            yaxis_title=f"Cumulative P&L ({ACCOUNT_CURRENCY})", # Using ACCOUNT_CURRENCY
                            hovermode='x unified',
                            height=400
                        )
                        st.plotly_chart(fig_cumulative, use_container_width=True)
                        
                        # Trading activity frequency
                        trade_frequency = trades_df_sorted.groupby('session_step').size()
                        if len(trade_frequency) > 1:
                            fig_frequency = go.Figure()
                            fig_frequency.add_trace(go.Bar(
                                x=trade_frequency.index,
                                y=trade_frequency.values,
                                name='Trades per Step',
                                marker_color='orange',
                                opacity=0.7
                            ))
                            
                            fig_frequency.update_layout(
                                title="Trading Activity Frequency",
                                xaxis_title="Training Steps (Session)",
                                yaxis_title="Number of Trades",
                                height=350
                            )
                            st.plotly_chart(fig_frequency, use_container_width=True)
                    
                else:
                    st.info("Training step information not available. Using timestamp-based timeline instead.")
                    if 'timestamp' in trades_df.columns and trades_df['timestamp'].notna().any():
                        # Fallback to timestamp-based visualization
                        fig_trades_time = go.Figure()
                        
                        buy_trades = trades_df[trades_df['action'] == 'buy']
                        sell_trades = trades_df[trades_df['action'] == 'sell']
                        
                        if not buy_trades.empty:
                            fig_trades_time.add_trace(go.Scatter(
                                x=buy_trades['timestamp'],
                                y=buy_trades['profit_loss'] if 'profit_loss' in buy_trades.columns else [0] * len(buy_trades),
                                mode='markers',
                                name='Buy Trades',
                                marker=dict(color='green', size=8, symbol='triangle-up')
                            ))
                        
                        if not sell_trades.empty:
                            fig_trades_time.add_trace(go.Scatter(
                                x=sell_trades['timestamp'],
                                y=sell_trades['profit_loss'] if 'profit_loss' in sell_trades.columns else [0] * len(sell_trades),
                                mode='markers',
                                name='Sell Trades',
                                marker=dict(color='red', size=8, symbol='triangle-down')
                            ))
                        
                        fig_trades_time.update_layout(
                            title="Trade Execution Timeline (by Timestamp)",
                            xaxis_title="Time",
                            yaxis_title=f"Profit/Loss per Trade ({ACCOUNT_CURRENCY})", # Using ACCOUNT_CURRENCY
                            height=400
                        )
                        st.plotly_chart(fig_trades_time, use_container_width=True)
                    else:
                        st.info("No timestamp data available for timeline visualization.")
            
            with trade_tab2:
                st.subheader("Profit & Loss Analysis")
                
                # Ensure profit_loss column exists for histogram
                if 'profit_loss' in trades_df.columns and trades_df['profit_loss'].notna().any():
                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Histogram(
                        x=trades_df['profit_loss'],
                        nbinsx=30,
                        name='P&L Distribution',
                        marker_color='lightblue'
                    ))
                    fig_pnl.update_layout(
                        title="Profit & Loss Distribution",
                        xaxis_title=f"Profit/Loss ({ACCOUNT_CURRENCY})", # Using ACCOUNT_CURRENCY
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)
                    
                    # P&L statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total P&L", f"{ACCOUNT_CURRENCY} {trades_df['profit_loss'].sum():.2f}") # Using ACCOUNT_CURRENCY
                    with col2:
                        st.metric("Average P&L", f"{ACCOUNT_CURRENCY} {trades_df['profit_loss'].mean():.2f}") # Using ACCOUNT_CURRENCY
                    with col3:
                        winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
                        total_trades = len(trades_df)
                        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    with col4:
                        st.metric("Total Trades", f"{total_trades}")
                else:
                    st.info("No profit and loss data to display for trades.")
            
            with trade_tab3:
                st.subheader("Symbol Activity Analysis")
                
                if 'symbol' in trades_df.columns and trades_df['symbol'].notna().any():
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
                    
                    # Symbol P&L breakdown if available
                    if 'profit_loss' in trades_df.columns:
                        symbol_pnl = trades_df.groupby('symbol')['profit_loss'].agg(['sum', 'mean', 'count']).round(2)
                        symbol_pnl.columns = ['Total P&L', 'Avg P&L', 'Trade Count']
                        symbol_pnl = symbol_pnl.sort_values('Total P&L', ascending=False)
                        
                        st.subheader("P&L by Symbol")
                        st.dataframe(symbol_pnl, use_container_width=True)
                        
                        # Symbol P&L pie chart
                        positive_pnl = symbol_pnl[symbol_pnl['Total P&L'] > 0]
                        if not positive_pnl.empty:
                            fig_symbol_pnl = go.Figure(data=[
                                go.Pie(labels=positive_pnl.index, 
                                       values=positive_pnl['Total P&L'],
                                       title="Profitable Symbols Distribution")
                            ])
                            fig_symbol_pnl.update_layout(height=400)
                            st.plotly_chart(fig_symbol_pnl, use_container_width=True)
                else:
                    st.info("No symbol data available for analysis.")
        else:
            st.info("No trading data available yet.")

# 新增：智能更新管理器
class SmartUpdateManager:
    """Smart Update Manager - Intelligently adjusts update frequency based on training status and data changes.""" # Translated
    
    def __init__(self):
        self.last_update_time = {}
        self.update_intervals = {
            'metrics': 2,      # Critical metrics every 2s
            'charts': 5,       # Charts every 5s
            'system': 10,      # System resources every 10s
            'logs': 15         # Logs every 15s
        }
        self.training_state = 'idle'
        
    def should_update(self, component: str) -> bool:
        """Determine if the specified component should be updated.""" # Translated
        current_time = time.time()
        
        # 根據訓練狀態調整更新頻率
        if self.training_state == 'running':
            multiplier = 1.0
        elif self.training_state in ['starting', 'stopping']:
            multiplier = 0.5  # 更頻繁的更新
        else:
            multiplier = 2.0  # 較少的更新
            
        interval = self.update_intervals.get(component, 5) * multiplier
        last_update = self.last_update_time.get(component, 0)
        
        if current_time - last_update >= interval:
            self.last_update_time[component] = current_time
            return True
        return False
    
    def update_training_state(self, state: str):
        """Update the training state.""" # Translated
        self.training_state = state

# 全局更新管理器實例
@st.cache_resource
def get_update_manager():
    return SmartUpdateManager()

def display_training_status():
    """Display current training status with enhanced information"""
    shared_manager = st.session_state.shared_data_manager
    status_info = shared_manager.get_current_status()
    status = status_info['status']
    progress = status_info['progress']
    error = status_info['error']
    current_metrics = status_info['current_metrics'] # This contains global_step

    current_global_step = current_metrics.get('step', 0)  # Total steps of the model (cumulative mileage)

    # Capture initial global step for the current session
    if status == 'running' and not st.session_state.get('first_metric_received', False) and hasattr(st.session_state, 'shared_data_manager'):
        # This condition ensures we capture the first step reported by the trainer for this session
        # current_global_step here is the first step value received from the trainer after clear_data()
        st.session_state.initial_global_step_of_session = current_global_step
        st.session_state.first_metric_received = True
        logger.info(f"Captured initial global step for session: {st.session_state.initial_global_step_of_session}")

    initial_global_step = st.session_state.get('initial_global_step_of_session', 0)
    
    # Calculate current session steps (starting from 1, not cumulative)
    if initial_global_step is None or initial_global_step == 0:
        if status == 'running' and current_global_step > 0:
            # If training is ongoing but initial step wasn't recorded, set to current step
            st.session_state.initial_global_step_of_session = current_global_step
            initial_global_step = current_global_step
            logger.info(f"Late capture of initial global step for session: {initial_global_step}")
        else:
            initial_global_step = 0

    # Training steps for the current session (steps completed in this session)
    current_session_steps = max(0, current_global_step - initial_global_step)
    session_target_steps = st.session_state.get('total_timesteps', 0)

    # --- Training speed and ETA calculation ---
    steps_per_sec = None
    eta_text = None
    if status == 'running' and current_global_step > initial_global_step: # Ensure some progress in global steps
        metrics_for_speed_calc = shared_manager.get_latest_metrics(20) 
        if len(metrics_for_speed_calc) >= 2:
            global_steps_from_deque = [m['step'] for m in metrics_for_speed_calc]
            # Ensure timestamps are datetime objects
            times_from_deque = [] # Initialize to empty list
            try:
                times_from_deque = [m['timestamp'] for m in metrics_for_speed_calc]
            except (KeyError, TypeError):
                st.warning("Could not retrieve valid timestamps for speed calculation.") # Log or handle error appropriately

            if len(times_from_deque) >= 2: # Check if populated successfully and has enough elements
                if isinstance(times_from_deque[0], str):
                    try:
                        times_from_deque = [pd.to_datetime(t) for t in times_from_deque]
                    except Exception as e:
                        st.warning(f"Error converting timestamps to datetime: {e}")
                        times_from_deque = [] # Reset if conversion fails

            if len(times_from_deque) >= 2 and len(global_steps_from_deque) >=2: # Ensure both have enough data
                dt = (times_from_deque[-1] - times_from_deque[0]).total_seconds()
                d_global_steps = global_steps_from_deque[-1] - global_steps_from_deque[0]

                if dt > 0 and d_global_steps > 0:
                    steps_per_sec = d_global_steps / dt
                
                # Calculate session steps left based on current session progress
                session_steps_left = session_target_steps - current_session_steps
                
                if steps_per_sec > 0 and session_steps_left > 0:
                    eta_sec = int(session_steps_left / steps_per_sec)
                    h, m, s = eta_sec // 3600, (eta_sec % 3600) // 60, eta_sec % 60
                    eta_text = f"ETA: {h:02d}:{m:02d}:{s:02d}"

    if status == 'running':
        # Calculate progress based on session steps
        session_progress_percentage = (current_session_steps / session_target_steps * 100) if session_target_steps > 0 else 0
        session_progress_percentage = max(0, min(100, session_progress_percentage))

        st.success(f"🚀 Training in Progress - {session_progress_percentage:.1f}% Complete (Session)")
        st.progress(session_progress_percentage / 100)
        
        if current_metrics and current_global_step >= initial_global_step: # Check if current_metrics is populated
            # Display key metrics in columns with clear distinction
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                # 模型總步數（累積里程）- 模型自創建以來的總訓練步數
                st.metric("Model Total Steps", f"{current_global_step:,}", help="Total number of training steps the model has undergone since its creation (cumulative mileage).")
            
            with col2:
                # 當前會話步數 - 本次訓練會話的步數，從0開始計數
                step_display_text = f"{current_session_steps:,} / {session_target_steps:,}"
                st.metric("Current Session Steps", step_display_text, help="Progress of training steps in the current session, counting from 0 towards the session target.")
            
            with col3:
                st.metric("Latest Reward", f"{current_metrics['reward']:.3f}")
            
            with col4:
                st.metric("Portfolio Value", f"{ACCOUNT_CURRENCY} {current_metrics['portfolio_value']:,.2f}") # Using ACCOUNT_CURRENCY
            
            with col5:
                # 顯示Transformer L2範數
                l2_norm_val = current_metrics.get('l2_norm', float('nan'))
                st.metric("Transformer L2 Norm", f"{l2_norm_val:.4f}")
            
            with col6:
                # 顯示梯度範數
                grad_norm_val = current_metrics.get('grad_norm', float('nan'))
                st.metric("Gradient Norm", f"{grad_norm_val:.4f}")
        
        if steps_per_sec:
            st.info(f"Training Speed: {steps_per_sec:.2f} steps/sec")
        if eta_text:
            st.info(eta_text)
        
    elif status == 'completed':
        st.success("✅ Training Completed Successfully!")
        if current_metrics: # current_metrics still holds the last global state
            final_session_steps = current_global_step - initial_global_step
            st.info(f"Final Session Steps: {final_session_steps:,} | Model Total Steps: {current_global_step:,} | Final Portfolio: {ACCOUNT_CURRENCY} {current_metrics['portfolio_value']:,.2f}") # Using ACCOUNT_CURRENCY
        
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

# --- Fragment-based UI Updates for Better Performance ---

@st.fragment(run_every=2)
def update_critical_training_metrics():
    """Update critical training metrics in real-time - updates every 2 seconds""" # Translated
    try:
        if 'shared_data_manager' not in st.session_state:
            return
        
        shared_manager = st.session_state.shared_data_manager
        current_status = shared_manager.get_current_status()
        
        # 只在訓練進行時更新，減少不必要的計算
        if current_status.get('status') not in ['running', 'starting']:
            return
        
        # 創建指標顯示
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_step = current_status.get('current_metrics', {}).get('step', 0) # This should be global_step from metrics
            total_steps = st.session_state.get('total_timesteps', 0) # This is session target
            
            # Calculate session steps for display
            initial_global_step = st.session_state.get('initial_global_step_of_session', 0)
            session_current_steps = max(0, current_step - initial_global_step)

            st.metric(
                "Training Steps (Session)", 
                f"{session_current_steps:,}",
                delta=f"/ {total_steps:,}" if total_steps > 0 else None,
                help="Current steps in this training session out of the target steps for this session. This shows how much of the current training job is complete."
            )
        
        with col2:
            portfolio_value = current_status.get('current_metrics', {}).get('portfolio_value', INITIAL_CAPITAL) # Use INITIAL_CAPITAL as default
            initial_cap_for_delta = INITIAL_CAPITAL # Ensure this is the correct initial capital for comparison
            st.metric(
                "Portfolio Value", 
                f"{ACCOUNT_CURRENCY} {portfolio_value:,.2f}", # Using ACCOUNT_CURRENCY
                delta=f"{((portfolio_value / initial_cap_for_delta - 1) * 100):+.2f}%" if initial_cap_for_delta > 0 and portfolio_value != initial_cap_for_delta else None,
                help=f"Current value of the trading portfolio in {ACCOUNT_CURRENCY} and its percentage change from the initial capital. This reflects the overall performance."
            )
        
        with col3:
            status = current_status.get('status', 'idle')
            status_color = "🟢" if status == 'running' else "🟡" if status == 'starting' else "🔴" if status == 'error' else "⏸️" # Added error and idle differentiation
            st.metric("Training Status", f"{status_color} {status.upper()}", help="Current operational status of the training process: Running, Starting, Error, Idle, or Completed.")
        
        with col4:
            # Progress should be based on session steps vs session target
            session_progress = (session_current_steps / total_steps * 100) if total_steps > 0 else 0
            session_progress = max(0, min(100, session_progress)) # Clamp between 0 and 100
            st.metric("Session Progress", f"{session_progress:.1f}%", help="Percentage of training steps completed in the current session. This provides a visual of how far along the current training is.")
            
    except Exception as e:
        logger.error(f"Fragment update error: {e}")

@st.fragment(run_every=5)
def update_performance_charts():
    """Update performance charts - updates every 5 seconds""" # Translated
    try:
        if 'shared_data_manager' not in st.session_state:
            return
        
        shared_manager = st.session_state.shared_data_manager
        current_status = shared_manager.get_current_status()
        
        # 只在有數據時更新圖表
        if current_status.get('status') not in ['running', 'starting']:
            return
            
        # 只更新圖表數據，不重新渲染整個頁面
        latest_metrics = shared_manager.get_latest_metrics(50)  # 減少數據量
        
        if latest_metrics and len(latest_metrics) > 1:
            df = pd.DataFrame(latest_metrics)
            
            # 使用更輕量的圖表更新
            if not df.empty and 'step' in df.columns:
                # 只顯示關鍵指標的簡化圖表
                # Determine initial global step for session-relative plotting
                initial_global_step = st.session_state.get('initial_global_step_of_session')
                if initial_global_step is None: # If not captured yet
                    if 'step' in df.columns and not df.empty:
                        initial_global_step = df['step'].min()
                    else:
                        initial_global_step = 0 # Fallback
                
                df['session_step'] = df['step'] - initial_global_step
                df['session_step'] = df['session_step'].clip(lower=0) # Ensure non-negative
                
                chart_data = df.set_index('session_step')[['reward', 'portfolio_value']].tail(30)  # Display only the last 30 points
                st.line_chart(chart_data, height=250)
                
    except Exception as e:
        logger.error(f"Chart fragment update error: {e}")

@st.fragment(run_every=10)
def update_system_resources():
    """Update system resource monitoring - updates every 10 seconds""" # Translated
    try:
        system_info = get_system_info()
        if not system_info:
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            if system_info.get('gpu'):
                for gpu in system_info['gpu']:
                    st.metric(
                        f"GPU {gpu['id']} Usage",
                        f"{gpu['load']:.1f}%",
                        delta=f"Memory: {gpu['memory_percent']:.1f}%",
                        help=f"GPU Name: {gpu['name']}. Shows current utilization (load) and video memory (VRAM) usage percentage. Helps monitor GPU performance and bottlenecks."
                    )
        
        with col2:
            memory = system_info.get('memory', {})
            if memory:
                st.metric(
                    "System Memory",
                    f"{memory['percent']:.1f}%",
                    delta=f"{memory['used']:.1f}GB / {memory['total']:.1f}GB",
                    help="Overall system RAM usage. Shows percentage used, amount used in GB, and total available RAM. Useful for identifying memory constraints."
                )
                
    except Exception as e:
        logger.error(f"System resource fragment update error: {e}")

def main():
    """Main application function"""
    
    # Initialize session state ONCE at the beginning of main if not already done.
    # The 'app_initialized' flag handles the very first run of the script in a session.
    # init_session_state() ensures all necessary keys are present.
    if not st.session_state.get('session_state_initialized', False):
       init_session_state()
       logger.info("main(): Called init_session_state() because 'session_state_initialized' was not True.")
    
    st.title("🚀 OANDA AI Trading Model")
    st.markdown("**Enhanced Real-time Trading Monitor with GPU Support**")
    
    if not TRAINER_AVAILABLE:
        st.warning("⚠️ Running in simulation mode - trainer modules not available. Full functionality may be limited.")
        # logger.info("Main: TRAINER_AVAILABLE is False.") # Logged once at import usually
    else:
        st.success("✅ All modules loaded successfully. System is ready for training.")
        # logger.info("Main: TRAINER_AVAILABLE is True.")
    
    with st.sidebar:
        st.header("⚙️ Training Configuration")
        
        st.subheader("Trading Symbols")
        # --- Dynamic OANDA symbol selection UI ---
        categorized = get_categorized_symbols_and_details()
        selected_symbols = []
        for cat, symbols in sorted(categorized.items()):
            with st.expander(f"{cat} ({len(symbols)})", expanded=(cat=="CURRENCY")):
                options = [f"{sym} - {display}" for sym, display, _ in symbols] # Ensure f-string is correctly formatted
                default = [options[0]] if cat=="CURRENCY" and options and options[0] in options else [] # Default to first currency for convenience, ensure default is valid
                selected = st.multiselect(
                    f"Select {cat} symbols:",
                    options,
                    default=default,
                    help=f"Choose the financial instruments from the {cat} category for the trading model. You can select multiple symbols. The model will be trained on the selected instruments."
                )
                # Map back to symbol codes
                for sel in selected:
                    code = sel.split(" - ")[0]
                    selected_symbols.append(code)
        # Limit to MAX_SYMBOLS_ALLOWED
        from src.common.config import MAX_SYMBOLS_ALLOWED
        # --- 新增：顯示選取狀態 ---
        st.markdown(f"<span style='font-size:16px;'>Max <b style='color:#0072C6;'>{MAX_SYMBOLS_ALLOWED}</b> symbols can be selected. Currently selected: <b style='color:{'red' if len(selected_symbols)>MAX_SYMBOLS_ALLOWED else '#009900'};'>{len(selected_symbols)}</b>.</span>", unsafe_allow_html=True) # Translated
        if len(selected_symbols) > MAX_SYMBOLS_ALLOWED:
            st.warning(f"You selected {len(selected_symbols)} symbols, but only {MAX_SYMBOLS_ALLOWED} are allowed. Truncating to the first {MAX_SYMBOLS_ALLOWED}.") # Enhanced warning
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
                help="Select the beginning date for the historical data used in training. The model will learn from data starting from this date. Ensure this date is before the End Date.",
                format="DD/MM/YYYY"
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=datetime.now().date() - timedelta(days=1),
                help="Select the end date for the historical training data. The data up to this date (exclusive of the current day) will be used. This date must be after the Start Date.",
                format="DD/MM/YYYY"
            )
        total_timesteps = st.number_input(
            "Total Training Steps",
            min_value=1000,
            max_value=1000000,
            value=50000,
            step=1000,
            help="Define the total number of iterations or steps the training process will run for. A higher number generally means more learning but will take longer to complete."
        )
        # --- Custom training parameters ---
        col1, col2 = st.columns(2)
        with col1:
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=1000.0,
                max_value=1e8,
                value=float(INITIAL_CAPITAL),
                step=1000.0,                help="Set the initial amount of virtual capital for the simulated trading account (e.g., 100000). This is the starting equity for backtesting and training sessions."
            )
            risk_pct = st.number_input(
                "Max Risk % per Trade",
                min_value=0.1,
                max_value=100.0,
                value=float(MAX_ACCOUNT_RISK_PERCENTAGE * 100),  # Convert to percentage
                step=0.1,
                format="%.1f",
                help="Specify the maximum percentage (e.g., 1.0 for 1%) of the total account equity that can be risked on any single trade. This helps in managing overall portfolio risk."
            )
            # Convert back to decimal for internal use
            risk_pct = risk_pct / 100
        with col2:
            atr_mult = st.number_input(
                "ATR Stop Multiplier",
                min_value=0.5,
                max_value=10.0,
                value=float(ATR_STOP_LOSS_MULTIPLIER),
                step=0.1,                help="Set the multiplier for the Average True Range (ATR) to determine stop-loss levels (e.g., 2.5). A larger multiplier means a wider stop loss, potentially reducing premature exits but increasing risk per trade."
            )
            max_pos_pct = st.number_input(
                "Max Position Size %",
                min_value=0.1,
                max_value=100.0,
                value=float(MAX_POSITION_SIZE_PERCENTAGE_OF_EQUITY * 100),  # Convert to percentage                step=0.1,
                format="%.1f",
                help="Define the maximum percentage (e.g., 5.0 for 5%) of account equity that can be allocated to a single position. This controls exposure to any one instrument."
            )
            # Convert back to decimal for internal use
            max_pos_pct = max_pos_pct / 100
        # --- Save/Eval frequency controls ---
        col1, col2 = st.columns(2)
        with col1:
            save_freq = st.number_input(
                "Save Frequency",
                min_value=100,
                max_value=10000,
                value=2000,
                step=100,
                help="Determine how frequently (in training steps, e.g., every 2000 steps) the current state of the model should be saved to disk. Frequent saves allow resuming training but can take up disk space."
            )
        with col2:
            eval_freq = st.number_input(
                "Eval Frequency",
                min_value=100,
                max_value=10000,
                value=5000,
                step=100,
                help="Specify how often (in training steps, e.g., every 5000 steps) the model should be evaluated on a separate validation dataset (if configured). Evaluation provides insights into generalization."
            )
        
        st.subheader("Training Controls")
        
               
        # Ensure shared_manager is available, init_session_state should have created it.
        shared_manager = st.session_state.get('shared_data_manager')
        if not shared_manager:
            st.error("Critical Error: Shared Data Manager not found in session state. Please refresh.")
            logger.error("CRITICAL: shared_data_manager not found in session_state in main().")
            return # Stop further rendering if this happens

        current_status_dict = shared_manager.get_current_status()
        current_status = current_status_dict.get('status', 'idle')
        
        col1, col2 = st.columns(2)
        
        with col1:
            if current_status not in ['running', 'starting']: # Check for 'starting' as well
                if st.button("🚀 Start Training", type="primary", use_container_width=True):
                    if not selected_symbols:
                        st.error("Please select at least one trading symbol before starting.")
                    elif start_date >= end_date:
                        st.error("Start date must be before end date. Please adjust the dates.")
                    else:
                        # --- Data download progress bar before training ---
                        logger.info("main(): 'Start Training' button clicked. Proceeding with data download and start_training.")
                        download_data_with_progress(selected_symbols, start_date, end_date)
                        # Pass the actual initial_capital, risk_pct etc. to start_training if they need to override config
                        # For now, assuming EnhancedUniversalTrainer uses values from config.py or its defaults
                        success = start_training(
                            selected_symbols, start_date, end_date, 
                            total_timesteps, save_freq, eval_freq
                        )
                        if success:
                            st.success("Training start initiated successfully! Monitor progress below.")
                            logger.info("main(): start_training call returned True. Rerunning Streamlit.")
                            st.rerun()
                        else:
                            st.error("Failed to initiate training. Please check the logs for more details.")
                            logger.warning("main(): start_training call returned False.")
            else:
                st.button("🚀 Start Training", disabled=True, use_container_width=True, help="Training is currently active or in the process of starting. Please wait or stop the current session to start a new one.")
        
        with col2:
            if current_status in ['running', 'starting']: # Check for 'starting' as well
                if st.button("⏹️ Stop Training", type="secondary", use_container_width=True):
                    logger.info("main(): 'Stop Training' button clicked. Calling stop_training.")
                    success = stop_training()
                    if success: # stop_training now always returns True, success means signal sent
                        st.info("Stop signal sent. Training will halt shortly and data will be reset.")
                        logger.info("main(): stop_training call completed. Rerunning Streamlit.")
                        st.rerun() # Rerun to update UI based on new state
                    # else: # stop_training doesn't return False in the new logic for failure to stop
                    # st.error("Failed to send stop signal properly.")
            else:
                st.button("⏹️ Stop Training", disabled=True, use_container_width=True, help="No active training session to stop. Start training first to enable this button.")
        
        st.subheader("Auto Refresh")
        auto_refresh = st.checkbox("Enable Auto Refresh", value=st.session_state.auto_refresh, help="Automatically refresh the dashboard at a set interval to see live updates. Disable for manual control or to reduce system load if experiencing performance issues.")
        st.session_state.auto_refresh = auto_refresh
        
        # 新增：顯示模式控制
        st.subheader("Display Settings")
        chart_mode = st.selectbox(
            "Chart Display Mode",
            options=['full', 'lite', 'minimal'],
            index=['full', 'lite', 'minimal'].index(st.session_state.get('chart_display_mode', 'full')),
            help="Adjust the detail level of charts. 'Full' shows all details and plot types. 'Lite' offers a balanced view with fewer points or simplified charts. 'Minimal' provides the most lightweight display, suitable for performance on slower systems or connections."
        )
        st.session_state.chart_display_mode = chart_mode
        
        # 智能更新控制
        enable_smart_updates = st.checkbox(
            "Enable Smart Updates", 
            value=st.session_state.get('enable_smart_updates', True),
            help="Dynamically adjust data refresh frequency based on training status. This means faster updates during active training or startup/shutdown phases, and slower updates when the system is idle or has completed training, optimizing performance."
        )
        st.session_state.enable_smart_updates = enable_smart_updates
        
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh Interval (seconds)",
                min_value=1,
                max_value=30,
                value=st.session_state.refresh_interval,
                help="Set how often (in seconds) the data and charts on the dashboard should refresh. This setting is active only if 'Enable Auto Refresh' is checked above."
            )
            st.session_state.refresh_interval = refresh_interval
        
        if st.button("🔄 Manual Refresh", use_container_width=True):
            st.rerun()
    
    # Main content area

    display_training_status()
    
    # 添加性能監控指示器
    if st.session_state.get('enable_smart_updates', True):
        update_manager = get_update_manager()
        with st.expander("🔧 Performance Monitor", expanded=False): # Title already in English
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Chart Mode", st.session_state.get('chart_display_mode', 'full').upper(), help="Current display mode for charts (Full, Lite, or Minimal). This affects the amount of data shown and chart complexity.")
            
            with col2:
                current_status_info = shared_manager.get_current_status() # Renamed to avoid conflict
                training_status_display = current_status_info.get('status', 'idle') # Renamed
                st.metric("Training Status", training_status_display.upper(), help="Live status of the model training process (e.g., Running, Idle, Error, Completed).")
            
            with col3:
                # Display current effective refresh interval
                base_interval = st.session_state.refresh_interval
                effective_interval_info = f"{base_interval}s"
                
                if st.session_state.get('enable_smart_updates', True):
                    if training_status_display == 'running':
                        effective_interval_info += " (Normal)"
                    elif training_status_display in ['starting', 'stopping']:
                        effective_interval_info = f"{max(1, base_interval // 2)}s (Fast)" # Show actual fast interval
                    else: # idle, completed, error
                        effective_interval_info = f"{base_interval * 2}s (Slow)" # Show actual slow interval
                st.metric("Effective Refresh", effective_interval_info, help="Current dashboard refresh rate. This rate adapts if 'Enable Smart Updates' is active, becoming faster during critical operations and slower when idle.")
      # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Real-time Charts", "💻 System Monitor", "📋 Training Data"]) # Changed "Training Logs" to "Training Data"
    
    with tab1:
        create_real_time_charts()
    
    with tab2:
        display_system_monitoring()
    
    with tab3:
        st.subheader("📋 Recent Trades") # MOVED UP, Title translated
        shared_manager = st.session_state.shared_data_manager
        latest_trades = shared_manager.get_latest_trades(50) # Increased number of trades to fetch
        
        if latest_trades:
            trades_df = pd.DataFrame(latest_trades)
            
            # Ensure trades_df_display is defined from trades_df before styling
            trades_df_display = trades_df.copy()

            # Remove timestamp column if it exists
            if 'timestamp' in trades_df_display.columns:
                trades_df_display = trades_df_display.drop(columns=['timestamp'])

            # Rename 'profit_loss' to 'Total_PnL' to match the expected column name for styling
            if 'profit_loss' in trades_df_display.columns:
                trades_df_display.rename(columns={'profit_loss': 'Total_PnL'}, inplace=True)
              # Apply currency prefix and rounding to 'Total_PnL' if the column exists
            if 'Total_PnL' in trades_df_display.columns:
                trades_df_display['Total_PnL'] = trades_df_display['Total_PnL'].apply(
                    lambda x: f"{ACCOUNT_CURRENCY} $ {x:.2f}" if isinstance(x, (int, float)) else x
                )
            # Apply rounding to 'Price' if the column exists
            if 'Price' in trades_df_display.columns:
                trades_df_display['Price'] = trades_df_display['Price'].apply(
                    lambda x: f"{x:.5f}" if isinstance(x, (int, float)) else x # Assuming price needs more precision
                )

            TARGET_FONT_SIZE_PX = '24px'            # Helper function for PnL column styling (color, background, and font size)
            def format_and_color_pnl(val_str):
                style_rules = [f'font-size: {TARGET_FONT_SIZE_PX}', 'font-weight: bold']
                color = 'black' # Default color
                background_color = 'white' # Default background
                try:
                    # Extract numeric value for color determination
                    match = re.search(r'([-+]?\d*\.?\d+)', str(val_str))
                    if match:
                        numeric_val = float(match.group(1))
                        if numeric_val >= 0:
                            color = 'green'
                            background_color = '#d4edda'  # Light green background for profit
                        else:
                            color = 'red'
                            background_color = '#f8d7da'  # Light red background for loss
                except ValueError:
                    pass # If float conversion fails, color remains black
                style_rules.append(f'color: {color}')
                style_rules.append(f'background-color: {background_color}')
                return '; '.join(style_rules)            # Helper function for general column styling (font size only)
            def general_font_style(_): # Value not used, but map expects a function taking one arg
                return f'font-size: {TARGET_FONT_SIZE_PX};'

            # Check if trades_df_display is not empty before styling
            if not trades_df_display.empty:
                styled_df = trades_df_display.style # Initialize Styler object

                # Apply PnL specific styling if the column exists
                if 'Total_PnL' in trades_df_display.columns:
                    styled_df = styled_df.map(
                        format_and_color_pnl, subset=['Total_PnL']
                    )

                # Apply general font styling to all other columns
                for col_name in trades_df_display.columns:
                    if col_name != 'Total_PnL': # Only apply general style if not the P&L column (or if P&L doesn't exist)
                        styled_df = styled_df.map(general_font_style, subset=[col_name])
                
                st.dataframe(styled_df, height=600, use_container_width=True)
            elif 'trades_df' in locals() and trades_df.empty: # If original df was empty
                 st.info("No trade records available yet.")
            else: # If trades_df_display somehow became empty but trades_df wasn't (should not happen with copy)
                 st.info("No trade records to display after formatting.")

        else:
            st.info("No trade records available yet.")

        st.divider() # Visual separator        st.subheader("📋 Training Metrics Log") # MOVED DOWN, Title translated
        latest_metrics = shared_manager.get_latest_metrics(50) 
        if latest_metrics:
            df_metrics = pd.DataFrame(latest_metrics) 
            df_metrics = df_metrics.sort_values('step', ascending=False)

            # Convert global steps to session steps for display
            initial_global_step = st.session_state.get('initial_global_step_of_session', 0)
            if initial_global_step is None:
                initial_global_step = 0
            
            # Add session step column
            df_metrics['session_step'] = df_metrics['step'] - initial_global_step
            df_metrics['session_step'] = df_metrics['session_step'].clip(lower=0)  # Ensure non-negative

            if 'portfolio_value' in df_metrics.columns:
                df_metrics['portfolio_value_formatted'] = df_metrics['portfolio_value'].apply(lambda x: f"{ACCOUNT_CURRENCY} {x:,.2f}")
            
            metric_display_cols = {
                'session_step': 'Step',  # Use session_step instead of step
                'reward': 'Reward',
                'portfolio_value_formatted': 'Portfolio Value', # Use formatted column
                'actor_loss': 'Actor Loss',
                'critic_loss': 'Critic Loss'
            }            # Ensure only existing columns are selected and renamed
            cols_to_display = [col for col in metric_display_cols.keys() if col in df_metrics.columns or col == 'portfolio_value_formatted']
            df_metrics_display = df_metrics[cols_to_display].rename(columns=metric_display_cols)

            st.dataframe(
                df_metrics_display,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No training metrics log available yet. Metrics will populate as training progresses.")
        
    # Auto refresh functionality with smart intervals
    if st.session_state.auto_refresh:
        update_manager = get_update_manager()
        
        # 根據訓練狀態動態調整刷新間隔
        current_status = shared_manager.get_current_status()
        training_status = current_status.get('status', 'idle')
        
        if st.session_state.get('enable_smart_updates', True):
            if training_status == 'running':
                refresh_interval = st.session_state.refresh_interval
            elif training_status in ['starting', 'stopping']:
                refresh_interval = max(1, st.session_state.refresh_interval // 2)  # 更頻繁
            else:
                refresh_interval = st.session_state.refresh_interval * 2  # 較少刷新
        else:
            refresh_interval = st.session_state.refresh_interval
            
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    # Optional: Add a specific log for direct script run if needed
    # logger.info(f"Executing streamlit_app_complete.py as __main__.")
    main()