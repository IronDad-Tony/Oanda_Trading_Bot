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
    """設置項目的 Python 路徑"""
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

project_root = setup_project_path()
# --- END OF 路徑設置 ---

from src.common.logger_setup import logger # This will run logger_setup.py

# --- Session State Initialization Flag ---
# This helps ensure that expensive or critical one-time initializations
# for the session are managed correctly.
if 'app_initialized' not in st.session_state:
    logger.info("Streamlit App: First time initialization of session state flag.")
    st.session_state.app_initialized = True
    # Other truly one-time initializations for the entire session can go here.

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

# Try to import GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Try to import trainer, use fallback if failed
try:
    from src.trainer.universal_trainer import UniversalTrainer as EnhancedUniversalTrainer, create_training_time_range
    # logger is already imported globally
    from src.common.config import ACCOUNT_CURRENCY, INITIAL_CAPITAL, DEVICE, USE_AMP
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
    """Initialize all session state variables if they don't exist."""
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
        st.session_state.auto_refresh = False
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 5
    if 'total_timesteps' not in st.session_state: # For ETA calculation
        st.session_state.total_timesteps = 0

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

def training_worker(trainer_instance, shared_manager, symbols, total_timesteps): # Renamed arg for clarity
    """Training worker thread - uses shared data manager"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        logger.info(f"Training worker thread started. Trainer instance: {trainer_instance}, Symbols: {symbols}, Steps: {total_timesteps}")
        
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
    
    st.session_state.training_thread = None # Clear thread from session state
    if st.session_state.get('trainer') is not None: # If a trainer instance exists
        if hasattr(st.session_state.trainer, 'cleanup'):
            logger.info("Calling trainer.cleanup() as part of stop_training finalization.")
            try:
                st.session_state.trainer.cleanup()
            except Exception as e_cleanup:
                logger.error(f"Error during trainer.cleanup() in stop_training: {e_cleanup}", exc_info=True)
        st.session_state.trainer = None # Clear trainer from session state
        logger.info("Trainer instance cleared from session state.")
    
    logger.info("stop_training function execution completed.")
    return True # Assume success in signaling stop, actual stop depends on thread.

def create_real_time_charts():
    """Create real-time training monitoring charts"""
    shared_manager = st.session_state.shared_data_manager
    latest_metrics = shared_manager.get_latest_metrics(200)
    
    if not latest_metrics:
        st.info("No training data available. Start training to view real-time charts.")
        return
    
    df = pd.DataFrame(latest_metrics)
    
    if 'timestamp' in df.columns:
        # Ensure timestamp is parsed from ISO string to datetime objects
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
            if 'timestamp' in trades_df.columns:
                # Ensure timestamp is parsed from ISO string to datetime objects
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
        metrics_for_speed_calc = shared_manager.get_latest_metrics(20) # Use a different variable name
        if len(metrics_for_speed_calc) >= 2:
            steps = [m['step'] for m in metrics_for_speed_calc]
            times = [m['timestamp'] for m in metrics_for_speed_calc]
            # Ensure timestamp is parsed from ISO string to datetime objects
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
    
    # Initialize session state ONCE at the beginning of main if not already done.
    # The 'app_initialized' flag handles the very first run of the script in a session.
    # init_session_state() ensures all necessary keys are present.
    if not st.session_state.get('session_state_initialized', False):
       init_session_state()
       logger.info("main(): Called init_session_state() because 'session_state_initialized' was not True.")
    
    st.title("🚀 OANDA AI Trading Model")
    st.markdown("**Enhanced Real-time Trading Monitor with GPU Support**")
    
    if not TRAINER_AVAILABLE:
        st.warning("⚠️ Running in simulation mode - trainer modules not available")
        # logger.info("Main: TRAINER_AVAILABLE is False.") # Logged once at import usually
    else:
        st.success("✅ All modules loaded successfully")
        # logger.info("Main: TRAINER_AVAILABLE is True.")
    
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
                        st.error("Please select at least one trading symbol")
                    elif start_date >= end_date:
                        st.error("Start date must be before end date")
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
                            st.success("Training start initiated successfully!")
                            logger.info("main(): start_training call returned True. Rerunning Streamlit.")
                            st.rerun()
                        else:
                            st.error("Failed to initiate training. Check logs.")
                            logger.warning("main(): start_training call returned False.")
            else:
                st.button("🚀 Start Training", disabled=True, use_container_width=True, help="Training is currently active or starting.")
        
        with col2:
            if current_status in ['running', 'starting']: # Check for 'starting' as well
                if st.button("⏹️ Stop Training", type="secondary", use_container_width=True):
                    logger.info("main(): 'Stop Training' button clicked. Calling stop_training.")
                    success = stop_training()
                    if success: # stop_training now always returns True, success means signal sent
                        st.info("Stop signal sent. Training will halt shortly.")
                        logger.info("main(): stop_training call completed. Rerunning Streamlit.")
                        st.rerun() # Rerun to update UI based on new state
                    # else: # stop_training doesn't return False in the new logic for failure to stop
                    # st.error("Failed to send stop signal properly.")
            else:
                st.button("⏹️ Stop Training", disabled=True, use_container_width=True, help="No active training to stop.")
        
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
    # Optional: Add a specific log for direct script run if needed
    # logger.info(f"Executing streamlit_app_complete.py as __main__.")
    main()