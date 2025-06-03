# src/common/shared_data_manager.py
"""
共享數據管理器 - 線程安全的訓練數據同步 (使用 multiprocessing.Queue)
用於在訓練線程和Streamlit UI之間安全地共享數據
"""

import threading
import time
# import json # No longer needed for JSON
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
# from pathlib import Path # No longer needed for self.save_path
import numpy as np
import multiprocessing # Added
from queue import Empty as QueueEmptyException # Added for non-blocking get

try:
    from common.config import INITIAL_CAPITAL # LOGS_DIR no longer needed here
    from common.logger_setup import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    INITIAL_CAPITAL = 100000
    # LOGS_DIR = Path("logs") # No longer needed

class SharedTrainingDataManager:
    """
    共享訓練數據管理器
    
    使用 multiprocessing.Manager().Queue 和線程安全的數據結構來在訓練線程/進程和UI線程之間共享數據。
    """
    
    _manager = None # Class variable to hold the multiprocessing.Manager instance    @classmethod
    def get_manager(cls):
        if cls._manager is None:
            cls._manager = multiprocessing.Manager()
        return cls._manager

    def __init__(self, max_metrics=2000, max_trades=10000):
        """
        初始化共享數據管理器
        
        Args:
            max_metrics: 最大保存的訓練指標數量 (用於內部deque)
            max_trades: 最大保存的交易記錄數量 (用於內部deque)
        """
        self.lock = threading.RLock()  # 可重入鎖，主要保護內部狀態和deques
        
        # 訓練狀態
        self.training_status = 'idle'
        self.training_progress = 0.0
        self.training_error = None
        self.stop_requested = False
        self.training_start_time = None
        
        # Store actual initial capital for accurate return calculations
        self.actual_initial_capital = None
        
        # multiprocessing Queues for communication from trainer
        manager = self.get_manager()
        self.metrics_mp_queue = manager.Queue(maxsize=max_metrics * 2) # Larger buffer for mp queue
        self.trades_mp_queue = manager.Queue(maxsize=max_trades * 2)  # Larger buffer for mp queue

        # Internal deques for UI to read from (populated from mp_queues)
        self.metrics_queue = deque(maxlen=max_metrics)
        self.trade_queue = deque(maxlen=max_trades)
        
        # 當前統計數據 (由訓練過程直接更新，也通過鎖保護)
        self.current_metrics = {
            'step': 0,
            'reward': 0.0,
            'portfolio_value': float(INITIAL_CAPITAL),
            'actor_loss': np.nan, # Changed from 0.0
            'critic_loss': np.nan, # Changed from 0.0
            'l2_norm': np.nan, # Changed from 0.0
            'grad_norm': np.nan, # Changed from 0.0
            'timestamp': datetime.now(timezone.utc).isoformat() # Changed to isoformat for consistency
        }
        
        self.symbol_stats = {} # Updated by add_trade_record
        self.performance_stats = { # Updated by add_training_metric
            'total_episodes': 0,
            'total_steps': 0,
            'best_reward': float('-inf'),
            'best_portfolio_value': float(INITIAL_CAPITAL),
            'avg_episode_length': 0,
            'training_efficiency': 0.0
        }
        
        # No more file-based persistence attributes
        # self.save_path = LOGS_DIR / "shared_training_data.json"
        # self.auto_save_interval = 300
        # self.last_save_time = time.time()
        
        logger.info("SharedTrainingDataManager 初始化完成 (使用 multiprocessing.Queue)")
    
    def _pull_data_from_mp_queues(self):
        """Internal method to pull data from multiprocessing queues into internal deques."""
        with self.lock: # Protect access to internal deques
            # Process metrics queue
            while True:
                try:
                    metric = self.metrics_mp_queue.get_nowait()
                    self.metrics_queue.append(metric)
                except QueueEmptyException:
                    break
                except Exception as e:
                    logger.error(f"Error pulling from metrics_mp_queue: {e}", exc_info=False) # Log lightly
                    break 
            
            # Process trades queue
            while True:
                try:
                    trade = self.trades_mp_queue.get_nowait()
                    self.trade_queue.append(trade)
                except QueueEmptyException:
                    break
                except Exception as e:
                    logger.error(f"Error pulling from trades_mp_queue: {e}", exc_info=False) # Log lightly
                    break

    def update_training_status(self, status: str, progress: Optional[float] = None, 
                             error: Optional[str] = None):
        """
        更新訓練狀態
        
        Args:
            status: 訓練狀態 ('idle', 'running', 'completed', 'error')
            progress: 訓練進度 (0-100)
            error: 錯誤信息
        """
        with self.lock:
            old_status = self.training_status
            self.training_status = status
            
            if progress is not None:
                self.training_progress = max(0, min(100, progress))
            
            if error is not None:
                self.training_error = error
            
            # 記錄狀態變化
            if old_status != status:
                if status == 'running':
                    self.training_start_time = datetime.now(timezone.utc)
                    logger.info(f"訓練狀態變更: {old_status} -> {status}")
                elif status in ['completed', 'error']:
                    if self.training_start_time:
                        duration = datetime.now(timezone.utc) - self.training_start_time
                        logger.info(f"訓練狀態變更: {old_status} -> {status}, 持續時間: {duration}")
    
    def request_stop(self):
        """請求停止訓練"""
        with self.lock:
            self.stop_requested = True
            logger.info("收到停止訓練請求")
    
    def is_stop_requested(self) -> bool:
        """檢查是否請求停止"""
        with self.lock:
            return self.stop_requested
    
    def reset_stop_flag(self):
        """重置停止標誌"""
        with self.lock:
            self.stop_requested = False
            logger.debug("停止標誌已重置")
    
    def add_training_metric(self, step: int, reward: float, portfolio_value: float,
                           actor_loss: Optional[float] = None, critic_loss: Optional[float] = None,
                           l2_norm: Optional[float] = None, grad_norm: Optional[float] = None):
        """
        添加訓練指標 - 由訓練過程調用
        Puts data onto metrics_mp_queue and updates current/performance stats.
        """
        metric = {
            'step': step,
            'reward': float(reward),
            'portfolio_value': float(portfolio_value),
            'actor_loss': float(actor_loss) if actor_loss is not None and not np.isnan(actor_loss) else np.nan,
            'critic_loss': float(critic_loss) if critic_loss is not None and not np.isnan(critic_loss) else np.nan,
            'l2_norm': float(l2_norm) if l2_norm is not None and not np.isnan(l2_norm) else np.nan,
            'grad_norm': float(grad_norm) if grad_norm is not None and not np.isnan(grad_norm) else np.nan,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            self.metrics_mp_queue.put_nowait(metric) # Use put_nowait to avoid blocking trainer
        except multiprocessing.queues.Full:
            logger.warning("Metrics multiprocessing queue is full. Metric may be lost.")
        except Exception as e:
            logger.error(f"Error putting to metrics_mp_queue: {e}", exc_info=False)        # Update current and performance stats directly (thread-safe due to GIL or explicit lock if needed)
        with self.lock:
            self.current_metrics = {
                'step': step,
                'reward': float(reward),
                'portfolio_value': float(portfolio_value),
                'actor_loss': float(actor_loss) if actor_loss is not None and not np.isnan(actor_loss) else np.nan,
                'critic_loss': float(critic_loss) if critic_loss is not None and not np.isnan(critic_loss) else np.nan,
                'l2_norm': float(l2_norm) if l2_norm is not None and not np.isnan(l2_norm) else np.nan,
                'grad_norm': float(grad_norm) if grad_norm is not None and not np.isnan(grad_norm) else np.nan,
                'timestamp': metric['timestamp'] # Use the same timestamp
            }
            self.performance_stats['total_steps'] = step
            if reward > self.performance_stats['best_reward']:
                self.performance_stats['best_reward'] = float(reward)
            if portfolio_value > self.performance_stats['best_portfolio_value']:
                self.performance_stats['best_portfolio_value'] = float(portfolio_value)
            # No more auto-save check
            # if time.time() - self.last_save_time > self.auto_save_interval:
            #     self._auto_save() # This method will be removed
    
    def add_trade_record(self, symbol: str, action: str, price: float, 
                        quantity: float, profit_loss: float, 
                        training_step: int, timestamp: Optional[datetime] = None):
        """
        添加交易記錄 - 由訓練過程調用
        Puts data onto trades_mp_queue and updates symbol_stats.
        
        Args:
            symbol: 交易品種
            action: 交易動作 ('buy', 'sell')
            price: 交易價格
            quantity: 交易數量
            profit_loss: 盈虧
            training_step: 訓練步數 (主要時間軸)
            timestamp: 實際時間戳 (輔助信息)
        """
        ts = timestamp or datetime.now(timezone.utc)
        trade = {
            'symbol': symbol,
            'action': action,
            'price': float(price),
            'quantity': float(quantity),
            'profit_loss': float(profit_loss),
            'training_step': int(training_step),  # 新增：訓練步數作為主要時間軸
            'timestamp': ts.isoformat()  # 保留實際時間戳作為輔助信息
        }
        
        try:
            self.trades_mp_queue.put_nowait(trade) # Use put_nowait
        except multiprocessing.queues.Full:
            logger.warning("Trades multiprocessing queue is full. Trade record may be lost.")
        except Exception as e:
            logger.error(f"Error putting to trades_mp_queue: {e}", exc_info=False)

        # Update symbol_stats directly (thread-safe due to GIL or explicit lock)
        with self.lock:
            if symbol not in self.symbol_stats:
                self.symbol_stats[symbol] = {
                    'trades': 0, 'total_profit': 0.0, 'wins': 0, 'losses': 0,
                    'returns': [], 'win_rate': 0.0, 'avg_return': 0.0,
                    'max_return': 0.0, 'max_loss': 0.0, 'sharpe_ratio': 0.0
                }
            
            stats = self.symbol_stats[symbol]
            stats['trades'] += 1
            stats['total_profit'] += float(profit_loss)
            # For simplicity, 'returns' list for sharpe might be better calculated from the trade_queue later if needed
            # For now, keep direct update for basic stats
            if profit_loss > 0: stats['wins'] += 1
            elif profit_loss < 0: stats['losses'] += 1
            if stats['trades'] > 0: stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
            # More complex stats like avg_return, sharpe_ratio might need to pull from the deque/queue

    def get_latest_metrics(self, count: int = 100) -> List[Dict[str, Any]]:
        """獲取最新的訓練指標 - 由 UI 調用"""
        self._pull_data_from_mp_queues() # Ensure internal deques are updated
        with self.lock:
            # Convert ISO string timestamps back to datetime objects for UI if needed
            # For now, assume UI can handle ISO strings or Plotly handles them.
            return list(self.metrics_queue)[-count:]

    def get_latest_trades(self, count: int = 100) -> List[Dict[str, Any]]:
        """獲取最新的交易記錄 - 由 UI 調用"""
        self._pull_data_from_mp_queues() # Ensure internal deques are updated
        with self.lock:
            return list(self.trade_queue)[-count:]

    def get_all_metrics(self) -> List[Dict[str, Any]]: # New method
        """獲取內部隊列中所有可用的訓練指標 - 由 UI 調用"""
        self._pull_data_from_mp_queues() # Ensure internal deques are updated
        with self.lock:
            return list(self.metrics_queue)

    def get_all_trades(self) -> List[Dict[str, Any]]: # New method
        """獲取內部隊列中所有可用的交易記錄 - 由 UI 調用"""
        self._pull_data_from_mp_queues() # Ensure internal deques are updated
        with self.lock:
            return list(self.trade_queue)
    
    def get_metrics_in_range(self, start_step: int, end_step: int) -> List[Dict[str, Any]]:
        """
        獲取指定步數範圍內的訓練指標
        
        Args:
            start_step: 開始步數
            end_step: 結束步數
            
        Returns:
            指定範圍內的訓練指標
        """
        with self.lock:
            return [m for m in self.metrics_queue 
                   if start_step <= m['step'] <= end_step]
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        獲取當前完整狀態
        
        Returns:
            包含所有狀態信息的字典
        """
        with self.lock:
            return {
                'status': self.training_status,
                'progress': self.training_progress,
                'error': self.training_error,
                'stop_requested': self.stop_requested,
                'current_metrics': self.current_metrics.copy(),
                'symbol_stats': {k: v.copy() for k, v in self.symbol_stats.items()},
                'performance_stats': self.performance_stats.copy(),
                'data_counts': {
                    'metrics': len(self.metrics_queue),
                    'trades': len(self.trade_queue),
                    'symbols': len(self.symbol_stats)
                },
                'training_start_time': self.training_start_time
            }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        獲取訓練摘要統計
        
        Returns:
            訓練摘要信息
        """
        with self.lock:
            if not self.metrics_queue:
                return {}
            
            metrics_list = list(self.metrics_queue)
            rewards = [m['reward'] for m in metrics_list]
            portfolio_values = [m['portfolio_value'] for m in metrics_list]
            
            return {
                'total_steps': len(metrics_list),
                'avg_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards),
                'final_portfolio_value': portfolio_values[-1],
                'max_portfolio_value': np.max(portfolio_values),
                'min_portfolio_value': np.min(portfolio_values),
                'total_return': ((portfolio_values[-1] - self.get_actual_initial_capital()) / self.get_actual_initial_capital()) * 100,
                'max_drawdown': self._calculate_max_drawdown(portfolio_values),
                'volatility': np.std(portfolio_values) / np.mean(portfolio_values) * 100
            }
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """計算最大回撤"""
        if len(portfolio_values) < 2:
            return 0.0
        
        peak = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def clear_data(self):
        """清除所有數據"""
        with self.lock:
            # Drain multiprocessing queues
            while not self.metrics_mp_queue.empty():
                try:
                    self.metrics_mp_queue.get_nowait()
                except QueueEmptyException:
                    break
                except Exception: # Catch any other exception during draining
                    break
            while not self.trades_mp_queue.empty():
                try:
                    self.trades_mp_queue.get_nowait()
                except QueueEmptyException:
                    break
                except Exception:
                    break

            self.metrics_queue.clear()
            self.trade_queue.clear()
            self.symbol_stats.clear()
            
            self.training_status = 'idle'
            self.training_progress = 0.0
            self.training_error = None
            self.stop_requested = False
            self.training_start_time = None
            
            # Reset actual initial capital
            self.actual_initial_capital = None
            
            self.current_metrics = {
                'step': 0, 'reward': 0.0, 'portfolio_value': float(INITIAL_CAPITAL),
                'actor_loss': np.nan, # Changed
                'critic_loss': np.nan, # Changed
                'l2_norm': np.nan, # Changed
                'grad_norm': np.nan, # Changed
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            self.performance_stats = {
                'total_episodes': 0, 'total_steps': 0, 'best_reward': float('-inf'),
                'best_portfolio_value': float(INITIAL_CAPITAL), 'avg_episode_length': 0,
                'training_efficiency': 0.0
            }
            logger.info("共享數據已清除 (包括 multiprocessing Queues)")

    def set_actual_initial_capital(self, initial_capital: float):
        """
        Set the actual initial capital used by the trainer
        
        Args:
            initial_capital: The actual initial capital amount
        """
        with self.lock:
            self.actual_initial_capital = initial_capital
            logger.info(f"Set actual initial capital: {initial_capital}")
    
    def get_actual_initial_capital(self) -> float:
        """
        Get the actual initial capital, falling back to config value if not set
        
        Returns:
            The actual initial capital or config default
        """
        with self.lock:
            return self.actual_initial_capital if self.actual_initial_capital is not None else float(INITIAL_CAPITAL)

# 全局共享數據管理器實例
_global_shared_manager_instance = None # Renamed for clarity
_manager_init_lock = threading.Lock() # Lock for initializing the manager instance

def get_shared_data_manager() -> SharedTrainingDataManager:
    """
    獲取全局共享數據管理器實例（單例模式）
    Ensures that the multiprocessing.Manager is also handled as a singleton if needed.
    """
    global _global_shared_manager_instance
    with _manager_init_lock: # Protect the instantiation of the manager and the singleton
        if _global_shared_manager_instance is None:
            # The SharedTrainingDataManager class now handles its own mp.Manager
            _global_shared_manager_instance = SharedTrainingDataManager()
            logger.info("創建全局 SharedTrainingDataManager 實例 (with mp.Manager)")
        return _global_shared_manager_instance

def reset_shared_data_manager():
    """重置全局共享數據管理器"""
    global _global_shared_manager_instance
    
    with _manager_init_lock:
        if _global_shared_manager_instance is not None:
            _global_shared_manager_instance.clear_data()
            logger.info("全局共享數據管理器已重置")