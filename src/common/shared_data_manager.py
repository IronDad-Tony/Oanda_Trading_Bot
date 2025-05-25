# src/common/shared_data_manager.py
"""
共享數據管理器 - 線程安全的訓練數據同步
用於在訓練線程和Streamlit UI之間安全地共享數據
"""

import threading
import time
import json
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np

try:
    from common.config import INITIAL_CAPITAL, LOGS_DIR
    from common.logger_setup import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    INITIAL_CAPITAL = 100000
    LOGS_DIR = Path("logs")

class SharedTrainingDataManager:
    """
    共享訓練數據管理器
    
    使用線程安全的數據結構來在訓練線程和UI線程之間共享數據
    避免直接訪問Streamlit的session_state導致的ScriptRunContext問題
    """
    
    def __init__(self, max_metrics=2000, max_trades=10000):
        """
        初始化共享數據管理器
        
        Args:
            max_metrics: 最大保存的訓練指標數量
            max_trades: 最大保存的交易記錄數量
        """
        self.lock = threading.RLock()  # 使用可重入鎖
        
        # 訓練狀態
        self.training_status = 'idle'  # idle, running, completed, error
        self.training_progress = 0.0
        self.training_error = None
        self.stop_requested = False
        self.training_start_time = None
        
        # 使用deque作為線程安全的序列，自動限制大小
        self.metrics_queue = deque(maxlen=max_metrics)
        self.trade_queue = deque(maxlen=max_trades)
        
        # 當前統計數據
        self.current_metrics = {
            'step': 0,
            'reward': 0.0,
            'portfolio_value': float(INITIAL_CAPITAL),
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'l2_norm': 0.0,
            'grad_norm': 0.0,
            'timestamp': datetime.now(timezone.utc)
        }
        
        # 交易品種統計
        self.symbol_stats = {}
        
        # 性能統計
        self.performance_stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'best_reward': float('-inf'),
            'best_portfolio_value': float(INITIAL_CAPITAL),
            'avg_episode_length': 0,
            'training_efficiency': 0.0
        }
        
        # 持久化設置
        self.save_path = LOGS_DIR / "shared_training_data.json"
        self.auto_save_interval = 300  # 5分鐘自動保存一次
        self.last_save_time = time.time()
        
        logger.info("SharedTrainingDataManager 初始化完成")
    
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
        添加訓練指標
        
        Args:
            step: 訓練步數
            reward: 獎勵值
            portfolio_value: 投資組合價值
            actor_loss: Actor網絡損失
            critic_loss: Critic網絡損失
            l2_norm: L2範數
            grad_norm: 梯度範數
        """
        metric = {
            'step': step,
            'reward': float(reward),
            'portfolio_value': float(portfolio_value),
            'actor_loss': float(actor_loss) if actor_loss is not None else 0.0,
            'critic_loss': float(critic_loss) if critic_loss is not None else 0.0,
            'l2_norm': float(l2_norm) if l2_norm is not None else 0.0,
            'grad_norm': float(grad_norm) if grad_norm is not None else 0.0,
            'timestamp': datetime.now(timezone.utc)
        }
        
        with self.lock:
            self.metrics_queue.append(metric)
            self.current_metrics = metric.copy()
            
            # 更新性能統計
            self.performance_stats['total_steps'] = step
            if reward > self.performance_stats['best_reward']:
                self.performance_stats['best_reward'] = reward
            if portfolio_value > self.performance_stats['best_portfolio_value']:
                self.performance_stats['best_portfolio_value'] = portfolio_value
            
            # 自動保存檢查
            if time.time() - self.last_save_time > self.auto_save_interval:
                self._auto_save()
    
    def add_trade_record(self, symbol: str, action: str, price: float, 
                        quantity: float, profit_loss: float, 
                        timestamp: Optional[datetime] = None):
        """
        添加交易記錄
        
        Args:
            symbol: 交易品種
            action: 交易動作 ('buy', 'sell', 'hold', 'close')
            price: 交易價格
            quantity: 交易數量
            profit_loss: 盈虧
            timestamp: 時間戳
        """
        trade = {
            'symbol': symbol,
            'action': action,
            'price': float(price),
            'quantity': float(quantity),
            'profit_loss': float(profit_loss),
            'timestamp': timestamp or datetime.now(timezone.utc)
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
                    'returns': [],
                    'win_rate': 0.0,
                    'avg_return': 0.0,
                    'max_return': 0.0,
                    'max_loss': 0.0,
                    'sharpe_ratio': 0.0
                }
            
            stats = self.symbol_stats[symbol]
            stats['trades'] += 1
            stats['total_profit'] += profit_loss
            stats['returns'].append(profit_loss)
            
            if profit_loss > 0:
                stats['wins'] += 1
            elif profit_loss < 0:
                stats['losses'] += 1
            
            # 計算統計指標
            if stats['trades'] > 0:
                stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
                stats['avg_return'] = np.mean(stats['returns'])
                stats['max_return'] = np.max(stats['returns'])
                stats['max_loss'] = np.min(stats['returns'])
                
                # 計算夏普比率
                if len(stats['returns']) > 1:
                    returns_std = np.std(stats['returns'])
                    stats['sharpe_ratio'] = stats['avg_return'] / returns_std if returns_std > 0 else 0
    
    def get_latest_metrics(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        獲取最新的訓練指標
        
        Args:
            count: 獲取的數量
            
        Returns:
            訓練指標列表
        """
        with self.lock:
            return list(self.metrics_queue)[-count:] if self.metrics_queue else []
    
    def get_latest_trades(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        獲取最新的交易記錄
        
        Args:
            count: 獲取的數量
            
        Returns:
            交易記錄列表
        """
        with self.lock:
            return list(self.trade_queue)[-count:] if self.trade_queue else []
    
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
                'total_return': ((portfolio_values[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100,
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
            self.metrics_queue.clear()
            self.trade_queue.clear()
            self.symbol_stats.clear()
            self.training_status = 'idle'
            self.training_progress = 0.0
            self.training_error = None
            self.stop_requested = False
            self.training_start_time = None
            
            self.current_metrics = {
                'step': 0,
                'reward': 0.0,
                'portfolio_value': float(INITIAL_CAPITAL),
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'l2_norm': 0.0,
                'grad_norm': 0.0,
                'timestamp': datetime.now(timezone.utc)
            }
            
            self.performance_stats = {
                'total_episodes': 0,
                'total_steps': 0,
                'best_reward': float('-inf'),
                'best_portfolio_value': float(INITIAL_CAPITAL),
                'avg_episode_length': 0,
                'training_efficiency': 0.0
            }
            
            logger.info("共享數據已清除")
    
    def _auto_save(self):
        """自動保存數據到文件"""
        try:
            # 只保存關鍵統計信息，不保存所有原始數據
            save_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'training_status': self.training_status,
                'training_progress': self.training_progress,
                'current_metrics': self.current_metrics,
                'symbol_stats': self.symbol_stats,
                'performance_stats': self.performance_stats,
                'training_summary': self.get_training_summary()
            }
            
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            self.last_save_time = time.time()
            logger.debug(f"自動保存完成: {self.save_path}")
            
        except Exception as e:
            logger.warning(f"自動保存失敗: {e}")
    
    def save_to_file(self, file_path: Optional[Path] = None):
        """
        手動保存數據到文件
        
        Args:
            file_path: 保存路徑，如果為None則使用默認路徑
        """
        save_path = file_path or self.save_path
        
        try:
            with self.lock:
                save_data = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'training_status': self.training_status,
                    'training_progress': self.training_progress,
                    'current_metrics': self.current_metrics,
                    'symbol_stats': self.symbol_stats,
                    'performance_stats': self.performance_stats,
                    'training_summary': self.get_training_summary(),
                    'recent_metrics': self.get_latest_metrics(100),
                    'recent_trades': self.get_latest_trades(100)
                }
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            logger.info(f"數據已保存到: {save_path}")
            
        except Exception as e:
            logger.error(f"保存數據失敗: {e}")
            raise
    
    def load_from_file(self, file_path: Optional[Path] = None):
        """
        從文件加載數據
        
        Args:
            file_path: 加載路徑，如果為None則使用默認路徑
        """
        load_path = file_path or self.save_path
        
        if not load_path.exists():
            logger.info(f"保存文件不存在: {load_path}")
            return
        
        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with self.lock:
                self.training_status = data.get('training_status', 'idle')
                self.training_progress = data.get('training_progress', 0.0)
                self.current_metrics = data.get('current_metrics', self.current_metrics)
                self.symbol_stats = data.get('symbol_stats', {})
                self.performance_stats = data.get('performance_stats', self.performance_stats)
                
                # 恢復最近的指標和交易數據
                recent_metrics = data.get('recent_metrics', [])
                recent_trades = data.get('recent_trades', [])
                
                for metric in recent_metrics:
                    self.metrics_queue.append(metric)
                
                for trade in recent_trades:
                    self.trade_queue.append(trade)
            
            logger.info(f"數據已從文件加載: {load_path}")
            
        except Exception as e:
            logger.error(f"加載數據失敗: {e}")
            raise


# 全局共享數據管理器實例
_global_shared_manager = None
_manager_lock = threading.Lock()

def get_shared_data_manager() -> SharedTrainingDataManager:
    """
    獲取全局共享數據管理器實例（單例模式）
    
    Returns:
        SharedTrainingDataManager實例
    """
    global _global_shared_manager
    
    with _manager_lock:
        if _global_shared_manager is None:
            _global_shared_manager = SharedTrainingDataManager()
            logger.info("創建全局共享數據管理器實例")
        
        return _global_shared_manager

def reset_shared_data_manager():
    """重置全局共享數據管理器"""
    global _global_shared_manager
    
    with _manager_lock:
        if _global_shared_manager is not None:
            _global_shared_manager.clear_data()
            logger.info("全局共享數據管理器已重置")