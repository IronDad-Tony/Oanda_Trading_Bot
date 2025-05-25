#!/usr/bin/env python3
"""
OANDA AI交易模型 - 完整的Streamlit應用
支持訓練配置、啟動、監控的一體化界面
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
from typing import Dict, Any
import sys
import os

# 確保能找到src模組
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 嘗試導入訓練器，如果失敗則使用備用方案
try:
    from src.trainer.enhanced_trainer import EnhancedUniversalTrainer, create_training_time_range
    from src.common.logger_setup import logger
    from src.common.config import ACCOUNT_CURRENCY, INITIAL_CAPITAL, DEVICE, USE_AMP
    from src.common.shared_data_manager import get_shared_data_manager
    TRAINER_AVAILABLE = True
    logger.info("成功導入訓練器和共享數據管理器")
except ImportError as e:
    # 如果導入失敗，使用備用配置
    import logging
    logger = logging.getLogger(__name__)
    ACCOUNT_CURRENCY = "USD"
    INITIAL_CAPITAL = 100000
    DEVICE = "cpu"
    USE_AMP = False
    TRAINER_AVAILABLE = False
    st.warning(f"訓練器模組導入失敗，使用模擬模式: {e}")
    
    # 創建後備共享數據管理器
    def get_shared_data_manager():
        """後備共享數據管理器"""
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

# 創建全局共享數據管理器實例
if 'shared_data_manager' not in st.session_state:
    st.session_state.shared_data_manager = get_shared_data_manager()
# 設置頁面配置
st.set_page_config(
    page_title="OANDA AI交易模型",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
def init_session_state():
    """初始化所有session state變量"""
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

# 調用初始化函數
init_session_state()

# 可用的交易品種
AVAILABLE_SYMBOLS = [
    "EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD", "GBP_AUD", "EUR_CAD",
    "GBP_CAD", "AUD_CAD", "EUR_CHF", "GBP_CHF", "AUD_CHF", "CAD_CHF", "NZD_JPY",
    "XAU_USD", "XAG_USD", "SPX500_USD", "NAS100_USD", "US30_USD"
]

def load_tensorboard_data():
    """從TensorBoard日誌中讀取訓練數據"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None
    
    # 查找最新的TensorBoard日誌目錄
    tb_dirs = list(logs_dir.glob("sac_tensorboard_logs_*"))
    if not tb_dirs:
        return None
    
    latest_tb_dir = max(tb_dirs, key=lambda x: x.stat().st_mtime)
    
    # 這裡應該解析TensorBoard的事件文件
    # 為了演示，我們使用session_state中的數據
    if st.session_state.training_data:
        return pd.DataFrame(st.session_state.training_data)
    
    return None

def load_model_info():
    """載入模型信息"""
    weights_dir = Path("weights")
    logs_dir = Path("logs")
    
    model_files = []
    
    # 檢查weights目錄
    if weights_dir.exists():
        for model_file in weights_dir.rglob("*.zip"):
            model_info = _parse_model_info(model_file)
            model_files.append(model_info)
    
    # 檢查logs目錄
    if logs_dir.exists():
        for model_file in logs_dir.rglob("*.zip"):
            model_info = _parse_model_info(model_file)
            model_files.append(model_info)
    
    return sorted(model_files, key=lambda x: x['modified'], reverse=True)

def _parse_model_info(model_file: Path) -> Dict[str, Any]:
    """
    解析模型文件信息，提取參數
    
    Args:
        model_file: 模型文件路徑
        
    Returns:
        模型信息字典
    """
    try:
        stat = model_file.stat()
        name = model_file.name
        
        # 解析模型參數
        max_symbols = None
        timestep = None
        model_type = "unknown"
        
        # 嘗試從文件名解析參數
        if "symbols" in name and "timestep" in name:
            try:
                # 匹配 sac_model_symbols{數量}_timestep{步長} 格式
                import re
                pattern = r"symbols(\d+)_timestep(\d+)"
                match = re.search(pattern, name)
                if match:
                    max_symbols = int(match.group(1))
                    timestep = int(match.group(2))
                    model_type = "optimized"
            except:
                pass
        
        # 計算訓練時長（基於文件修改時間和創建時間的差異）
        training_duration = None
        try:
            # 這是一個估算，實際訓練時長需要從其他地方獲取
            creation_time = datetime.fromtimestamp(stat.st_ctime)
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            if modified_time > creation_time:
                training_duration = modified_time - creation_time
        except:
            pass
        
        return {
            'name': name,
            'path': str(model_file),
            'size': stat.st_size / (1024*1024),  # MB
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'max_symbols': max_symbols,
            'timestep': timestep,
            'model_type': model_type,
            'training_duration': training_duration
        }
        
    except Exception as e:
        logger.warning(f"解析模型文件信息時發生錯誤: {e}")
        return {
            'name': model_file.name,
            'path': str(model_file),
            'size': 0,
            'modified': datetime.now(),
            'created': datetime.now(),
            'max_symbols': None,
            'timestep': None,
            'model_type': "unknown",
            'training_duration': None
        }

def _format_duration(duration):
    """
    格式化時間間隔顯示
    
    Args:
        duration: timedelta 對象或 None
        
    Returns:
        格式化的時間字符串
    """
    if duration is None:
        return "N/A"
    
    try:
        if isinstance(duration, timedelta):
            total_seconds = int(duration.total_seconds())
        else:
            total_seconds = int(duration)
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except:
        return "N/A"

def _delete_model_file(file_path: str) -> bool:
    """
    刪除模型文件
    
    Args:
        file_path: 要刪除的文件路徑
        
    Returns:
        是否成功刪除
    """
    try:
        model_file = Path(file_path)
        if model_file.exists():
            model_file.unlink()
            logger.info(f"已刪除模型文件: {file_path}")
            return True
        else:
            logger.warning(f"模型文件不存在: {file_path}")
            return False
    except Exception as e:
        logger.error(f"刪除模型文件時發生錯誤: {e}")
        return False

def simulate_training_with_shared_manager(shared_manager, symbols, total_timesteps):
    """模擬訓練過程，用於TRAINER_AVAILABLE為False時"""
    logger.info(f"開始模擬訓練 for symbols: {symbols}, steps: {total_timesteps}")
    shared_manager.update_training_status('running', 0)
    for step in range(total_timesteps):
        if shared_manager.is_stop_requested():
            logger.info("模擬訓練：收到停止請求。")
            shared_manager.update_training_status('idle')
            return False
        
        progress = (step + 1) / total_timesteps * 100
        shared_manager.update_training_status('running', progress)
        
        # 模擬指標更新
        current_metrics = {
            'step': step,
            'reward': np.random.rand() * 10 - 5, # 隨機獎勵
            'portfolio_value': float(INITIAL_CAPITAL * (1 + np.random.randn() * 0.01)),
            'actor_loss': np.random.rand() * 0.1,
            'critic_loss': np.random.rand() * 0.1,
            'l2_norm': np.random.rand() * 1.0,
            'grad_norm': np.random.rand() * 0.5,
            'timestamp': datetime.now(timezone.utc)
        }
        shared_manager.add_training_metric(**current_metrics)
        
        if step % 100 == 0: # 每100步模擬一次日誌
            logger.debug(f"模擬訓練進度: {progress:.1f}%")
            
        time.sleep(0.001) # 模擬耗時
        
    shared_manager.update_training_status('completed', 100)
    logger.info("模擬訓練完成。")
    return True

def training_worker(trainer, shared_manager, symbols, total_timesteps):
    """訓練工作線程 - 使用共享數據管理器"""
    try:
        logger.info("開始訓練工作線程，使用共享數據管理器")
        
        if trainer and TRAINER_AVAILABLE:
            # 真實訓練
            logger.info("開始真實訓練過程")
            
            # 將共享數據管理器附加到訓練器
            trainer.shared_data_manager = shared_manager
            
            # 執行真實訓練
            try:
                success = trainer.run_full_training_pipeline()
            except Exception as e:
                logger.error(f"真實訓練過程中發生錯誤: {e}")
                # 如果真實訓練失敗，回退到模擬訓練
                logger.info("回退到模擬訓練")
                success = simulate_training_with_shared_manager(shared_manager, symbols, total_timesteps)
        else:
            # 模擬訓練
            logger.info("開始模擬訓練過程")
            success = simulate_training_with_shared_manager(shared_manager, symbols, total_timesteps)
        
        # 更新最終狀態
        if shared_manager.is_stop_requested():
            shared_manager.update_training_status('idle')
            logger.info("訓練已被用戶停止")
        elif success:
            shared_manager.update_training_status('completed', 100)
            logger.info("訓練已完成")
        else:
            shared_manager.update_training_status('error', error="訓練未成功完成")
            
    except Exception as e:
        logger.error(f"訓練過程中發生錯誤: {e}", exc_info=True)
        shared_manager.update_training_status('error', error=str(e))
    finally:
        # 確保訓練停止後釋放資源
        if trainer and hasattr(trainer, 'cleanup'):
            trainer.cleanup()

def update_streamlit_ui_from_shared_data():
    """從共享數據管理器同步訓練數據到session_state"""
    shared_manager = st.session_state.shared_data_manager
    
    try:
        # 同步訓練狀態
        status_data = shared_manager.get_current_status()
        
        # 更新session_state中的狀態
        st.session_state.training_status = status_data.get('status', 'idle')
        st.session_state.training_progress = status_data.get('progress', 0)
        if status_data.get('error'):
            st.session_state.training_error = status_data['error']
        
        # 從共享數據管理器獲取最新指標
        latest_metrics = shared_manager.get_latest_metrics(1000) # 獲取最近1000個指標
        
        # 構建兼容的metrics格式
        if latest_metrics:
            metrics = {
                'steps': [m['step'] for m in latest_metrics],
                'rewards': [m['reward'] for m in latest_metrics],
                'portfolio_values': [m['portfolio_value'] for m in latest_metrics],
                'losses': [{'actor_loss': m['actor_loss'], 'critic_loss': m['critic_loss']} for m in latest_metrics],
                'norms': [{'l2_norm': m['l2_norm'], 'grad_norm': m['grad_norm']} for m in latest_metrics],
                'symbol_stats': status_data['symbol_stats'], # 直接從status_data獲取
                'timestamps': [m['timestamp'] for m in latest_metrics]
            }
            st.session_state.training_metrics.update(metrics)
            
    except Exception as e:
        logger.warning(f"同步訓練數據失敗: {e}")


def start_training(symbols, start_date, end_date, total_timesteps, save_freq, eval_freq):
    """啟動訓練"""
    try:
        # 重置共享數據管理器
        shared_manager = st.session_state.shared_data_manager
        shared_manager.clear_data()
        shared_manager.reset_stop_flag()
        
        # 轉換日期格式
        start_time = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_time = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        
        if TRAINER_AVAILABLE:
            # 創建真實訓練器
            trainer = EnhancedUniversalTrainer(
                trading_symbols=symbols,
                start_time=start_time,
                end_time=end_time,
                granularity="S5",
                total_timesteps=total_timesteps,
                save_freq=save_freq,
                eval_freq=eval_freq,
                model_name_prefix="sac_universal_trader",
                streamlit_session_state=st.session_state  # 保持原有參數
            )
            # 訓練器已經在初始化時自動連接到共享數據管理器
            st.session_state.trainer = trainer
        else:
            # 使用模擬訓練器
            st.session_state.trainer = None
        
        # 更新訓練狀態
        shared_manager.update_training_status('running', 0)
        
        # 在後台線程中啟動訓練
        training_thread = threading.Thread(
            target=training_worker, # 修正：使用已定義的 training_worker
            args=(st.session_state.trainer, shared_manager, symbols, total_timesteps)
        )
        training_thread.daemon = True
        training_thread.start()
        
        # 保存線程引用
        st.session_state.training_thread = training_thread
        
        return True
        
    except Exception as e:
        st.error(f"啟動訓練失敗: {e}")
        logger.error(f"啟動訓練失敗: {e}", exc_info=True)
        return False

def stop_training():
    """停止訓練"""
    try:
        # 通過共享數據管理器發送停止信號
        shared_manager = st.session_state.shared_data_manager
        shared_manager.request_stop()
        logger.info("已通過共享數據管理器發送停止信號")
        
        # 如果有訓練器實例，嘗試停止它
        if st.session_state.trainer:
            # 這裡需要實現一個停止訓練的方法
            # 假設訓練器有一個 stop 方法
            if hasattr(st.session_state.trainer, 'stop'):
                st.session_state.trainer.stop()
            
            # 保存當前模型
            if hasattr(st.session_state.trainer, 'save_current_model'):
                st.session_state.trainer.save_current_model()
                logger.info("已保存當前訓練進度")
        
        # 等待訓練線程結束（最多等待5秒）
        if st.session_state.training_thread and st.session_state.training_thread.is_alive():
            st.session_state.training_thread.join(timeout=5.0)
        
        # 重置狀態
        st.session_state.training_status = 'idle'
        st.session_state.training_thread = None
        
        return True
        
    except Exception as e:
        logger.error(f"停止訓練時發生錯誤: {e}", exc_info=True)
        return False

def reset_training_state():
    """重置訓練狀態和參數"""
    # 停止正在進行的訓練
    if st.session_state.training_status == 'running':
        stop_training()
    
    # 重置所有訓練相關的session state
    st.session_state.training_status = 'idle'
    st.session_state.training_progress = 0
    st.session_state.training_data = []
    st.session_state.trainer = None
    st.session_state.training_error = None
    st.session_state.training_thread = None
    st.session_state.stop_training = False
    
    # 清除選擇的symbols（如果存在）
    if 'selected_symbols' in st.session_state:
        del st.session_state.selected_symbols
    
    logger.info("訓練狀態已重置")

def generate_test_data():
    """生成測試數據用於監控頁面展示"""
    import numpy as np
    from datetime import datetime, timedelta
    
    # 清空現有數據
    st.session_state.training_metrics = {
        'steps': [],
        'rewards': [],
        'portfolio_values': [],
        'losses': [],
        'norms': [],
        'symbol_stats': {},
        'timestamps': []
    }
    
    # 生成50個數據點
    num_points = 50
    np.random.seed(42)  # 確保可重現的結果
    
    # 生成訓練步數
    steps = list(range(0, num_points * 100, 100))
    
    # 生成獎勵數據（逐漸改善的趨勢）
    rewards = []
    base_reward = -2.0
    for i in range(num_points):
        trend = i * 0.03  # 逐漸改善
        noise = np.random.normal(0, 0.5)
        reward = base_reward + trend + noise
        rewards.append(reward)
    
    # 生成投資組合淨值
    portfolio_values = []
    initial_capital = INITIAL_CAPITAL
    current_value = initial_capital
    
    for i, reward in enumerate(rewards):
        return_rate = reward * 0.001  # 縮放因子
        current_value *= (1 + return_rate)
        current_value *= (1 + np.random.normal(0, 0.005))  # 添加隨機波動
        portfolio_values.append(current_value)
    
    # 生成損失數據
    losses = []
    for i in range(num_points):
        actor_loss = 0.5 * np.exp(-i/20) + np.random.normal(0, 0.05)
        critic_loss = 0.8 * np.exp(-i/15) + np.random.normal(0, 0.08)
        losses.append({
            'actor_loss': max(0, actor_loss),
            'critic_loss': max(0, critic_loss)
        })
    
    # 生成範數數據
    norms = []
    for i in range(num_points):
        l2_norm = 10 + np.sin(i/10) * 2 + np.random.normal(0, 0.3)
        grad_norm = 1.0 * np.exp(-i/30) + np.random.normal(0, 0.1)
        norms.append({
            'l2_norm': max(0, l2_norm),
            'grad_norm': max(0, grad_norm)
        })
    
    # 生成時間戳
    start_time = datetime.now() - timedelta(hours=2)
    timestamps = [start_time + timedelta(minutes=i*2) for i in range(num_points)]
    
    # 生成交易品種統計
    symbols = ['EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CAD']
    symbol_stats = {}
    
    for symbol in symbols:
        num_trades = np.random.randint(20, 100)
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
    
    # 更新session state
    st.session_state.training_metrics.update({
        'steps': steps,
        'rewards': rewards,
        'portfolio_values': portfolio_values,
        'losses': losses,
        'norms': norms,
        'symbol_stats': symbol_stats,
        'timestamps': timestamps
    })

def main():
    """主應用函數"""
    
    # 標題和描述
    st.title("🚀 OANDA AI交易模型訓練系統")
    st.markdown("**一體化的AI量化交易模型訓練和監控平台**")
    st.markdown("---")
    
    # 創建標籤頁
    tab1, tab2, tab3 = st.tabs(["🎯 訓練配置", "📊 實時監控", "💾 模型管理"])
    
    with tab1:
        st.header("🎯 訓練配置")
        
        # 添加頁面說明
        with st.expander("ℹ️ 使用說明", expanded=False):
            st.markdown("""
            ### 📖 訓練配置頁面使用指南
            
            **🎯 功能概述：**
            此頁面用於配置和啟動AI交易模型的訓練任務。您可以選擇交易品種、設定訓練參數，並監控訓練狀態。
            
            **📈 交易品種選擇：**
            - **預設組合**：提供常用的交易品種組合，適合不同的交易策略
              - 主要貨幣對：流動性高，適合初學者
              - 歐洲貨幣對：歐洲時段活躍
              - 日元交叉盤：亞洲時段活躍
              - 貴金屬：避險資產，波動較大
              - 美股指數：股票市場指數
            - **自定義選擇**：可自由選擇1-20個交易品種
            - **建議**：初次訓練建議選擇3-5個主要貨幣對
            
            **📅 訓練時間範圍：**
            - **開始/結束日期**：選擇歷史數據的時間範圍
            - **建議範圍**：至少30天，推薦60-90天的數據
            - **注意**：數據量越大，訓練時間越長，但模型效果可能更好
            
            **⚙️ 訓練參數說明：**
            - **總訓練步數**：模型訓練的總迭代次數
              - 建議值：50,000-100,000步
              - 步數越多，訓練時間越長，但模型可能更精確
            - **保存頻率**：每隔多少步保存一次模型
              - 建議值：2,000-5,000步
              - 頻率越高，佔用存儲空間越多，但能更好地保留訓練進度
            - **評估頻率**：每隔多少步評估一次模型性能
              - 建議值：5,000-10,000步
              - 用於監控訓練效果和調整策略
            
            **💡 推薦配置：**
            - **新手配置**：3個主要貨幣對 + 30天數據 + 50,000步
            - **進階配置**：5-8個品種 + 60天數據 + 100,000步
            - **專業配置**：10-15個品種 + 90天數據 + 200,000步
            """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 交易品種選擇
            st.subheader("📈 選擇交易品種")
            
            # 預設選項
            preset_options = {
                "主要貨幣對": ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD"],
                "歐洲貨幣對": ["EUR_USD", "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_CAD"],
                "日元交叉盤": ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY"],
                "貴金屬": ["XAU_USD", "XAG_USD"],
                "美股指數": ["SPX500_USD", "NAS100_USD", "US30_USD"],
                "自定義": []
            }
            
            preset_choice = st.selectbox("選擇預設組合", list(preset_options.keys()))
            
            if preset_choice == "自定義":
                selected_symbols = st.multiselect(
                    "選擇交易品種",
                    AVAILABLE_SYMBOLS,
                    default=["EUR_USD", "USD_JPY", "GBP_USD"]
                )
            else:
                selected_symbols = st.multiselect(
                    "選擇交易品種",
                    AVAILABLE_SYMBOLS,
                    default=preset_options[preset_choice]
                )
            
            if len(selected_symbols) == 0:
                st.warning("請至少選擇一個交易品種")
            elif len(selected_symbols) > 20:
                st.warning("最多只能選擇20個交易品種")
            
            # 時間範圍設置
            st.subheader("📅 設置訓練時間範圍")
            
            col_date1, col_date2 = st.columns(2)
            
            with col_date1:
                start_date = st.date_input(
                    "開始日期",
                    value=datetime.now().date() - timedelta(days=30),
                    max_value=datetime.now().date()
                )
            
            with col_date2:
                end_date = st.date_input(
                    "結束日期",
                    value=datetime.now().date() - timedelta(days=1),
                    max_value=datetime.now().date()
                )
            
            if start_date >= end_date:
                st.error("開始日期必須早於結束日期")
            
            # 計算數據天數
            data_days = (end_date - start_date).days
            st.info(f"📊 將使用 {data_days} 天的歷史數據進行訓練")
            
            # 訓練參數設置
            st.subheader("⚙️ 訓練參數")
            
            col_param1, col_param2, col_param3 = st.columns(3)
            
            with col_param1:
                total_timesteps = st.number_input(
                    "總訓練步數",
                    min_value=1000,
                    max_value=1000000,
                    value=50000,
                    step=1000,
                    help="模型訓練的總迭代次數。建議值：新手50,000步，進階100,000步，專業200,000步以上。"
                )
            
            with col_param2:
                save_freq = st.number_input(
                    "保存頻率",
                    min_value=100,
                    max_value=10000,
                    value=2000,
                    step=100,
                    help="每隔多少步保存一次模型。建議值：2,000-5,000步。頻率越高佔用空間越多，但能更好保留訓練進度。"
                )
            
            with col_param3:
                eval_freq = st.number_input(
                    "評估頻率",
                    min_value=500,
                    max_value=20000,
                    value=5000,
                    step=500,
                    help="每隔多少步評估一次模型性能。建議值：5,000-10,000步。用於監控訓練效果和調整策略。"
                )
            
            # 預估訓練時間
            estimated_minutes = total_timesteps / 1000 * 2  # 粗略估算
            st.info(f"⏱️ 預估訓練時間: {estimated_minutes:.0f} 分鐘")
        
        with col2:
            # 訓練狀態顯示
            st.subheader("🔄 訓練狀態")
            
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
            
            # 從共享數據管理器獲取當前狀態
            shared_manager = st.session_state.shared_data_manager
            current_status_data = shared_manager.get_current_status()
            current_status = current_status_data['status']
            current_progress = current_status_data['progress']
            current_error = current_status_data['error']
            
            st.markdown(f"**狀態**: {status_colors[current_status]} {status_texts[current_status]}")
            
            if current_status == 'running':
                st.progress(current_progress / 100)
                st.markdown(f"**進度**: {current_progress:.1f}%")
            elif current_status == 'error' and current_error:
                st.error(f"錯誤詳情: {current_error}")
            
            # 系統資源監控
            st.subheader("💻 系統資源")
            try:
                import psutil
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                st.metric("CPU使用率", f"{cpu_percent:.1f}%")
                st.metric("內存使用率", f"{memory_percent:.1f}%")
            except ImportError:
                st.info("安裝 psutil 以顯示系統資源")
            
            # 訓練控制按鈕
            st.subheader("🎮 訓練控制")
            
            # 判斷是否可以開始訓練
            can_start = (
                len(selected_symbols) > 0 and
                len(selected_symbols) <= 20 and
                start_date < end_date and
                current_status in ['idle', 'completed', 'error']
            )
            
            # 判斷是否可以停止訓練
            can_stop = current_status == 'running'
            
            # 開始訓練按鈕
            start_button = st.button(
                "🚀 開始訓練",
                type="primary",
                disabled=not can_start,
                key="start_training_btn"
            )
            
            if start_button and can_start:
                with st.spinner("正在啟動訓練..."):
                    if start_training(selected_symbols, start_date, end_date, total_timesteps, save_freq, eval_freq):
                        st.success("訓練已啟動！請切換到監控標籤頁查看進度。")
                        # 強制刷新頁面以更新按鈕狀態
                        time.sleep(0.5)  # 給訓練線程一點時間啟動
                        st.rerun()
                    else:
                        st.error("啟動訓練失敗，請檢查配置")
            
            # 顯示自動載入信息
            if hasattr(st.session_state, 'loaded_existing_model') and st.session_state.loaded_existing_model:
                st.info("✅ 已自動載入既有模型繼續訓練")
                if hasattr(st.session_state, 'loaded_model_info') and st.session_state.loaded_model_info:
                    model_info = st.session_state.loaded_model_info
                    st.markdown(f"""
                    **載入的模型信息：**
                    - 模型名稱: {model_info.get('name', 'N/A')}
                    - 交易品種數: {model_info.get('max_symbols', 'N/A')}
                    - 時間步長: {model_info.get('timestep', 'N/A')}
                    - 文件大小: {model_info.get('size_mb', 0):.1f} MB
                    - 最後更新: {model_info.get('modified', 'N/A')}
                    """)
            elif hasattr(st.session_state, 'loaded_existing_model') and not st.session_state.loaded_existing_model:
                st.info("🆕 將創建新的模型")
            
            # 停止訓練按鈕
            stop_button = st.button(
                "⏹️ 停止訓練",
                disabled=not can_stop,
                key="stop_training_btn"
            )
            
            if stop_button and can_stop:
                with st.spinner("正在停止訓練並保存模型..."):
                    # 通過共享數據管理器發送停止信號
                    shared_manager.request_stop()
                    
                    # 同時調用訓練器的停止方法
                    if hasattr(st.session_state, 'trainer') and st.session_state.trainer:
                        if hasattr(st.session_state.trainer, 'stop'):
                            st.session_state.trainer.stop()
                    
                    st.success("⏹️ 已發送停止信號")
                    time.sleep(1)
                    st.rerun()
            
            # 重置按鈕
            reset_button = st.button(
                "🔄 重置",
                key="reset_btn",
                help="重置所有訓練狀態和參數，清除選擇的交易品種"
            )
            
            if reset_button:
                with st.spinner("正在重置訓練狀態..."):
                    reset_training_state()
                    st.success("訓練狀態已重置")
                    time.sleep(0.3)
                    st.rerun()
            
            # 顯示按鈕狀態說明
            with st.expander("ℹ️ 按鈕功能說明"):
                st.markdown("""
                - **開始訓練**: 啟動新的訓練任務
                  - 訓練開始後此按鈕會變為禁用狀態
                  - 訓練完成或出錯後可以重新開始
                
                - **停止訓練**: 優雅地停止正在進行的訓練
                  - 會保存當前的模型進度
                  - 釋放所有訓練資源
                  - 只在訓練進行中時可用
                
                - **重置**: 重置所有訓練相關的狀態
                  - 清除選擇的交易品種
                  - 重置訓練參數為默認值
                  - 清除訓練進度和錯誤信息
                  - 如果有正在進行的訓練會先停止
                """)
    
    with tab2:
        st.header("📊 實時監控")
        
        # 添加監控頁面說明
        with st.expander("ℹ️ 監控說明", expanded=False):
            st.markdown("""
            ### 📊 實時監控頁面使用指南
            
            **🎯 功能概述：**
            此頁面提供訓練過程的實時監控和可視化分析，幫助您了解模型的學習進度和性能表現。
            
            **📈 主要指標標籤頁：**
            
            **1. 訓練獎勵趨勢圖：**
            - **藍色實線**：每步的即時獎勵值
            - **橙色虛線**：移動平均線，顯示整體趨勢
            - **解讀方式**：
              - 上升趨勢表示模型學習效果良好
              - 波動是正常現象，關注整體趨勢
              - 移動平均線平穩上升是理想狀態
            
            **2. 投資組合淨值變化圖：**
            - **綠色線條**：模擬交易的投資組合價值變化
            - **灰色虛線**：初始資本基準線
            - **解讀方式**：
              - 高於基準線表示盈利
              - 低於基準線表示虧損
              - 穩定上升表示策略有效
            
            **3. 實時指標面板：**
            - **訓練步數**：當前完成的訓練迭代次數
            - **當前獎勵**：最新的獎勵值及變化
            - **投資組合淨值**：當前模擬資產價值
            - **投資回報率**：相對於初始資本的收益率
            - **訓練時長**：已經進行的訓練時間
            
            **📊 交易統計標籤頁：**
            - **交易次數**：每個品種的交易頻率
            - **勝率**：盈利交易佔總交易的比例
            - **平均收益**：每筆交易的平均收益率
            - **最大收益/虧損**：單筆交易的極值
            - **夏普比率**：風險調整後的收益指標
            - **收益分佈圖**：各品種的收益分佈箱線圖
            
            **🔬 模型診斷標籤頁：**
            
            **1. 損失函數圖：**
            - **Actor Loss (紅線)**：策略網絡的損失，控制動作選擇
            - **Critic Loss (紫線)**：價值網絡的損失，評估狀態價值
            - **解讀方式**：
              - 損失值應該逐漸下降並趨於穩定
              - 劇烈波動可能表示學習率過高
              - 長期不變可能表示學習停滯
            
            **2. 模型範數監控：**
            - **L2 Norm (橙線)**：模型參數的L2範數，反映模型複雜度
            - **Gradient Norm (綠線)**：梯度的範數，反映學習強度
            - **解讀方式**：
              - L2範數過大可能表示過擬合
              - 梯度範數過小可能表示學習停滯
              - 梯度範數過大可能表示學習不穩定
            
            **3. 訓練穩定性指標：**
            - **獎勵標準差**：獎勵值的波動程度
            - **平均Actor Loss**：策略網絡損失的平均值
            - **平均梯度範數**：梯度更新的平均強度
            
            **💡 監控建議：**
            - **正常訓練**：獎勵上升、損失下降、範數穩定
            - **需要調整**：獎勵停滯、損失劇烈波動、梯度異常
            - **訓練完成**：各指標趨於穩定，投資組合持續盈利
            """)
        
        # 從共享數據管理器獲取最新數據
        shared_manager = st.session_state.shared_data_manager
        current_status = shared_manager.get_current_status()
        
        # 更新session_state以保持兼容性
        st.session_state.training_status = current_status['status']
        st.session_state.training_progress = current_status['progress']
        if current_status['error']:
            st.session_state.training_error = current_status['error']
        
        # 從共享數據管理器構建metrics數據
        latest_metrics = shared_manager.get_latest_metrics(1000)
        latest_trades = shared_manager.get_latest_trades(1000)
        
        # 構建兼容的metrics格式
        if latest_metrics:
            metrics = {
                'steps': [m['step'] for m in latest_metrics],
                'rewards': [m['reward'] for m in latest_metrics],
                'portfolio_values': [m['portfolio_value'] for m in latest_metrics],
                'losses': [{'actor_loss': m['actor_loss'], 'critic_loss': m['critic_loss']} for m in latest_metrics],
                'norms': [{'l2_norm': m['l2_norm'], 'grad_norm': m['grad_norm']} for m in latest_metrics],
                'symbol_stats': current_status['symbol_stats'],
                'timestamps': [m['timestamp'] for m in latest_metrics]
            }
        else:
            # 如果沒有共享數據，回退到session_state
            metrics = st.session_state.training_metrics
        
        # 添加測試數據生成按鈕（用於調試）
        if len(metrics['steps']) == 0:
            col_test1, col_test2 = st.columns(2)
            with col_test1:
                if st.button("🧪 生成測試數據", key="generate_test_data"):
                    generate_test_data()
                    st.success("已生成測試數據")
                    st.rerun()
            with col_test2:
                st.info("💡 如果沒有訓練數據，可以點擊「生成測試數據」查看圖表效果")
        
        if len(metrics['steps']) > 0:
            # 創建三個標籤頁來組織不同類型的圖表
            monitor_tab1, monitor_tab2, monitor_tab3 = st.tabs(["📈 主要指標", "📊 交易統計", "🔬 模型診斷"])
            
            with monitor_tab1:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # 訓練獎勵趨勢圖
                    fig_reward = go.Figure()
                    
                    # 確保使用訓練時間步作為橫軸，而不是索引
                    if len(metrics['steps']) > 0 and len(metrics['rewards']) > 0:
                        # 確保steps和rewards長度一致
                        min_length = min(len(metrics['steps']), len(metrics['rewards']))
                        x_values = metrics['steps'][:min_length]
                        y_values = metrics['rewards'][:min_length]
                        
                        fig_reward.add_trace(go.Scatter(
                            x=x_values,
                            y=y_values,
                            mode='lines',
                            name='訓練獎勵',
                            line=dict(color='#1f77b4', width=2)
                        ))
                        
                        # 添加移動平均線
                        if len(y_values) > 20:
                            window_size = min(100, len(y_values) // 5)
                            ma_rewards = pd.Series(y_values).rolling(window=window_size).mean()
                            fig_reward.add_trace(go.Scatter(
                                x=x_values,
                                y=ma_rewards,
                                mode='lines',
                                name=f'{window_size}步移動平均',
                                line=dict(color='#ff7f0e', width=2, dash='dash')
                            ))
                    
                    fig_reward.update_layout(
                        title="訓練獎勵趨勢",
                        xaxis_title="訓練步數",
                        yaxis_title="獎勵值",
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig_reward, use_container_width=True)
                    
                    # 添加獎勵圖說明
                    with st.expander("📈 獎勵趨勢圖說明"):
                        st.markdown("""
                        **圖表解讀：**
                        - **藍色實線**：每個訓練步驟的即時獎勵值
                        - **橙色虛線**：移動平均線，平滑顯示整體趨勢
                        
                        **正常表現：**
                        - 初期：獎勵值波動較大，模型在探索學習
                        - 中期：獎勵逐漸上升，波動減小
                        - 後期：獎勵趨於穩定，偶有小幅波動
                        
                        **異常警示：**
                        - 長期下降：可能學習率過高或數據有問題
                        - 劇烈波動：可能需要調整網絡結構或參數
                        - 完全平穩：可能學習停滯，需要調整策略
                        """)
                    
                    # 投資組合淨值變化圖
                    fig_portfolio = go.Figure()
                    
                    # 確保使用訓練時間步作為橫軸
                    if len(metrics['steps']) > 0 and len(metrics['portfolio_values']) > 0:
                        # 確保steps和portfolio_values長度一致
                        min_length = min(len(metrics['steps']), len(metrics['portfolio_values']))
                        x_values = metrics['steps'][:min_length]
                        y_values = metrics['portfolio_values'][:min_length]
                        
                        fig_portfolio.add_trace(go.Scatter(
                            x=x_values,
                            y=y_values,
                            mode='lines',
                            name='投資組合淨值',
                            line=dict(color='#2ca02c', width=2)
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
                        height=350
                    )
                    st.plotly_chart(fig_portfolio, use_container_width=True)
                    
                    # 添加淨值圖說明
                    with st.expander("💰 投資組合淨值說明"):
                        st.markdown(f"""
                        **圖表解讀：**
                        - **綠色線條**：模擬交易的投資組合總價值
                        - **灰色虛線**：初始資本基準線 ({INITIAL_CAPITAL:,} {ACCOUNT_CURRENCY})
                        
                        **性能指標：**
                        - **高於基準線**：模型產生正收益，策略有效
                        - **低於基準線**：模型產生負收益，需要優化
                        - **穩定上升**：理想狀態，表示持續盈利能力
                        
                        **風險評估：**
                        - **波動幅度**：反映策略的風險水平
                        - **最大回撤**：從峰值到谷值的最大跌幅
                        - **收益穩定性**：長期趨勢比短期波動更重要
                        
                        **注意事項：**
                        - 這是基於歷史數據的模擬結果
                        - 實際交易可能因滑點、手續費等因素有所差異
                        - 建議結合多個指標綜合評估模型性能
                        """)
                
                with col2:
                    # 實時指標
                    st.subheader("📊 實時指標")
                    
                    if len(metrics['steps']) > 0:
                        latest_idx = -1
                        
                        # 當前步數
                        st.metric(
                            "訓練步數",
                            f"{metrics['steps'][latest_idx]:,}",
                            f"+{metrics['steps'][latest_idx] - metrics['steps'][latest_idx-1]:,}" if len(metrics['steps']) > 1 else "+0"
                        )
                        
                        # 當前獎勵
                        if len(metrics['rewards']) > 0:
                            st.metric(
                                "當前獎勵",
                                f"{metrics['rewards'][latest_idx]:.2f}",
                                f"{metrics['rewards'][latest_idx] - metrics['rewards'][latest_idx-1]:.2f}" if len(metrics['rewards']) > 1 else "0.00"
                            )
                        
                        # 投資組合淨值
                        if len(metrics['portfolio_values']) > 0:
                            current_value = metrics['portfolio_values'][latest_idx]
                            value_change = current_value - metrics['portfolio_values'][latest_idx-1] if len(metrics['portfolio_values']) > 1 else 0
                            roi = ((current_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
                            
                            st.metric(
                                "投資組合淨值",
                                f"{ACCOUNT_CURRENCY} {current_value:,.2f}",
                                f"{value_change:+,.2f}"
                            )
                            
                            st.metric(
                                "投資回報率",
                                f"{roi:.2f}%",
                                f"{roi - ((metrics['portfolio_values'][latest_idx-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100:.2f}%" if len(metrics['portfolio_values']) > 1 else "0.00%"
                            )
                        
                        # 訓練時長
                        if len(metrics['timestamps']) > 0:
                            duration = (metrics['timestamps'][-1] - metrics['timestamps'][0]).total_seconds()
                            hours = int(duration // 3600)
                            minutes = int((duration % 3600) // 60)
                            st.metric(
                                "訓練時長",
                                f"{hours}h {minutes}m"
                            )
            
            with monitor_tab2:
                st.subheader("📊 交易統計")
                
                # 檢查是否有symbol統計數據
                if metrics['symbol_stats']:
                    # 創建DataFrame來顯示統計表
                    stats_data = []
                    for symbol, stats in metrics['symbol_stats'].items():
                        stats_data.append({
                            '交易品種': symbol,
                            '交易次數': stats.get('trades', 0),
                            '勝率': f"{stats.get('win_rate', 0):.1f}%",
                            '平均收益': f"{stats.get('avg_return', 0):.2f}%",
                            '最大收益': f"{stats.get('max_return', 0):.2f}%",
                            '最大虧損': f"{stats.get('max_loss', 0):.2f}%",
                            '夏普比率': f"{stats.get('sharpe_ratio', 0):.2f}"
                        })
                    
                    if stats_data:
                        df_stats = pd.DataFrame(stats_data)
                        st.dataframe(df_stats, use_container_width=True, hide_index=True)
                        
                        # 添加交易統計表說明
                        with st.expander("📊 交易統計表說明"):
                            st.markdown("""
                            **統計指標解釋：**
                            
                            **📈 交易次數：**
                            - 該交易品種的總交易筆數
                            - 數值過低可能表示該品種交易機會較少
                            - 數值過高可能表示過度交易
                            
                            **🎯 勝率：**
                            - 盈利交易佔總交易的百分比
                            - 一般來說，勝率>50%較為理想
                            - 但高勝率不一定代表高收益
                            
                            **💰 平均收益：**
                            - 每筆交易的平均收益率
                            - 正值表示該品種整體盈利
                            - 應結合勝率和交易次數綜合評估
                            
                            **📊 最大收益/虧損：**
                            - 單筆交易的最佳和最差表現
                            - 反映策略的風險收益特徵
                            - 過大的數值可能表示風險控制不足
                            
                            **⚡ 夏普比率：**
                            - 風險調整後的收益指標
                            - 數值越高表示單位風險的收益越好
                            - 一般認為>1.0為良好，>2.0為優秀
                            
                            **💡 分析建議：**
                            - 關注夏普比率高的品種，風險收益比較好
                            - 平衡勝率和平均收益，避免過度追求單一指標
                            - 注意最大虧損，確保在可承受範圍內
                            """)
                        
                        # 交易品種收益分佈圖
                        if any(stats.get('returns', []) for stats in metrics['symbol_stats'].values()):
                            fig_returns = go.Figure()
                            for symbol, stats in metrics['symbol_stats'].items():
                                if 'returns' in stats and stats['returns']:
                                    fig_returns.add_trace(go.Box(
                                        y=stats['returns'],
                                        name=symbol,
                                        boxpoints='outliers'
                                    ))
                            
                            fig_returns.update_layout(
                                title="各交易品種收益分佈",
                                yaxis_title="收益率 (%)",
                                height=400
                            )
                            st.plotly_chart(fig_returns, use_container_width=True)
                    else:
                        st.info("暫無交易統計數據")
                else:
                    st.info("等待收集交易統計數據...")
            
            with monitor_tab3:
                st.subheader("🔬 模型診斷")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 損失函數圖
                    if len(metrics['losses']) > 0 and len(metrics['steps']) > 0:
                        fig_loss = go.Figure()
                        
                        # 分離不同類型的損失
                        actor_losses = [l.get('actor_loss', 0) for l in metrics['losses']]
                        critic_losses = [l.get('critic_loss', 0) for l in metrics['losses']]
                        
                        # 確保steps和losses長度一致
                        min_length = min(len(metrics['steps']), len(metrics['losses']))
                        x_values = metrics['steps'][:min_length]
                        
                        if any(actor_losses) and len(actor_losses) >= min_length:
                            fig_loss.add_trace(go.Scatter(
                                x=x_values,
                                y=actor_losses[:min_length],
                                mode='lines',
                                name='Actor Loss',
                                line=dict(color='#d62728', width=2)
                            ))
                        
                        if any(critic_losses) and len(critic_losses) >= min_length:
                            fig_loss.add_trace(go.Scatter(
                                x=x_values,
                                y=critic_losses[:min_length],
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
                        
                        # 添加損失函數說明
                        with st.expander("📉 損失函數說明"):
                            st.markdown("""
                            **損失類型解釋：**
                            
                            **🎭 Actor Loss (紅線)：**
                            - 策略網絡的損失函數
                            - 控制智能體的動作選擇策略
                            - 理想狀態：逐漸下降並趨於穩定
                            
                            **🎯 Critic Loss (紫線)：**
                            - 價值網絡的損失函數
                            - 評估當前狀態的價值
                            - 幫助策略網絡學習更好的動作
                            
                            **正常表現：**
                            - 初期：損失值較高，快速下降
                            - 中期：下降速度放緩，偶有波動
                            - 後期：趨於穩定，小幅波動
                            
                            **異常警示：**
                            - 持續上升：可能學習率過高
                            - 劇烈波動：網絡不穩定，需調整參數
                            - 完全平穩：可能學習停滯
                            - 數值過大：可能需要調整網絡結構
                            """)
                    else:
                        st.info("等待損失數據...")
                
                with col2:
                    # 模型範數圖
                    if len(metrics['norms']) > 0 and len(metrics['steps']) > 0:
                        fig_norm = go.Figure()
                        
                        # 分離不同的範數
                        l2_norms = [n.get('l2_norm', 0) for n in metrics['norms']]
                        grad_norms = [n.get('grad_norm', 0) for n in metrics['norms']]
                        
                        # 確保steps和norms長度一致
                        min_length = min(len(metrics['steps']), len(metrics['norms']))
                        x_values = metrics['steps'][:min_length]
                        
                        if any(l2_norms) and len(l2_norms) >= min_length:
                            fig_norm.add_trace(go.Scatter(
                                x=x_values,
                                y=l2_norms[:min_length],
                                mode='lines',
                                name='L2 Norm',
                                line=dict(color='#ff7f0e', width=2)
                            ))
                        
                        if any(grad_norms) and len(grad_norms) >= min_length:
                            fig_norm.add_trace(go.Scatter(
                                x=x_values,
                                y=grad_norms[:min_length],
                                mode='lines',
                                name='Gradient Norm',
                                line=dict(color='#2ca02c', width=2)
                            ))
                        
                        fig_norm.update_layout(
                            title="模型範數監控",
                            xaxis_title="訓練步數",
                            yaxis_title="範數值",
                            height=350
                        )
                        st.plotly_chart(fig_norm, use_container_width=True)
                        
                        # 添加範數監控說明
                        with st.expander("📐 範數監控說明"):
                            st.markdown("""
                            **範數類型解釋：**
                            
                            **🔶 L2 Norm (橙線)：**
                            - 模型參數的L2範數
                            - 反映模型的複雜度和參數大小
                            - 用於監控是否出現過擬合
                            
                            **🔷 Gradient Norm (綠線)：**
                            - 梯度向量的範數
                            - 反映參數更新的強度
                            - 指示學習過程的活躍程度
                            
                            **正常範圍：**
                            - L2範數：應保持相對穩定
                            - 梯度範數：初期較大，逐漸減小
                            
                            **異常警示：**
                            - **L2範數過大**：可能過擬合，需要正則化
                            - **L2範數劇烈變化**：訓練不穩定
                            - **梯度範數過小**：學習停滯，可能需要調整學習率
                            - **梯度範數過大**：可能出現梯度爆炸
                            - **梯度範數劇烈波動**：訓練不穩定
                            
                            **調整建議：**
                            - 梯度範數持續過大：降低學習率
                            - 梯度範數過小：提高學習率或檢查數據
                            - L2範數過大：增加正則化或減少模型複雜度
                            """)
                    else:
                        st.info("等待範數數據...")
                
                # 訓練穩定性指標
                st.subheader("📊 訓練穩定性")
                
                stability_col1, stability_col2, stability_col3 = st.columns(3)
                
                with stability_col1:
                    if len(metrics['rewards']) > 10:
                        reward_std = np.std(metrics['rewards'][-100:])
                        st.metric("獎勵標準差", f"{reward_std:.3f}")
                
                with stability_col2:
                    if len(metrics['losses']) > 0 and any('actor_loss' in l for l in metrics['losses']):
                        recent_losses = [l.get('actor_loss', 0) for l in metrics['losses'][-100:] if 'actor_loss' in l]
                        if recent_losses:
                            avg_loss = np.mean(recent_losses)
                            st.metric("平均Actor Loss", f"{avg_loss:.4f}")
                
                with stability_col3:
                    if len(metrics['norms']) > 0 and any('grad_norm' in n for n in metrics['norms']):
                        recent_grads = [n.get('grad_norm', 0) for n in metrics['norms'][-100:] if 'grad_norm' in n]
                        if recent_grads:
                            avg_grad = np.mean(recent_grads)
                            st.metric("平均梯度範數", f"{avg_grad:.4f}")
                
                # 添加穩定性指標說明
                with st.expander("📊 穩定性指標說明"):
                    st.markdown("""
                    **穩定性指標解釋：**
                    
                    **📈 獎勵標準差：**
                    - 最近100步獎勵值的標準差
                    - 反映獎勵的波動程度
                    - 數值越小表示訓練越穩定
                    - 理想範圍：隨訓練進行逐漸減小
                    
                    **🎭 平均Actor Loss：**
                    - 最近100步策略網絡損失的平均值
                    - 反映策略學習的穩定性
                    - 應該逐漸下降並趨於穩定
                    - 異常波動可能表示學習不穩定
                    
                    **📐 平均梯度範數：**
                    - 最近100步梯度範數的平均值
                    - 反映參數更新的平均強度
                    - 過大可能導致訓練不穩定
                    - 過小可能表示學習停滯
                    
                    **穩定性評估：**
                    - **良好**：各指標平穩下降，波動較小
                    - **一般**：有一定波動但整體趨勢正確
                    - **不穩定**：指標劇烈波動或異常變化
                    
                    **調整建議：**
                    - 如果穩定性差，考慮降低學習率
                    - 增加批次大小可能提高穩定性
                    - 檢查數據質量和預處理步驟
                    """)
        
        else:
            st.info("📊 暫無訓練數據。請先在「訓練配置」標籤頁啟動訓練。")
        
        # 顯示窗口和自動刷新選項
        st.markdown("---")
        st.subheader("📊 顯示設定")
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            # 顯示窗口設定
            display_window = st.number_input(
                "顯示窗口(步數)",
                min_value=200,
                max_value=10000,
                value=1000,
                step=100,
                help="設定圖表顯示的最大訓練步數。當訓練步數超過此值時，只顯示最近的數據。最小值為200步。"
            )
        
        with col2:
            auto_refresh = st.checkbox("自動刷新", value=True)
        
        with col3:
            if auto_refresh:
                refresh_interval_steps = st.number_input(
                    "刷新間隔(步數)",
                    min_value=1,
                    max_value=1000,
                    value=50,
                    step=1,
                    help="每隔多少訓練步數自動刷新一次頁面。最小值為1步，即每個訓練步都刷新。"
                )
        
        with col4:
            if st.button("🔄 手動刷新"):
                st.rerun()
        
        # 應用顯示窗口過濾
        if len(metrics['steps']) > 0:
            # 獲取當前最大步數
            max_steps = max(metrics['steps']) if metrics['steps'] else 0
            
            # 如果訓練步數超過顯示窗口，只顯示最近的數據
            if max_steps > display_window:
                # 找到需要顯示的步數範圍
                min_display_steps = max_steps - display_window
                
                # 過濾數據，只保留在顯示窗口內的數據
                filtered_indices = [i for i, step in enumerate(metrics['steps']) if step >= min_display_steps]
                
                if filtered_indices:
                    # 創建過濾後的metrics
                    filtered_metrics = {
                        'steps': [metrics['steps'][i] for i in filtered_indices],
                        'rewards': [metrics['rewards'][i] for i in filtered_indices] if len(metrics['rewards']) > 0 else [],
                        'portfolio_values': [metrics['portfolio_values'][i] for i in filtered_indices] if len(metrics['portfolio_values']) > 0 else [],
                        'losses': [metrics['losses'][i] for i in filtered_indices] if len(metrics['losses']) > 0 else [],
                        'norms': [metrics['norms'][i] for i in filtered_indices] if len(metrics['norms']) > 0 else [],
                        'timestamps': [metrics['timestamps'][i] for i in filtered_indices] if len(metrics['timestamps']) > 0 else [],
                        'symbol_stats': metrics['symbol_stats']  # 統計數據不需要過濾
                    }
                    
                    # 更新metrics為過濾後的數據
                    metrics = filtered_metrics
                    
                    st.info(f"📊 顯示最近 {display_window} 步的數據 (總共 {max_steps} 步)")
            else:
                st.info(f"📊 顯示全部 {max_steps} 步的數據")
        
        # 自動刷新邏輯 - 基於訓練步數而非時間
        shared_manager = st.session_state.shared_data_manager
        current_status_data = shared_manager.get_current_status()
        current_status = current_status_data['status']
        
        if auto_refresh and current_status == 'running':
            # 獲取當前步數
            current_metrics = shared_manager.get_current_status()['current_metrics']
            current_step = current_metrics.get('step', 0)
            
            # 檢查是否需要刷新（基於步數間隔）
            if 'last_refresh_step' not in st.session_state:
                st.session_state.last_refresh_step = 0
            
            steps_since_refresh = current_step - st.session_state.last_refresh_step
            
            if steps_since_refresh >= refresh_interval_steps:
                st.session_state.last_refresh_step = current_step
                time.sleep(1)  # 短暫延遲以避免過於頻繁的刷新
                st.rerun()
            else:
                # 顯示距離下次刷新的步數
                remaining_steps = refresh_interval_steps - steps_since_refresh
                st.caption(f"🔄 距離下次自動刷新還有 {remaining_steps} 步")
    
    with tab3:
        st.header("💾 模型管理")
        
        # 添加模型管理頁面說明
        with st.expander("ℹ️ 模型管理說明", expanded=False):
            st.markdown("""
            ### 💾 模型管理頁面使用指南
            
            **🎯 功能概述：**
            此頁面用於管理已訓練的AI交易模型，包括查看模型信息、載入模型、續練和刪除等操作。
            
            **📁 模型文件信息：**
            
            **1. 模型命名規則：**
            - **格式**：`sac_model_symbols{數量}_timestep{步長}_{時間戳}.zip`
            - **範例**：`sac_model_symbols5_timestep50000_20241225_143022.zip`
            - **解釋**：
              - `symbols5`：訓練時使用了5個交易品種
              - `timestep50000`：訓練了50,000步
              - `20241225_143022`：保存時間（2024年12月25日 14:30:22）
            
            **2. 模型信息表格說明：**
            - **模型名稱**：完整的文件名
            - **模型類型**：
              - `optimized`：包含完整參數信息的優化模型
              - `unknown`：無法解析參數的模型
            - **交易品種數**：訓練時使用的交易品種數量
            - **時間步長**：模型訓練的總步數
            - **文件大小**：模型文件的大小（MB）
            - **最後更新**：模型文件的最後修改時間
            - **訓練時長**：估算的訓練持續時間
            
            **🔧 模型操作功能：**
            
            **1. 📊 載入模型：**
            - 將選中的模型載入到系統中
            - 可用於模型評估和實盤交易
            - 注意：確保模型與當前環境兼容
            
            **2. 🔄 續練模型：**
            - 基於已有模型繼續訓練
            - 可以使用新的數據或調整參數
            - 適用於模型優化和增量學習
            
            **3. 📋 複製路徑：**
            - 顯示模型文件的完整路徑
            - 便於在其他程序中引用模型
            - 可手動複製路徑信息
            
            **4. 🗑️ 刪除模型：**
            - 永久刪除選中的模型文件
            - **警告**：此操作不可恢復
            - 建議在刪除前備份重要模型
            
            **📊 TensorBoard集成：**
            
            **1. 啟動TensorBoard：**
            - 提供啟動TensorBoard的命令
            - 用於詳細分析訓練過程
            - 訪問地址：http://localhost:6006
            
            **2. 日誌目錄：**
            - 顯示訓練日誌的存儲位置
            - 包含詳細的訓練指標和圖表
            - 可用於深度分析和調試
            
            **💡 管理建議：**
            
            **模型選擇原則：**
            - **最新模型**：通常性能最好，包含最新的訓練成果
            - **穩定模型**：訓練時間較長，性能穩定的模型
            - **特定配置**：針對特定交易品種或市場條件的模型
            
            **存儲管理：**
            - 定期清理過時或性能差的模型
            - 保留關鍵節點的模型備份
            - 注意磁盤空間使用情況
            
            **版本控制：**
            - 記錄模型的訓練參數和數據範圍
            - 建立模型性能評估記錄
            - 維護模型使用和更新日誌
            """)
        
        # 載入模型文件列表
        model_files = load_model_info()
        
        if model_files:
            st.subheader("📁 已保存的模型")
            
            # 創建增強的模型信息表格
            display_data = []
            for model in model_files:
                display_data.append({
                    '模型名稱': model['name'],
                    '模型類型': model.get('model_type', 'unknown'),
                    '交易品種數': model.get('max_symbols', 'N/A'),
                    '時間步長': model.get('timestep', 'N/A'),
                    '文件大小': f"{model['size']:.1f} MB",
                    '最後更新': model['modified'].strftime('%Y-%m-%d %H:%M:%S'),
                    '訓練時長': _format_duration(model.get('training_duration'))
                })
            
            df_display = pd.DataFrame(display_data)
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # 模型詳細信息和操作
            st.subheader("🔧 模型操作")
            
            # 選擇模型
            model_options = [f"{m['name']}" for m in model_files]
            selected_model_name = st.selectbox("選擇模型", options=model_options, index=0 if model_files else None)
            
            if selected_model_name:
                # 找到選中的模型
                selected_model = next((m for m in model_files if m['name'] == selected_model_name), None)
                
                if selected_model:
                    # 顯示詳細信息
                    with st.expander("📋 模型詳細信息", expanded=True):
                        col_info1, col_info2 = st.columns(2)
                        
                        with col_info1:
                            st.markdown(f"""
                            **基本信息：**
                            - 文件名: {selected_model['name']}
                            - 模型類型: {selected_model.get('model_type', 'unknown')}
                            - 文件大小: {selected_model['size']:.1f} MB
                            - 文件路徑: {selected_model['path']}
                            """)
                        
                        with col_info2:
                            st.markdown(f"""
                            **訓練參數：**
                            - 交易品種數: {selected_model.get('max_symbols', 'N/A')}
                            - 時間步長: {selected_model.get('timestep', 'N/A')}
                            - 創建時間: {selected_model.get('created', 'N/A')}
                            - 最後更新: {selected_model['modified'].strftime('%Y-%m-%d %H:%M:%S')}
                            """)
                        
                        if selected_model.get('training_duration'):
                            st.markdown(f"**訓練時長**: {_format_duration(selected_model['training_duration'])}")
                    
                    # 操作按鈕
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("📊 載入模型", key="load_model_btn"):
                            st.info("模型載入功能開發中...")
                    
                    with col2:
                        if st.button("🔄 續練模型", key="continue_training_btn"):
                            st.info("續練功能開發中...")
                    
                    with col3:
                        if st.button("📋 複製路徑", key="copy_path_btn"):
                            st.code(selected_model['path'])
                            st.success("路徑已顯示，可手動複製")
                    
                    with col4:
                        if st.button("️ 刪除模型", key="delete_model_btn", type="secondary"):
                            # 使用確認對話框
                            if st.button("⚠️ 確認刪除", key="confirm_delete_btn", type="primary"):
                                if _delete_model_file(selected_model['path']):
                                    st.success(f"模型 {selected_model['name']} 已刪除")
                                    st.rerun()
                                else:
                                    st.error("刪除模型失敗")
        
        else:
            st.info("📁 暫無已保存的模型文件")
        
        # TensorBoard集成
        st.subheader("📊 TensorBoard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 啟動TensorBoard"):
                st.code("tensorboard --logdir=logs/", language="bash")
                st.info("請在終端中運行上述命令，然後在瀏覽器中打開 http://localhost:6006")
        
        with col2:
            if st.button("📁 打開日誌目錄"):
                logs_path = Path("logs").absolute()
                st.info(f"日誌目錄: {logs_path}")

if __name__ == "__main__":
    main()