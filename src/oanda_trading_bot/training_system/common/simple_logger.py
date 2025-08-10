# src/common/simple_logger.py
"""
簡化版日誌配置模組 - Windows 友好版本
專門為解決 Windows 系統日誌檔案滾動權限問題而設計
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import os
import time

# 計算項目根目錄和日誌目錄
project_root = Path(__file__).resolve().parent.parent.parent
logs_dir = project_root / "logs"

def get_simple_log_file_path():
    """生成基於時間戳的日誌檔案路徑，避免檔案滾動問題"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    process_id = os.getpid()
    return logs_dir / f"trading_system_{timestamp}_{process_id}.log"

def setup_simple_logger():
    """設置簡化版的日誌記錄器，避免檔案滾動問題"""
    
    # 確保日誌目錄存在
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 獲取或創建記錄器
    logger = logging.getLogger("TradingSystem")
    
    # 如果已經設置過，直接返回
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Windows UTF-8 支持
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # 控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    logger.addHandler(console_handler)
    
    # 檔案處理器 - 使用固定檔案名，避免滾動
    try:
        log_file_path = get_simple_log_file_path()
        file_handler = logging.FileHandler(
            log_file_path,
            mode='a',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"簡化版日誌記錄器初始化完成。日誌檔案: {log_file_path}")
        
    except Exception as e:
        logger.error(f"創建檔案日誌處理程序失敗: {e}")
    
    return logger

def cleanup_old_logs(max_files=50):
    """清理舊日誌檔案，保留最新的指定數量檔案"""
    try:
        if not logs_dir.exists():
            return
        
        # 獲取所有日誌檔案
        log_files = []
        for file_path in logs_dir.glob("trading_system_*.log"):
            if file_path.is_file():
                log_files.append((file_path, file_path.stat().st_mtime))
        
        # 按修改時間排序
        log_files.sort(key=lambda x: x[1], reverse=True)
        
        # 刪除超出保留數量的檔案
        if len(log_files) > max_files:
            logger = logging.getLogger("TradingSystem")
            for file_path, _ in log_files[max_files:]:
                try:
                    file_path.unlink()
                    logger.debug(f"已刪除舊日誌檔案: {file_path}")
                except Exception as e:
                    logger.warning(f"刪除舊日誌檔案失敗 {file_path}: {e}")
                    
    except Exception as e:
        logger = logging.getLogger("TradingSystem")
        logger.warning(f"清理舊日誌檔案時發生錯誤: {e}")

# 初始化記錄器
logger = setup_simple_logger()

# 清理舊日誌檔案
cleanup_old_logs()
