# live_trading_system/core/logger_setup.py
"""
全局日誌配置模組
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
from datetime import datetime

def setup_logger():
    """
    設定並返回全局日誌記錄器
    """
    # 設定日誌目錄
    project_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 產生唯一的日誌檔案名稱
    log_file_name = f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = logs_dir / log_file_name

    # 獲取根記錄器
    logger = logging.getLogger("LiveTradingSystem")
    logger.setLevel(logging.DEBUG)

    # 如果已經有處理程序，先清除，避免重複記錄
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- 控制台處理程序 (Console Handler) ---
    # 為 Windows 環境設定 UTF-8 編碼
    if sys.platform == 'win32':
        try:
            if sys.stdout.isatty():
                sys.stdout.reconfigure(encoding='utf-8')
            if sys.stderr.isatty():
                sys.stderr.reconfigure(encoding='utf-8')
        except TypeError:
            # 在非 TTY 環境 (如某些 IDE) 中可能會失敗
            pass

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # 控制台只顯示 INFO 等級以上的日誌
    console_formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(name)s - (%(module)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- 文件處理程序 (File Handler) ---
    # 使用 RotatingFileHandler，當檔案超過 5MB 時會自動輪替，最多保留 5 個備份檔案
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=5*1024*1024,  # 5 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # 檔案中記錄所有 DEBUG 等級以上的日誌
    file_formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info("Logger has been configured successfully.")
    logger.info(f"Log file will be saved to: {log_file_path}")
    
    return logger

# --- 範例 ---
if __name__ == '__main__':
    test_logger = setup_logger()
    test_logger.debug("This is a debug message.")
    test_logger.info("This is an info message.")
    test_logger.warning("This is a warning message.")
    test_logger.error("This is an error message.")
    test_logger.critical("This is a critical message.")
