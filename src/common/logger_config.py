# src/common/logger_config.py
"""
日誌配置選擇器
允許用戶在不同的日誌設定之間切換
"""
import os
from pathlib import Path

# 設定檔案路徑
CONFIG_FILE = Path(__file__).parent / "logger_mode.txt"

def set_logger_mode(mode: str):
    """
    設定日誌模式
    mode: "safe" (使用 Windows 安全模式) 或 "simple" (使用簡化模式)
    """
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        f.write(mode.strip().lower())

def get_logger_mode() -> str:
    """
    獲取當前日誌模式
    返回: "safe" 或 "simple"
    """
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                mode = f.read().strip().lower()
                if mode in ['safe', 'simple']:
                    return mode
    except Exception:
        pass
    
    # 預設使用安全模式
    return "safe"

def get_logger():
    """
    根據設定獲取適當的 logger
    """
    mode = get_logger_mode()
    
    if mode == "simple":
        # 使用簡化版日誌
        from .simple_logger import logger
        return logger
    else:
        # 使用安全版日誌（預設）
        from .logger_setup import logger
        return logger

# 設定預設模式為安全模式
if not CONFIG_FILE.exists():
    set_logger_mode("safe")
