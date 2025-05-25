# src/common/logger_setup.py
"""
全局日誌配置模組
提供一個統一的日誌記錄器實例供整個專案使用。
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 嘗試從 config 模組導入 LOGS_DIR 和其他可能的配置
# 這裡使用 try-except 是為了在單獨測試此文件時，或者在 config 尚未完全可用時，提供一個後備
try:
    from .config import LOGS_DIR, DEVICE # 假設 DEVICE 也可能影響日誌內容
    LOG_FILE_PATH = LOGS_DIR / "trading_system.log"
except ImportError:
    # 如果無法導入config (例如，在專案結構尚未完全建立時單獨運行此文件)
    # 提供一個後備的日誌路徑，通常這種情況不應在正常運行時發生
    print("警告: 無法從 common.config 導入 LOGS_DIR。日誌將輸出到當前目錄下的 trading_system.log")
    current_file_dir = Path(__file__).resolve().parent
    LOG_FILE_PATH = current_file_dir / "trading_system_fallback.log"
    DEVICE = "cpu" # 後備

# --- 日誌級別 ---
# DEBUG: 詳細的診斷信息，通常只在調試時開啟。
# INFO: 確認事情按預期工作。
# WARNING: 表明發生了一些意外情況，或者在不久的將來會出現一些問題（例如，磁盤空間不足）。程式仍在按預期工作。
# ERROR: 由於一個更嚴重的問題，程式無法執行某些功能。
# CRITICAL: 嚴重錯誤，表明程式本身可能無法繼續運行。

# --- 全局日誌記錄器實例 ---
# 使用專案的根記錄器，或者一個特定的名稱空間
# 這裡我們創建一個名為 'TradingSystem' 的記錄器
logger = logging.getLogger("TradingSystem")
logger.setLevel(logging.DEBUG) # 設置記錄器處理的最低級別，處理程序可以有自己的級別

# --- 防止重複添加處理程序 ---
# 如果記錄器已經有處理程序了（例如，在Jupyter Notebook中多次運行此儲存格），先清除它們
if logger.hasHandlers():
    logger.handlers.clear()

# --- 控制台處理程序 (Console Handler) ---
# 將日誌輸出到標準輸出 (終端)
# 在 Windows 上設置正確的編碼以支持中文
import io
if sys.platform == 'win32':
    # 在 Windows 上使用 UTF-8 編碼的流
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG) # 修改：控制台顯示DEBUG及以上級別的日誌
console_formatter = logging.Formatter(
    "%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)
# 設置控制台處理程序的編碼
if hasattr(console_handler.stream, 'reconfigure'):
    console_handler.stream.reconfigure(encoding='utf-8')
logger.addHandler(console_handler)

# --- 文件處理程序 (File Handler) ---
# 將日誌輸出到文件，並實現日誌輪轉
# RotatingFileHandler 會在日誌文件達到一定大小時自動創建新的日誌文件
# maxBytes: 每個日誌文件的最大大小 (這裡是 5MB)
# backupCount: 保留的舊日誌文件數量
try:
    # 在多進程環境中，使用更安全的日誌配置
    import os
    from logging.handlers import TimedRotatingFileHandler

    # 確保日誌目錄存在
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if os.name == 'nt':  # Windows 系統
        logger.info("檢測到 Windows 系統，將使用標準的 TimedRotatingFileHandler (無 fcntl 文件鎖)。")
        file_handler = TimedRotatingFileHandler(
            LOG_FILE_PATH,
            when='midnight',      # 每天午夜輪轉
            interval=1,           # 每1天輪轉一次
            backupCount=7,        # 保留7天的備份文件
            encoding='utf-8',     # 確保支持中文等字符
            delay=True            # 延遲創建文件，直到第一次寫入
        )
    else:  # 非 Windows 系統 (Unix/Linux/MacOS等)
        try:
            import fcntl
            logger.info("在非 Windows 系統上檢測到 fcntl 模組，將嘗試使用帶文件鎖的 SafeFileHandler。")

            class SafeFileHandler(TimedRotatingFileHandler):
                def emit(self, record):
                    try:
                        if self.stream is None: # 確保 stream 在使用前已打開
                            self.stream = self._open()
                        if self.stream: # 再次檢查 stream 是否成功打開
                            # 在寫入前獲取文件鎖
                            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
                            try:
                                super().emit(record)
                            finally:
                                # 釋放文件鎖
                                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
                        else: # 如果 stream 仍然是 None，則直接調用父類 emit (可能觸發錯誤處理)
                            super().emit(record)
                    except Exception: # 捕獲所有可能的異常，包括 stream 為 None 或鎖定失敗
                        self.handleError(record) # 使用 logging 內建的錯誤處理

            file_handler = SafeFileHandler(
                LOG_FILE_PATH,
                when='midnight',
                interval=1,
                backupCount=7,
                encoding='utf-8',
                delay=True
            )
        except ImportError:
            logger.warning("在非 Windows 系統上導入 fcntl 失敗，將使用標準的 TimedRotatingFileHandler。")
            file_handler = TimedRotatingFileHandler(
                LOG_FILE_PATH,
                when='midnight',
                interval=1,
                backupCount=7,
                encoding='utf-8',
                delay=True
            )
    
    file_handler.setLevel(logging.DEBUG) # 文件日誌記錄所有DEBUG及以上級別的信息
    file_formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(name)s - (%(module)s.%(funcName)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    # 如果文件處理程序創建失敗 (例如，權限問題)
    # 記錄一個錯誤到控制台，並繼續，至少控制台日誌還能工作
    logger.error(f"創建文件日誌處理程序失敗: {e}", exc_info=True)


# --- Streamlit 特定的日誌處理 (可選，如果Streamlit應用需要更精細的日誌展示) ---
# Streamlit 通常會捕獲標準輸出，所以 console_handler 的日誌應該能在Streamlit應用運行時的終端看到。
# 如果需要在Streamlit UI中直接顯示特定格式的日誌，可以創建一個自定義的Handler。
# 例如，一個將日誌消息存儲到一個列表，然後UI可以讀取這個列表來顯示。
# class StreamlitLogHandler(logging.Handler):
#     def __init__(self, max_entries=100):
#         super().__init__()
#         self.logs = []
#         self.max_entries = max_entries
#
#     def emit(self, record):
#         log_entry = self.format(record)
#         self.logs.append(log_entry)
#         if len(self.logs) > self.max_entries:
#             self.logs.pop(0)
#
# # streamlit_log_handler = StreamlitLogHandler()
# # streamlit_log_handler.setLevel(logging.INFO)
# # streamlit_log_handler.setFormatter(console_formatter) # 可以使用與控制台相同的格式
# # logger.addHandler(streamlit_log_handler)
# #
# # def get_streamlit_logs():
# # return streamlit_log_handler.logs


# --- 測試日誌記錄器 ---
# 在模塊首次被導入時，可以打印一條日誌來確認配置成功
# logger.info(f"交易系統日誌記錄器初始化完成。日誌級別: DEBUG (文件), INFO (控制台)。日誌文件: {LOG_FILE_PATH}")
# logger.debug(f"使用的設備 (來自config): {DEVICE}") # 示例如何使用從config導入的變量

# --- 設置 Pytorch 和其他庫的日誌級別 (可選) ---
# 有些庫（如 Pytorch, Matplotlib）可能會產生大量的調試日誌，如果不需要可以調高它們的日誌級別
# logging.getLogger("matplotlib").setLevel(logging.WARNING)
# logging.getLogger("PIL").setLevel(logging.WARNING)
# logging.getLogger("stable_baselines3").setLevel(logging.INFO) # SB3的日誌

# --- 避免日誌傳播到根記錄器 ---
# 如果不希望這個記錄器的日誌同時被根記錄器處理（可能導致重複輸出），可以設置 propagate = False
# logger.propagate = False

if __name__ == "__main__":
    # 這個部分只在直接運行 logger_setup.py 時執行，用於測試日誌配置
    logger.debug("這是一條 DEBUG 級別的日誌 (應該只出現在文件中)。")
    logger.info("這是一條 INFO 級別的日誌 (應該出現在文件和控制台)。")
    logger.warning("這是一條 WARNING 級別的日誌。")
    logger.error("這是一條 ERROR 級別的日誌。")
    logger.critical("這是一條 CRITICAL 級別的日誌。")

    try:
        1 / 0
    except ZeroDivisionError:
        logger.error("捕獲到一個異常 (ZeroDivisionError)", exc_info=True) # exc_info=True 會記錄堆棧跟踪

    print(f"\n請檢查日誌輸出:")
    print(f"1. 控制台輸出。")
    print(f"2. 日誌文件: {LOG_FILE_PATH.resolve()}")