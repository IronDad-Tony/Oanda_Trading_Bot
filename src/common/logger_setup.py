# src/common/logger_setup.py
"""
全局日誌配置模組
提供一個統一的日誌記錄器實例供整個專案使用。
"""
import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime
import os
import io # Moved import io to the top
import time # Moved import time to the top
import traceback # Added import traceback to fix NameError

# --- 全局變量確保日誌文件名只生成一次 和 stdout/stderr 只包裝一次 ---
_log_file_path_instance = None
_stdout_wrapped = False # Initialize before use
_stderr_wrapped = False # Initialize before use

# 避免循環導入，直接設置日誌路徑
project_root = Path(__file__).resolve().parent.parent.parent
logs_dir = project_root / "logs"

def get_log_file_path():
    """Return a unique log file path per process run using an environment-based timestamp."""
    global _log_file_path_instance # Ensure we modify the global instance if needed, though current logic re-evals env var
    run_id = os.environ.get('LOG_RUN_ID')
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Added microseconds for more uniqueness
        os.environ['LOG_RUN_ID'] = run_id
    # If _log_file_path_instance is None, or if we want to re-evaluate every call (current behavior):
    _log_file_path_instance = logs_dir / f"trading_system_{run_id}.log"
    return _log_file_path_instance

LOG_FILE_PATH = get_log_file_path()

logs_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("TradingSystem")
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

# --- 控制台處理程序 (Console Handler) ---
if sys.platform == 'win32':
    # This block should be executed only once.
    if not _stdout_wrapped:
        try:
            # Check if sys.stdout is a TTY, if not, it might be already redirected (e.g. in a pipe)
            if sys.stdout.isatty(): 
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            _stdout_wrapped = True
        except Exception as e:
            print(f"Initial Error wrapping stdout: {e}") 
    if not _stderr_wrapped:
        try:
            if sys.stderr.isatty():
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
            _stderr_wrapped = True
        except Exception as e:
            print(f"Initial Error wrapping stderr: {e}")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO) # Changed from DEBUG to INFO for console
console_formatter = logging.Formatter(
    "%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# --- 文件處理程序 (File Handler) ---
try:
    if os.name == 'nt':
        class WindowsSafeRotatingFileHandler(TimedRotatingFileHandler):
            def __init__(self, filename, when='h', interval=1, backupCount=0,
                         encoding=None, delay=False, utc=False, atTime=None):
                # Ensure encoding is passed to parent if not None
                super().__init__(filename, when, interval, backupCount, encoding, delay, utc, atTime)

            def doRollover(self):
                if self.stream:
                    self.stream.close()
                    self.stream = None
                
                current_time_tuple = time.localtime(time.time())
                dfn = self.rotation_filename(self.baseFilename + "." + time.strftime(self.suffix, current_time_tuple))

                max_retries = 5
                retry_delay = 0.2 # Slightly increased delay
                
                for attempt in range(max_retries):
                    try:
                        if os.path.exists(self.baseFilename):
                            if os.path.exists(dfn):
                                try:
                                    os.remove(dfn)
                                except OSError as e_remove:
                                    # Log to stderr if remove fails, as logger might be in inconsistent state
                                    sys.stderr.write(f"WindowsSafeRotatingFileHandler: Could not remove {dfn}: {e_remove}\n")
                                    # If removal fails, we might try to rename over it, or append unique suffix to dfn
                                    # For now, let rename attempt it.
                                    pass 
                            os.rename(self.baseFilename, dfn)
                        break 
                    except (OSError, PermissionError) as e_rename:
                        sys.stderr.write(f"WindowsSafeRotatingFileHandler: Rename attempt {attempt+1} failed for {self.baseFilename} to {dfn}: {e_rename}\n")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (2 ** attempt))
                            continue
                        else:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") 
                            original_base = Path(self.baseFilename).stem
                            new_base_name = f"{original_base}_{timestamp}.log"
                            new_full_path = Path(self.baseFilename).parent / new_base_name
                            sys.stderr.write(f"WindowsSafeRotatingFileHandler: All rename attempts failed. Using new base: {new_full_path}\n")
                            self.baseFilename = str(new_full_path) # Update baseFilename to the new unique name
                            # No need to rename here, new file will be created with _open()
                
                # Cleanup old files based on the original base name pattern if possible, or current if changed
                if self.backupCount > 0:
                    self._cleanup_old_files() 
                
                if not self.delay:
                    try:
                        self.stream = self._open() 
                    except Exception as e_open:
                        sys.stderr.write(f"WindowsSafeRotatingFileHandler: Error reopening stream {self.baseFilename}: {e_open}\n")
                        self.stream = None # Ensure stream is None if open fails
            
            def _cleanup_old_files(self):
                try:
                    log_dir = os.path.dirname(self.baseFilename)
                    # Construct the prefix based on the original naming pattern if possible
                    # This is tricky if baseFilename changed. For simplicity, use current baseFilename stem.
                    base_stem = Path(self.baseFilename).stem.split('_')[0] # Attempt to get original base part
                    log_files = []
                    
                    for filename in os.listdir(log_dir):
                        if filename.startswith(base_stem) and filename.endswith(".log") and Path(filename).suffix != self.suffix:
                            # This logic needs to be careful to match rotated files, e.g., trading_system_YYYYMMDD_HHMMSS.log.YYYY-MM-DD_HH
                            # The default TimedRotatingFileHandler naming is baseFilename + suffix (e.g. .2023-01-01_10)
                            # Let's rely on the parent's getFilesToDelete logic if possible, or simplify.
                            # For now, a simple approach matching the start of the base filename:
                            if filename.startswith(Path(self.baseFilename).name.split('.log')[0]):
                                full_path = os.path.join(log_dir, filename)
                                if os.path.isfile(full_path):
                                    log_files.append((full_path, os.path.getmtime(full_path)))
                    
                    log_files.sort(key=lambda x: x[1], reverse=True)
                    
                    if len(log_files) > self.backupCount:
                        for file_path, _ in log_files[self.backupCount:]:
                            try:
                                os.remove(file_path)
                            except OSError as e_remove_old:
                                sys.stderr.write(f"WindowsSafeRotatingFileHandler: Could not remove old log {file_path}: {e_remove_old}\n")
                                
                except Exception as e_cleanup:
                    sys.stderr.write(f"WindowsSafeRotatingFileHandler: Error during _cleanup_old_files: {e_cleanup}\n")

            def emit(self, record):
                """
                Emit a record.
                Handles rollover safely on Windows and defers to FileHandler.emit.
                The caller (logging.Handler.handle) is expected to call self.handleError if this method raises an exception.
                """
                try:
                    if self.shouldRollover(record):
                        self.doRollover() # Our custom, Windows-safe rollover
                    
                    # logging.FileHandler.emit will call self._open() if self.stream is None.
                    logging.FileHandler.emit(self, record) 
                
                except Exception as e: # Catch any exception from rollover or emit
                    # Log detailed error to stderr, including our class name for clarity
                    # Ensure record and record.getMessage() are accessed safely
                    record_message = "N/A"
                    if record:
                        try:
                            record_message = record.getMessage()
                        except Exception:
                            record_message = "Error getting record message"

                    sys.stderr.write(
                        f"{self.__class__.__name__}: Error during emit or rollover for {self.baseFilename}. "
                        f"Record: {record_message}. Error: {e}\\n"
                        f"Traceback: {traceback.format_exc()}\\n"
                    )
                    # self.handleError(record) # Removed: Let Handler.handle() call this.
                    raise # Re-raise the exception to be caught by logging.Handler.handle()

        file_handler = WindowsSafeRotatingFileHandler(
            LOG_FILE_PATH,
            when='H', 
            interval=1,
            backupCount=24, 
            encoding='utf-8',
            delay=True 
        )
        # Message moved after handler is added
    else:  # Non-Windows
        # ... (existing non-Windows fcntl or standard TimedRotatingFileHandler logic)
        # For brevity, assuming the non-Windows part was okay or needs separate review.
        # Fallback to standard if fcntl is not available or fails.
        try:
            import fcntl
            # logger.info("Non-Windows: Using SafeFileHandler with fcntl.") # Log after handler added
            class SafeFileHandler(TimedRotatingFileHandler):
                def emit(self, record):
                    try:
                        if self.stream is None: self.stream = self._open()
                        if self.stream:
                            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
                            try: super().emit(record)
                            finally: fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
                        else: super().emit(record)
                    except Exception: self.handleError(record)
            file_handler = SafeFileHandler(LOG_FILE_PATH, when='midnight', interval=1, backupCount=7, encoding='utf-8', delay=True)
        except ImportError:
            # logger.warning("Non-Windows: fcntl not available, using standard TimedRotatingFileHandler.") # Log after handler added
            file_handler = TimedRotatingFileHandler(LOG_FILE_PATH, when='midnight', interval=1, backupCount=7, encoding='utf-8', delay=True)
    
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(name)s - (%(module)s.%(funcName)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Log environment detection after handlers are set up
    if os.name == 'nt':
        logger.info("Windows environment detected. Using Windows-safe logging handler.")
    elif 'fcntl' in sys.modules:
        logger.info("Non-Windows: Using SafeFileHandler with fcntl.")
    else:
        logger.warning("Non-Windows: fcntl not available, using standard TimedRotatingFileHandler.")

except Exception as e:
    # Use print for this critical error as logger might be the cause
    print(f"CRITICAL: Failed to create file log handler: {e}\nTraceback: {traceback.format_exc()}\n", file=sys.stderr)


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
logger.propagate = False

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

    # 確保 LOG_FILE_PATH 在 __main__ 中也被正確引用
    current_log_file = get_log_file_path() # 再次調用以確保獲取的是同一個實例
    print(f"\n請檢查日誌輸出:")
    print(f"1. 控制台輸出。")
    print(f"2. 日誌文件: {current_log_file.resolve()}")