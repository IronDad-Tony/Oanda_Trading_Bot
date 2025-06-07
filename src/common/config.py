# src/common/config.py
"""
核心配置文件
儲存專案中常用的固定參數和路徑設置。
"""
import os
from pathlib import Path
import torch
from dotenv import load_dotenv

# 避免循環導入，推遲日誌記錄器的導入

# --- 基礎路徑 ---
# Path(__file__) 會獲取當前文件 (config.py) 的路徑
# .resolve() 將路徑轉換為絕對路徑
# .parent 指向上一級目錄 (common)
# .parent.parent 指向更上一級目錄 (src 的父目錄，即專案根目錄)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# print(f"BASE_DIR in config.py: {BASE_DIR}") # 調試時使用

# 加載 .env 文件中的環境變量
# load_dotenv 會自動尋找專案根目錄下的 .env 文件
dotenv_path = BASE_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    # print(f".env file loaded from: {dotenv_path}") # 調試時使用
else:
    # print(f".env file not found at: {dotenv_path}, please create it.") # 調試時使用
    # 在實際部署或運行時，如果.env不存在，應該有更強的警告或錯誤處理
    pass

# --- OANDA API 配置 ---
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_BASE_URL = os.getenv("OANDA_BASE_URL", "https://api-fxpractice.oanda.com/v3") # 默認為練習帳戶
ACCOUNT_CURRENCY = os.getenv("ACCOUNT_CURRENCY", "AUD") # 您的帳戶基礎貨幣

# --- 數據庫與文件路徑 ---
DATABASE_DIR = BASE_DIR / "data" / "database"
DATABASE_NAME = "oanda_s5_data.db" # S5數據的資料庫名
DATABASE_PATH = DATABASE_DIR / DATABASE_NAME

WEIGHTS_DIR = BASE_DIR / "weights" # 儲存模型權重
LOGS_DIR = BASE_DIR / "logs" # 儲存日誌文件
MMAP_DATA_DIR = BASE_DIR / "data" / "mmap_s5_universal" # 記憶體映射文件的儲存位置

# 確保目錄存在
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MMAP_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- 數據參數 ---
# 初始考慮的交易對列表 (可以從UI選擇，這裡作為預設或後備)
# 為了快速啟動，先用幾個常見的，您可以後續修改或通過UI傳入
DEFAULT_SYMBOLS = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "XAU_USD"]
GRANULARITY = "S5" # 數據時間粒度
MAX_SYMBOLS_ALLOWED = 5 # UI允許選擇的最大交易對數量 (用於Transformer輸入維度等)
PRICE_COLUMNS = ['bid_open', 'bid_high', 'bid_low', 'bid_close',
                 'ask_open', 'ask_high', 'ask_low', 'ask_close', 'volume']
PRICE_TYPES = {'open': ['bid_open', 'ask_open'],
               'high': ['bid_high', 'ask_high'],
               'low': ['bid_low', 'ask_low'],
               'close': ['bid_close', 'ask_close']}


# --- 模型與訓練參數 ---
# Enhanced Transformer 配置 - Phase 3 升級
TIMESTEPS = 128             # 輸入Transformer的時間步長 (序列長度)
TRANSFORMER_MODEL_DIM = 512 # 增強版Transformer維度 (從256提升到512)
TRANSFORMER_NUM_LAYERS = 12  # 增強版層數 (從4提升到12)
TRANSFORMER_NUM_HEADS = 16   # 增強版注意力頭數 (從8提升到16)
TRANSFORMER_FFN_DIM = TRANSFORMER_MODEL_DIM * 4  # FFN維度 (2048)
TRANSFORMER_DROPOUT_RATE = 0.1
TRANSFORMER_LAYER_NORM_EPS = 1e-5
TRANSFORMER_MAX_SEQ_LEN_POS_ENCODING = 5000 # PositionalEncoding 的 max_len
TRANSFORMER_OUTPUT_DIM_PER_SYMBOL = 128     # Transformer處理後，每個symbol的輸出特徵維度

# 增強版Transformer特有配置
ENHANCED_TRANSFORMER_USE_MULTI_SCALE = True    # 啟用多尺度特徵提取器
ENHANCED_TRANSFORMER_USE_CROSS_TIME_FUSION = True  # 啟用跨時間尺度融合
ENHANCED_TRANSFORMER_MULTI_SCALE_KERNELS = [3, 5, 7, 11]  # 多尺度卷積核大小
ENHANCED_TRANSFORMER_TIME_SCALES = [5, 15, 30, 60]  # 跨時間尺度

# src/common/config.py
# ... (其他配置) ...

# --- Trainer 與評估配置 ---
TRAINER_DEFAULT_TOTAL_TIMESTEPS = 1_000_000 # 訓練器默認的總訓練步數
TRAINER_MODEL_NAME_PREFIX = "sac_universal_trader" # 保存模型文件的前綴
TRAINER_SAVE_FREQ_STEPS = 20000             # 回調中定期保存模型的頻率 (步數)
TRAINER_EVAL_FREQ_STEPS = 10000             # 回調中執行評估的頻率 (步數)
TRAINER_N_EVAL_EPISODES = 3                 # 每次評估運行的episode數量
TRAINER_DETERMINISTIC_EVAL = True           # 評估時是否使用確定性動作

# 早停相關配置 (用於 UniversalCheckpointCallback)
EARLY_STOPPING_PATIENCE = 10 # 連續10次評估無顯著改善則早停
EARLY_STOPPING_MIN_DELTA_PERCENT = 0.1 # 至少改善 0.1% 才算顯著 (相對於最佳值的百分比)
EARLY_STOPPING_MIN_EVALS = 20 # 至少評估20次後才開始檢查早停

# Transformer 權重範數記錄頻率 (用於 UniversalCheckpointCallback)
# 設定為每步記錄L2範數和梯度範數
LOG_TRANSFORMER_NORM_FREQ_STEPS = 1  # 每1步記錄一次（即每步都記錄）

# 劃分訓練集和驗證集的時間點 (ISO格式 UTC)
# 示例：假設我們總共有從 2023-01-01 到 2024-01-01 的數據
# 我們可以用前10個月做訓練，後2個月做驗證
# 這些值需要您根據實際下載的數據範圍來設定
# 為了能跑通，我們先設一些能用的值，您之後需要根據您的數據調整
# 假設我們下載了從 2024-01-01 到 2024-05-24 的數據
# 訓練集: 2024-01-01T00:00:00Z 到 2024-04-30T23:59:59Z
# 驗證集: 2024-05-01T00:00:00Z 到 2024-05-22T23:59:59Z (假設23,24日市場關閉或數據不穩定)
# 這些日期需要保證數據庫中有對應的數據，否則Dataset會是空的
# 為了讓 __main__ 測試能跑，我們先用一個較短且固定的範圍
DEFAULT_TRAIN_START_ISO = "2024-05-20T00:00:00Z"
DEFAULT_TRAIN_END_ISO = "2024-05-21T23:59:59Z" # 2天訓練數據
DEFAULT_EVAL_START_ISO = "2024-05-22T00:00:00Z"
DEFAULT_EVAL_END_ISO = "2024-05-22T11:59:59Z"  # 12小時驗證數據

# 確保您的 WEIGHTS_DIR/best_model 目錄存在，如果Callback要保存最佳模型到那裡
BEST_MODEL_SUBDIR = "best_model" # 相對於 WEIGHTS_DIR

# SAC 相關
SAC_GAMMA = 0.95             # 折扣因子
SAC_LEARNING_RATE = 3e-5     # 學習率
SAC_BATCH_SIZE = 64          # 根據用戶要求設定批次大小
SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR = 500 # 每個交易對的緩衝區大小因子，恢復為 500
SAC_LEARNING_STARTS_FACTOR = 200 # 學習開始前收集的最小樣本數因子 (總樣本 = N_symbols * BATCH_SIZE * factor)
SAC_TRAIN_FREQ_STEPS = 32    # 減少訓練頻率以提高效率
SAC_GRADIENT_STEPS = 16      # 每次訓練迭代執行多少梯度步數，設定為 16
SAC_ENT_COEF = 'auto'        # 熵系數 ('auto' 或 float)
SAC_TARGET_UPDATE_INTERVAL = 1 # Target network 更新頻率 (相對於gradient_steps)
SAC_TAU = 0.005              # Target network 軟更新系數

# 訓練流程相關
INITIAL_CAPITAL = 100000.0              # 初始模擬資金 (以ACCOUNT_CURRENCY計價)
MAX_EPISODE_STEPS_DEFAULT = 20000       # 每個訓練episode的最大步數 (可調整)
TOTAL_TRAINING_TIMESTEPS_TARGET = 1_000_000 # 總訓練目標步數 (可根據需要調整)
SAVE_MODEL_INTERVAL_STEPS = 50000       # 模型保存間隔 (按總訓練steps)

# --- GPU優化配置 ---
def setup_gpu_optimization():
    """設置GPU優化配置"""
    try:
        if torch.cuda.is_available():
            # 啟用TensorFloat-32 (TF32) 以提高性能
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # 設置cuDNN基準測試以優化卷積性能
            torch.backends.cudnn.benchmark = True
            
            # 使用PyTorch預設的記憶體管理策略，讓CUDA自動管理
            # 移除自定義的記憶體分配限制，讓系統動態調整
            print("GPU優化：使用PyTorch預設記憶體管理策略")
            
            # 啟用混合精度訓練
            return True
    except Exception as e:
        print(f"Warning: GPU optimization setup failed: {e}")
        print("Continuing with CPU-only mode...")
    return False

# 設備配置
GPU_OPTIMIZED = setup_gpu_optimization()
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"✅ 檢測到CUDA GPU，設備設置為: {DEVICE}")
    print(f"   - GPU名稱: {torch.cuda.get_device_name(0)}")
    print(f"   - GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    DEVICE = "cpu"
    print("⚠️  未檢測到CUDA GPU，使用CPU模式")

# 混合精度訓練 (如果GPU支持且希望加速)
USE_AMP = GPU_OPTIMIZED

# 數值穩定性配置
GRADIENT_CLIP_NORM = 1.0            # 梯度裁剪範數
ENABLE_GRADIENT_CLIPPING = True     # 是否啟用梯度裁剪
NAN_CHECK_FREQUENCY = 500           # NaN檢查頻率
AMP_LOSS_SCALE_INIT = 2**15         # AMP初始損失縮放因子 (降低以提高穩定性)

# 如果檢測到數值不穩定，自動禁用AMP
AUTO_DISABLE_AMP_ON_INSTABILITY = True

# 早期停止容忍的NaN檢測次數
MAX_NAN_TOLERANCE = 10

# --- 日誌配置 ---
# AMP 轉換日誌控制 - 減少重複的 DEBUG 訊息頻率
AMP_CONVERSION_LOG_INTERVAL = 100   # 每 N 次 AMP 轉換才記錄一次日誌訊息

# --- 風險管理參數 ---
MAX_ACCOUNT_RISK_PERCENTAGE = 0.02  # 單筆交易最大可承受賬戶風險百分比 (例如 2%)
MAX_POSITION_SIZE_PERCENTAGE_OF_EQUITY = 0.10 # 單個倉位最大名義價值佔總淨值的百分比 (例如 10%)
ATR_PERIOD = 14                     # 計算ATR的週期
ATR_STOP_LOSS_MULTIPLIER = 2.0      # ATR止損倍數
STOP_LOSS_ATR_MULTIPLIER = 2.0      # ATR止損倍數 <--- 確保這一行存在且未註釋!
# DEFAULT_PIP_VALUE = {               # 主要貨幣對的近似點值 (相對於USD, 1標準手)
#     "AUD_USD": 10.0, "EUR_USD": 10.0, "GBP_USD": 10.0,
#     "USD_CAD": 7.0, "USD_CHF": 10.0, "USD_JPY": 9.0, # 近似值，實際會變化
#     "XAU_USD": 10.0 # 黃金
# }
# 實際點值和保證金率會從OANDA API獲取，這裡作為無法獲取時的後備或參考
TRADE_COMMISSION_PERCENTAGE = 0.0  # 名義交易手續費百分比 (例如 0.01%)
                                      # 如果OANDA點差已包含手續費，可以設為 0.0

# OANDA 強制平倉保證金水平 (Margin Closeout Percentage)
OANDA_MARGIN_CLOSEOUT_LEVEL = 0.50 # 50%

# --- 實盤交易相關 ---
LIVE_TRADING_POLL_INTERVAL = 5 # 實盤交易時輪詢市場數據和執行決策的間隔（秒）
OANDA_MAX_BATCH_CANDLES = 4800 # OANDA API 單次請求最大蠟燭數
OANDA_REQUEST_INTERVAL = 0.1   # API 請求間隔 (秒)，避免過於頻繁
OANDA_API_TIMEOUT_SECONDS = 20 # <--- 新增：API請求的通用超時時間 (秒)

# --- UI 相關 ---
STREAMLIT_THEME = "dark" # Streamlit UI主題 ("light" 或 "dark")
PLOT_REFRESH_INTERVAL_SECONDS = 5 # UI圖表刷新間隔
MAX_TRADE_LOG_DISPLAY = 100     # UI中顯示最近交易的條數

# --- 貨幣轉換相關 ---
# 假設基礎匯率對，用於將非USD計價的symbol盈虧轉換為USD，再轉換為ACCOUNT_CURRENCY
# 這些會在 LiveTrader 中嘗試從OANDA獲取實時匯率，這裡僅作結構示例
# KEY_FOREX_PAIRS_FOR_CONVERSION = ["AUD/USD", "EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "USD/CHF"]

# --- 環境設置 (PyTorch) ---
# 統一的CUDA內存分配配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"

# 這些在您的舊代碼中已有，可以保留或根據需要調整
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 主要用於調試，生產環境中可考慮移除
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.set_float32_matmul_precision('high') # PyTorch 1.12+

# 驗證API金鑰是否已加載
if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
    print(f"警告: OANDA_API_KEY 或 OANDA_ACCOUNT_ID 未在 .env 文件中找到或未正確加載。請檢查 {dotenv_path}。")
    # 在實際運行時，這裡可能需要更強的錯誤處理，例如直接退出程序
    # raise ValueError("OANDA API Key or Account ID not configured.") # 或者
    # sys.exit("OANDA API Key or Account ID not configured.")


# --- 輔助函數 (可選，放在這裡方便全局訪問) ---
def get_granularity_seconds(granularity_str: str) -> int:
    """根據OANDA的粒度字符串返回對應的秒數"""
    if granularity_str.startswith("S"):
        return int(granularity_str[1:])
    elif granularity_str.startswith("M"):
        return int(granularity_str[1:]) * 60
    elif granularity_str.startswith("H"):
        return int(granularity_str[1:]) * 3600
    elif granularity_str == "D":
        return 24 * 3600
    elif granularity_str == "W":
        return 7 * 24 * 3600
    elif granularity_str == "Mo": # OANDA的月是 'M'，但我們這裡用 'Mo' 避免與分鐘衝突
        return 30 * 24 * 3600 # 近似值
    raise ValueError(f"未知的粒度: {granularity_str}")

GRANULARITY_SECONDS = get_granularity_seconds(GRANULARITY)

# --- 輸出一些關鍵配置以供檢查 (可選) ---
# print(f"Configuration loaded. Base directory: {BASE_DIR}")
# print(f"Using device: {DEVICE}")
# print(f"Database path: {DATABASE_PATH}")
# print(f"Default symbols: {DEFAULT_SYMBOLS}")
# print(f"Granularity: {GRANULARITY} ({GRANULARITY_SECONDS} seconds)")

# --- 在文件末尾導入日誌記錄器以避免循環導入 ---
try:
    from .logger_setup import logger
except ImportError:
    # 如果相對導入失敗，嘗試絕對導入
    try:
        from src.common.logger_setup import logger
    except ImportError:
        # 最終後備：創建一個簡單的日誌記錄器
        import logging
        logger = logging.getLogger("config_fallback")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)