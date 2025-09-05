# src/common/config.py
"""
æ ¸å¿ƒé…ç½®æ–‡ä»¶
å„²å­˜å°ˆæ¡ˆä¸­å¸¸ç”¨çš„å›ºå®šåƒæ•¸å’Œè·¯å¾‘è¨­ç½®ã€‚
"""
import os
from pathlib import Path
import torch
from dotenv import load_dotenv

# é¿å…å¾ªç’°å°Žå…¥ï¼ŒæŽ¨é²æ—¥èªŒè¨˜éŒ„å™¨çš„å°Žå…¥

# --- åŸºç¤Žè·¯å¾‘ ---
# Path(__file__) æœƒç²å–ç•¶å‰æ–‡ä»¶ (config.py) çš„è·¯å¾‘
# .resolve() å°‡è·¯å¾‘è½‰æ›ç‚ºçµ•å°è·¯å¾‘
# .parent æŒ‡å‘ä¸Šä¸€ç´šç›®éŒ„ (common)
# .parent.parent.parent.parent.parent æŒ‡å‘å°ˆæ¡ˆæ ¹ç›®éŒ„
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
print(f"BASE_DIR in config.py: {BASE_DIR}") # èª¿è©¦æ™‚ä½¿ç”¨

# æ–°å¢žï¼šæŒ‡å‘ enhanced_transformer_config.json çš„è·¯å¾‘
ENHANCED_TRANSFORMER_CONFIG_PATH = BASE_DIR / "configs" / "enhanced_transformer_config.json"

# æ–°å¢žï¼šæŒ‡å‘ quantum_strategy_config.json çš„è·¯å¾‘
QUANTUM_STRATEGY_CONFIG_PATH = BASE_DIR / "configs" / "quantum_strategy_config.json"

# æ–°å¢žï¼šæŒ‡å‘ enhanced_model_config.json çš„è·¯å¾‘
ENHANCED_MODEL_CONFIG_PATH = BASE_DIR / "configs" / "enhanced_model_config.json"

# --- Quantum Strategy é…ç½® ---
QUANTUM_STRATEGY_NUM_STRATEGIES = 28  # ç¸½ç­–ç•¥æ•¸é‡ï¼Œæ ¹æ“š strategies/__init__.py è‡ªå‹•æ›´æ–°æˆ–æ‰‹å‹•ç¢ºèª
QUANTUM_STRATEGY_DROPOUT_RATE = 0.1 # Dropout rate for quantum strategy layer
QUANTUM_STRATEGY_INITIAL_TEMPERATURE = 1.0 # Initial temperature for Gumbel-Softmax
QUANTUM_STRATEGY_USE_GUMBEL_SOFTMAX = True # Whether to use Gumbel-Softmax for strategy selection
QUANTUM_ADAPTIVE_LR = 0.0001 # Learning rate for the quantum strategy layer's adaptive weights
QUANTUM_PERFORMANCE_EMA_ALPHA = 0.1 # EMA alpha for tracking strategy performance

# åŠ è¼‰ .env æ–‡ä»¶ä¸­çš„ç’°å¢ƒè®Šé‡
# load_dotenv æœƒè‡ªå‹•å°‹æ‰¾å°ˆæ¡ˆæ ¹ç›®éŒ„ä¸‹çš„ .env æ–‡ä»¶
dotenv_path = BASE_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f".env file loaded from: {dotenv_path}") # èª¿è©¦æ™‚ä½¿ç”¨
else:
    print(f".env file not found at: {dotenv_path}, please create it.") # èª¿è©¦æ™‚ä½¿ç”¨
    # åœ¨å¯¦éš›éƒ¨ç½²æˆ–é‹è¡Œæ™‚ï¼Œå¦‚æžœ.envä¸å­˜åœ¨ï¼Œæ‡‰è©²æœ‰æ›´å¼·çš„è­¦å‘Šæˆ–éŒ¯èª¤è™•ç†
    pass

# --- OANDA API é…ç½® ---
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice").lower()
OANDA_BASE_URL = os.getenv("OANDA_BASE_URL") or ("https://api-fxtrade.oanda.com/v3" if OANDA_ENVIRONMENT=="live" else "https://api-fxpractice.oanda.com/v3")
ACCOUNT_CURRENCY = os.getenv("ACCOUNT_CURRENCY", "AUD") # æ‚¨çš„å¸³æˆ¶åŸºç¤Žè²¨å¹£

# --- æ•¸æ“šåº«èˆ‡æ–‡ä»¶è·¯å¾‘ ---
DATABASE_DIR = BASE_DIR / "data" / "database"
DATABASE_NAME = "oanda_s5_data.db" # S5æ•¸æ“šçš„è³‡æ–™åº«å
DATABASE_PATH = DATABASE_DIR / DATABASE_NAME

WEIGHTS_DIR = BASE_DIR / "weights" # å„²å­˜æ¨¡åž‹æ¬Šé‡
LOGS_DIR = BASE_DIR / "logs" # å„²å­˜æ—¥èªŒæ–‡ä»¶
MMAP_DATA_DIR = BASE_DIR / "data" / "mmap_s5_universal" # è¨˜æ†¶é«”æ˜ å°„æ–‡ä»¶çš„å„²å­˜ä½ç½®

# ç¢ºä¿ç›®éŒ„å­˜åœ¨
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MMAP_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- æ•¸æ“šåƒæ•¸ ---
# åˆå§‹è€ƒæ…®çš„äº¤æ˜“å°åˆ—è¡¨ (å¯ä»¥å¾žUIé¸æ“‡ï¼Œé€™è£¡ä½œç‚ºé è¨­æˆ–å¾Œå‚™)
# ç‚ºäº†å¿«é€Ÿå•Ÿå‹•ï¼Œå…ˆç”¨å¹¾å€‹å¸¸è¦‹çš„ï¼Œæ‚¨å¯ä»¥å¾ŒçºŒä¿®æ”¹æˆ–é€šéŽUIå‚³å…¥
DEFAULT_SYMBOLS = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "XAU_USD"]
GRANULARITY = "S5" # æ•¸æ“šæ™‚é–“ç²’åº¦
MAX_SYMBOLS_ALLOWED = 10 # UIå…è¨±é¸æ“‡çš„æœ€å¤§äº¤æ˜“å°æ•¸é‡ (ç”¨æ–¼Transformerè¼¸å…¥ç¶­åº¦ç­‰)
PRICE_COLUMNS = ['bid_open', 'bid_high', 'bid_low', 'bid_close',
                 'ask_open', 'ask_high', 'ask_low', 'ask_close', 'volume']
PRICE_TYPES = {'open': ['bid_open', 'ask_open'],
               'high': ['bid_high', 'ask_high'],
               'low': ['bid_low', 'ask_low'],
               'close': ['bid_close', 'ask_close']}


# --- æ¨¡åž‹èˆ‡è¨“ç·´åƒæ•¸ ---
# Large Transformer é…ç½® - Phase 4 å‡ç´š
TIMESTEPS = 128             # è¼¸å…¥Transformerçš„æ™‚é–“æ­¥é•· (åºåˆ—é•·åº¦)
TRANSFORMER_MODEL_DIM = 512 # Large Transformerç¶­åº¦ (å¾ž512æå‡åˆ°768)
TRANSFORMER_NUM_LAYERS = 12  # Largeæ¨¡åž‹å±¤æ•¸ (å¾ž12æå‡åˆ°16)
TRANSFORMER_NUM_HEADS = 16   # Largeæ¨¡åž‹æ³¨æ„åŠ›é ­æ•¸ (å¾ž16æå‡åˆ°24)
TRANSFORMER_FFN_DIM = TRANSFORMER_MODEL_DIM * 2  # FFNç¶­åº¦ (3072)
TRANSFORMER_DROPOUT_RATE = 0.1
TRANSFORMER_LAYER_NORM_EPS = 1e-5
TRANSFORMER_MAX_SEQ_LEN_POS_ENCODING = 5000 # PositionalEncoding çš„ max_len
TRANSFORMER_OUTPUT_DIM_PER_SYMBOL = 192     # Transformerè™•ç†å¾Œï¼Œæ¯å€‹symbolçš„è¼¸å‡ºç‰¹å¾µç¶­åº¦ (å‹•æ…‹é©æ‡‰)

# å¢žå¼·ç‰ˆTransformerç‰¹æœ‰é…ç½®
ENHANCED_TRANSFORMER_USE_MULTI_SCALE = True    # å•Ÿç”¨å¤šå°ºåº¦ç‰¹å¾µæå–å™¨
ENHANCED_TRANSFORMER_USE_CROSS_TIME_FUSION = True  # å•Ÿç”¨è·¨æ™‚é–“å°ºåº¦èžåˆ
ENHANCED_TRANSFORMER_MULTI_SCALE_KERNELS = [3, 5, 7, 11]  # å¤šå°ºåº¦å·ç©æ ¸å¤§å°
ENHANCED_TRANSFORMER_TIME_SCALES = [5, 15, 30, 60]  # è·¨æ™‚é–“å°ºåº¦

# æ–°å¢žï¼šFourier å’Œ Wavelet ç‰¹å¾µé…ç½®
FOURIER_NUM_MODES = 16  # Fourier ç‰¹å¾µçš„æ¨¡å¼æ•¸é‡ (å¯èª¿æ•´)
WAVELET_LEVELS = 4      # Wavelet åˆ†è§£çš„å±¤ç´š (å¯èª¿æ•´)
WAVELET_NAME = 'db4'    # Wavelet æ¯æ³¢åç¨± (å¯èª¿æ•´, e.g., 'db4', 'haar', 'sym5')


# src/common/config.py
# ... (å…¶ä»–é…ç½®) ...

# --- Trainer èˆ‡è©•ä¼°é…ç½® ---
TRAINER_DEFAULT_TOTAL_TIMESTEPS = 1_000_000 # è¨“ç·´å™¨é»˜èªçš„ç¸½è¨“ç·´æ­¥æ•¸
TRAINER_MODEL_NAME_PREFIX = "sac_universal_trader" # ä¿å­˜æ¨¡åž‹æ–‡ä»¶çš„å‰ç¶´
TRAINER_SAVE_FREQ_STEPS = 20000             # å›žèª¿ä¸­å®šæœŸä¿å­˜æ¨¡åž‹çš„é »çŽ‡ (æ­¥æ•¸)
TRAINER_EVAL_FREQ_STEPS = 10000             # å›žèª¿ä¸­åŸ·è¡Œè©•ä¼°çš„é »çŽ‡ (æ­¥æ•¸)
TRAINER_N_EVAL_EPISODES = 3                 # æ¯æ¬¡è©•ä¼°é‹è¡Œçš„episodeæ•¸é‡
TRAINER_DETERMINISTIC_EVAL = True           # è©•ä¼°æ™‚æ˜¯å¦ä½¿ç”¨ç¢ºå®šæ€§å‹•ä½œ

# æ—©åœç›¸é—œé…ç½® (ç”¨æ–¼ UniversalCheckpointCallback)
EARLY_STOPPING_PATIENCE = 10 # é€£çºŒ10æ¬¡è©•ä¼°ç„¡é¡¯è‘—æ”¹å–„å‰‡æ—©åœ
EARLY_STOPPING_MIN_DELTA_PERCENT = 0.1 # è‡³å°‘æ”¹å–„ 0.1% æ‰ç®—é¡¯è‘— (ç›¸å°æ–¼æœ€ä½³å€¼çš„ç™¾åˆ†æ¯”)
EARLY_STOPPING_MIN_EVALS = 20 # è‡³å°‘è©•ä¼°20æ¬¡å¾Œæ‰é–‹å§‹æª¢æŸ¥æ—©åœ

# Transformer æ¬Šé‡ç¯„æ•¸è¨˜éŒ„é »çŽ‡ (ç”¨æ–¼ UniversalCheckpointCallback)
# è¨­å®šç‚ºæ¯æ­¥è¨˜éŒ„L2ç¯„æ•¸å’Œæ¢¯åº¦ç¯„æ•¸
LOG_TRANSFORMER_NORM_FREQ_STEPS = 1  # æ¯1æ­¥è¨˜éŒ„ä¸€æ¬¡ï¼ˆå³æ¯æ­¥éƒ½è¨˜éŒ„ï¼‰

# åŠƒåˆ†è¨“ç·´é›†å’Œé©—è­‰é›†çš„æ™‚é–“é»ž (ISOæ ¼å¼ UTC)
# ç¤ºä¾‹ï¼šå‡è¨­æˆ‘å€‘ç¸½å…±æœ‰å¾ž 2023-01-01 åˆ° 2024-01-01 çš„æ•¸æ“š
# æˆ‘å€‘å¯ä»¥ç”¨å‰10å€‹æœˆåšè¨“ç·´ï¼Œå¾Œ2å€‹æœˆåšé©—è­‰
# é€™äº›å€¼éœ€è¦æ‚¨æ ¹æ“šå¯¦éš›ä¸‹è¼‰çš„æ•¸æ“šç¯„åœä¾†è¨­å®š
# ç‚ºäº†èƒ½è·‘é€šï¼Œæˆ‘å€‘å…ˆè¨­ä¸€äº›èƒ½ç”¨çš„å€¼ï¼Œæ‚¨ä¹‹å¾Œéœ€è¦æ ¹æ“šæ‚¨çš„æ•¸æ“šèª¿æ•´
# å‡è¨­æˆ‘å€‘ä¸‹è¼‰äº†å¾ž 2024-01-01 åˆ° 2024-05-24 çš„æ•¸æ“š
# è¨“ç·´é›†: 2024-01-01T00:00:00Z åˆ° 2024-04-30T23:59:59Z
# é©—è­‰é›†: 2024-05-01T00:00:00Z åˆ° 2024-05-22T23:59:59Z (å‡è¨­23,24æ—¥å¸‚å ´é—œé–‰æˆ–æ•¸æ“šä¸ç©©å®š)
# é€™äº›æ—¥æœŸéœ€è¦ä¿è­‰æ•¸æ“šåº«ä¸­æœ‰å°æ‡‰çš„æ•¸æ“šï¼Œå¦å‰‡Datasetæœƒæ˜¯ç©ºçš„
# ç‚ºäº†è®“ __main__ æ¸¬è©¦èƒ½è·‘ï¼Œæˆ‘å€‘å…ˆç”¨ä¸€å€‹è¼ƒçŸ­ä¸”å›ºå®šçš„ç¯„åœ
DEFAULT_TRAIN_START_ISO = "2024-05-20T00:00:00Z"
DEFAULT_TRAIN_END_ISO = "2024-05-21T23:59:59Z" # 2å¤©è¨“ç·´æ•¸æ“š
DEFAULT_EVAL_START_ISO = "2024-05-22T00:00:00Z"
DEFAULT_EVAL_END_ISO = "2024-05-22T11:59:59Z"  # 12å°æ™‚é©—è­‰æ•¸æ“š

# ç¢ºä¿æ‚¨çš„ WEIGHTS_DIR/best_model ç›®éŒ„å­˜åœ¨ï¼Œå¦‚æžœCallbackè¦ä¿å­˜æœ€ä½³æ¨¡åž‹åˆ°é‚£è£¡
BEST_MODEL_SUBDIR = "best_model" # ç›¸å°æ–¼ WEIGHTS_DIR

# SAC ç›¸é—œ
SAC_GAMMA = 0.95             # æŠ˜æ‰£å› å­
SAC_LEARNING_RATE = 3e-5     # å­¸ç¿’çŽ‡
SAC_BATCH_SIZE = 102          # æ ¹æ“šç”¨æˆ¶è¦æ±‚è¨­å®šæ‰¹æ¬¡å¤§å°
SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR = 500 # æ¯å€‹äº¤æ˜“å°çš„ç·©è¡å€å¤§å°å› å­ï¼Œæ¢å¾©ç‚º 500
SAC_LEARNING_STARTS_FACTOR = 200 # å­¸ç¿’é–‹å§‹å‰æ”¶é›†çš„æœ€å°æ¨£æœ¬æ•¸å› å­ (ç¸½æ¨£æœ¬ = N_symbols * BATCH_SIZE * factor)
SAC_TRAIN_FREQ_STEPS = 32    # æ¸›å°‘è¨“ç·´é »çŽ‡ä»¥æé«˜æ•ˆçŽ‡
SAC_GRADIENT_STEPS = 16      # æ¯æ¬¡è¨“ç·´è¿­ä»£åŸ·è¡Œå¤šå°‘æ¢¯åº¦æ­¥æ•¸ï¼Œè¨­å®šç‚º 16
SAC_ENT_COEF = 'auto'        # ç†µç³»æ•¸ ('auto' æˆ– float)
SAC_TARGET_UPDATE_INTERVAL = 1 # Target network æ›´æ–°é »çŽ‡ (ç›¸å°æ–¼gradient_steps)
SAC_TAU = 0.005              # Target network è»Ÿæ›´æ–°ç³»æ•¸

# è¨“ç·´æµç¨‹ç›¸é—œ
INITIAL_CAPITAL = 100000.0              # åˆå§‹æ¨¡æ“¬è³‡é‡‘ (ä»¥ACCOUNT_CURRENCYè¨ˆåƒ¹)
MAX_EPISODE_STEPS_DEFAULT = 100000       # æ¯å€‹è¨“ç·´episodeçš„æœ€å¤§æ­¥æ•¸ (å¯èª¿æ•´)
TOTAL_TRAINING_TIMESTEPS_TARGET = 10_000_000 # ç¸½è¨“ç·´ç›®æ¨™æ­¥æ•¸ (å¯æ ¹æ“šéœ€è¦èª¿æ•´)
SAVE_MODEL_INTERVAL_STEPS = 2000       # æ¨¡åž‹ä¿å­˜é–“éš” (æŒ‰ç¸½è¨“ç·´steps)

# --- GPUå„ªåŒ–é…ç½® ---
def setup_gpu_optimization():
    """è¨­ç½®GPUå„ªåŒ–é…ç½®"""
    try:
        if torch.cuda.is_available():
            # å•Ÿç”¨TensorFloat-32 (TF32) ä»¥æé«˜æ€§èƒ½
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # è¨­ç½®cuDNNåŸºæº–æ¸¬è©¦ä»¥å„ªåŒ–å·ç©æ€§èƒ½
            torch.backends.cudnn.benchmark = True
            
            # ä½¿ç”¨PyTorché è¨­çš„è¨˜æ†¶é«”ç®¡ç†ç­–ç•¥ï¼Œè®“CUDAè‡ªå‹•ç®¡ç†
            # ç§»é™¤è‡ªå®šç¾©çš„è¨˜æ†¶é«”åˆ†é…é™åˆ¶ï¼Œè®“ç³»çµ±å‹•æ…‹èª¿æ•´
            print("GPUå„ªåŒ–ï¼šä½¿ç”¨PyTorché è¨­è¨˜æ†¶é«”ç®¡ç†ç­–ç•¥")
            
            # å•Ÿç”¨æ··åˆç²¾åº¦è¨“ç·´
            return True
    except Exception as e:
        print(f"Warning: GPU optimization setup failed: {e}")
        print("Continuing with CPU-only mode...")
    return False

# è¨­å‚™é…ç½®
GPU_OPTIMIZED = setup_gpu_optimization()
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"CUDA GPU detected, device set to: {DEVICE}")
    print(f"   - GPU name: {torch.cuda.get_device_name(0)}")
    print(f"   - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    DEVICE = "cpu"
    print("CUDA GPU not detected, using CPU mode")

# æ··åˆç²¾åº¦è¨“ç·´ (å¦‚æžœGPUæ”¯æŒä¸”å¸Œæœ›åŠ é€Ÿ)
USE_AMP = GPU_OPTIMIZED

# --- Overrides to align main training system with upgraded live pipeline ---
# Ensure global sequence length aligns with 256-step window
try:
    TIMESTEPS = 256  # override default 128 for main training system
except Exception:
    pass

# æ•¸å€¼ç©©å®šæ€§é…ç½®
GRADIENT_CLIP_NORM = 1.0            # æ¢¯åº¦è£å‰ªç¯„æ•¸
ENABLE_GRADIENT_CLIPPING = True     # æ˜¯å¦å•Ÿç”¨æ¢¯åº¦è£å‰ª
NAN_CHECK_FREQUENCY = 500           # NaNæª¢æŸ¥é »çŽ‡
AMP_LOSS_SCALE_INIT = 2**15         # AMPåˆå§‹æå¤±ç¸®æ”¾å› å­ (é™ä½Žä»¥æé«˜ç©©å®šæ€§)

# å¦‚æžœæª¢æ¸¬åˆ°æ•¸å€¼ä¸ç©©å®šï¼Œè‡ªå‹•ç¦ç”¨AMP
AUTO_DISABLE_AMP_ON_INSTABILITY = True

# æ—©æœŸåœæ­¢å®¹å¿çš„NaNæª¢æ¸¬æ¬¡æ•¸
MAX_NAN_TOLERANCE = 10

# --- æ—¥èªŒé…ç½® ---
# AMP è½‰æ›æ—¥èªŒæŽ§åˆ¶ - æ¸›å°‘é‡è¤‡çš„ DEBUG è¨Šæ¯é »çŽ‡
AMP_CONVERSION_LOG_INTERVAL = 100   # æ¯ N æ¬¡ AMP è½‰æ›æ‰è¨˜éŒ„ä¸€æ¬¡æ—¥èªŒè¨Šæ¯

# --- é¢¨éšªç®¡ç†åƒæ•¸ ---
MAX_ACCOUNT_RISK_PERCENTAGE = 0.02  # å–®ç­†äº¤æ˜“æœ€å¤§å¯æ‰¿å—è³¬æˆ¶é¢¨éšªç™¾åˆ†æ¯” (ä¾‹å¦‚ 2%)
MAX_POSITION_SIZE_PERCENTAGE_OF_EQUITY = 0.10 # å–®å€‹å€‰ä½æœ€å¤§åç¾©åƒ¹å€¼ä½”ç¸½æ·¨å€¼çš„ç™¾åˆ†æ¯” (ä¾‹å¦‚ 10%)
ATR_PERIOD = 28                     # è¨ˆç®—ATRçš„é€±æœŸ
ATR_STOP_LOSS_MULTIPLIER = 3.5      # ATRæ­¢æå€æ•¸
STOP_LOSS_ATR_MULTIPLIER = 3.5      # ATRæ­¢æå€æ•¸ <--- ç¢ºä¿é€™ä¸€è¡Œå­˜åœ¨ä¸”æœªè¨»é‡‹!
# DEFAULT_PIP_VALUE = {               # ä¸»è¦è²¨å¹£å°çš„è¿‘ä¼¼é»žå€¼ (ç›¸å°æ–¼USD, 1æ¨™æº–æ‰‹)
#     "AUD_USD": 10.0, "EUR_USD": 10.0, "GBP_USD": 10.0,
#     "USD_CAD": 7.0, "USD_CHF": 10.0, "USD_JPY": 9.0, # è¿‘ä¼¼å€¼ï¼Œå¯¦éš›æœƒè®ŠåŒ–
#     "XAU_USD": 10.0 # é»ƒé‡‘
# }
# å¯¦éš›é»žå€¼å’Œä¿è­‰é‡‘çŽ‡æœƒå¾žOANDA APIç²å–ï¼Œé€™è£¡ä½œç‚ºç„¡æ³•ç²å–æ™‚çš„å¾Œå‚™æˆ–åƒè€ƒ
TRADE_COMMISSION_PERCENTAGE = 0.0  # åç¾©äº¤æ˜“æ‰‹çºŒè²»ç™¾åˆ†æ¯” (ä¾‹å¦‚ 0.01%)
                                      # å¦‚æžœOANDAé»žå·®å·²åŒ…å«æ‰‹çºŒè²»ï¼Œå¯ä»¥è¨­ç‚º 0.0

# OANDA å¼·åˆ¶å¹³å€‰ä¿è­‰é‡‘æ°´å¹³ (Margin Closeout Percentage)
OANDA_MARGIN_CLOSEOUT_LEVEL = 0.50 # 50%

# --- å¯¦ç›¤äº¤æ˜“ç›¸é—œ ---
LIVE_TRADING_POLL_INTERVAL = 5 # å¯¦ç›¤äº¤æ˜“æ™‚è¼ªè©¢å¸‚å ´æ•¸æ“šå’ŒåŸ·è¡Œæ±ºç­–çš„é–“éš”ï¼ˆç§’ï¼‰
OANDA_MAX_BATCH_CANDLES = 4800 # OANDA API å–®æ¬¡è«‹æ±‚æœ€å¤§è Ÿç‡­æ•¸
OANDA_REQUEST_INTERVAL = 0.1   # API è«‹æ±‚é–“éš” (ç§’)ï¼Œé¿å…éŽæ–¼é »ç¹
OANDA_API_TIMEOUT_SECONDS = 20 # <--- æ–°å¢žï¼šAPIè«‹æ±‚çš„é€šç”¨è¶…æ™‚æ™‚é–“ (ç§’)

# --- UI ç›¸é—œ ---
STREAMLIT_THEME = "dark" # Streamlit UIä¸»é¡Œ ("light" æˆ– "dark")
PLOT_REFRESH_INTERVAL_SECONDS = 5 # UIåœ–è¡¨åˆ·æ–°é–“éš”
MAX_TRADE_LOG_DISPLAY = 100     # UIä¸­é¡¯ç¤ºæœ€è¿‘äº¤æ˜“çš„æ¢æ•¸

# --- è²¨å¹£è½‰æ›ç›¸é—œ ---
# å‡è¨­åŸºç¤ŽåŒ¯çŽ‡å°ï¼Œç”¨æ–¼å°‡éžUSDè¨ˆåƒ¹çš„symbolç›ˆè™§è½‰æ¢ç‚ºUSDï¼Œå†è½‰æ›ç‚ºACCOUNT_CURRENCY
# é€™äº›æœƒåœ¨ LiveTrader ä¸­å˜—è©¦å¾žOANDAç²å–å¯¦æ™‚åŒ¯çŽ‡ï¼Œé€™è£¡åƒ…ä½œçµæ§‹ç¤ºä¾‹
# KEY_FOREX_PAIRS_FOR_CONVERSION = ["AUD/USD", "EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "USD/CHF"]

# --- ç’°å¢ƒè¨­ç½® (PyTorch) ---
# çµ±ä¸€çš„CUDAå…§å­˜åˆ†é…é…ç½®
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"

# é€™äº›åœ¨æ‚¨çš„èˆŠä»£ç¢¼ä¸­å·²æœ‰ï¼Œå¯ä»¥ä¿ç•™æˆ–æ ¹æ“šéœ€è¦èª¿æ•´
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # ä¸»è¦ç”¨æ–¼èª¿è©¦ï¼Œç”Ÿç”¢ç’°å¢ƒä¸­å¯è€ƒæ…®ç§»é™¤
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.set_float32_matmul_precision('high') # PyTorch 1.12+

# é©—è­‰APIé‡‘é‘°æ˜¯å¦å·²åŠ è¼‰
if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
    print(f"è­¦å‘Š: OANDA_API_KEY æˆ– OANDA_ACCOUNT_ID æœªåœ¨ .env æ–‡ä»¶ä¸­æ‰¾åˆ°æˆ–æœªæ­£ç¢ºåŠ è¼‰ã€‚è«‹æª¢æŸ¥ {dotenv_path}ã€‚")
    # åœ¨å¯¦éš›é‹è¡Œæ™‚ï¼Œé€™è£¡å¯èƒ½éœ€è¦æ›´å¼·çš„éŒ¯èª¤è™•ç†ï¼Œä¾‹å¦‚ç›´æŽ¥é€€å‡ºç¨‹åº
    # raise ValueError("OANDA API Key or Account ID not configured.") # æˆ–è€…
    # sys.exit("OANDA API Key or Account ID not configured.")


# --- è¼”åŠ©å‡½æ•¸ (å¯é¸ï¼Œæ”¾åœ¨é€™è£¡æ–¹ä¾¿å…¨å±€è¨ªå•) ---
def get_granularity_seconds(granularity_str: str) -> int:
    """æ ¹æ“šOANDAçš„ç²’åº¦å­—ç¬¦ä¸²è¿”å›žå°æ‡‰çš„ç§’æ•¸"""
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
    elif granularity_str == "Mo": # OANDAçš„æœˆæ˜¯ 'M'ï¼Œä½†æˆ‘å€‘é€™è£¡ç”¨ 'Mo' é¿å…èˆ‡åˆ†é˜è¡çª
        return 30 * 24 * 3600 # è¿‘ä¼¼å€¼
    raise ValueError(f"æœªçŸ¥çš„ç²’åº¦: {granularity_str}")

GRANULARITY_SECONDS = get_granularity_seconds(GRANULARITY)

# --- è¼¸å‡ºä¸€äº›é—œéµé…ç½®ä»¥ä¾›æª¢æŸ¥ (å¯é¸) ---
print(f"Configuration loaded. Base directory: {BASE_DIR}")
print(f"Using device: {DEVICE}")
print(f"Database path: {DATABASE_PATH}")
print(f"Default symbols: {DEFAULT_SYMBOLS}")
print(f"Granularity: {GRANULARITY} ({GRANULARITY_SECONDS} seconds)")

# --- åœ¨æ–‡ä»¶æœ«å°¾å°Žå…¥æ—¥èªŒè¨˜éŒ„å™¨ä»¥é¿å…å¾ªç’°å°Žå…¥ ---
try:
    from .logger_setup import logger
except ImportError:
    # å¦‚æžœç›¸å°å°Žå…¥å¤±æ•—ï¼Œå˜—è©¦çµ•å°å°Žå…¥
    try:
        from oanda_trading_bot.training_system.common.logger_setup import logger
    except ImportError:
        # æœ€çµ‚å¾Œå‚™ï¼šå‰µå»ºä¸€å€‹ç°¡å–®çš„æ—¥èªŒè¨˜éŒ„å™¨
        import logging
        logger = logging.getLogger("config_fallback")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
