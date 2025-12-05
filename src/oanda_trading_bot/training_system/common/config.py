"""
Central configuration for the training system.

This module normalises environment loading, resolves the account profile from
OANDA, and exposes shared constants so that downstream components operate with
consistent settings.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Final

import torch
from dotenv import load_dotenv

from oanda_trading_bot.common.account_profile import (
    AccountProfile,
    AccountProfileError,
    get_account_profile,
)


CONFIG_VERSION: Final[str] = "2025-10-01"

# ---------------------------------------------------------------------------
# Project paths and environment loading
# ---------------------------------------------------------------------------

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[5]
os.environ.setdefault("PROJECT_ROOT", str(BASE_DIR))

ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# ---------------------------------------------------------------------------
# Account profile and OANDA credentials
# ---------------------------------------------------------------------------

OANDA_API_KEY: Final[str | None] = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID: Final[str | None] = os.getenv("OANDA_ACCOUNT_ID")
OANDA_ENVIRONMENT: Final[str] = os.getenv("OANDA_ENVIRONMENT", "practice").strip().lower()

if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
    raise RuntimeError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be defined in the environment.")


def _resolve_account_profile() -> AccountProfile:
    try:
        return get_account_profile()
    except AccountProfileError as exc:
        raise RuntimeError("Failed to resolve OANDA account profile.") from exc


ACCOUNT_PROFILE: Final[AccountProfile] = _resolve_account_profile()
ACCOUNT_CURRENCY: Final[str] = ACCOUNT_PROFILE.currency

OANDA_BASE_URL: Final[str] = os.getenv(
    "OANDA_BASE_URL",
    "https://api-fxtrade.oanda.com/v3" if OANDA_ENVIRONMENT == "live" else "https://api-fxpractice.oanda.com/v3",
)

# ---------------------------------------------------------------------------
# File-system locations
# ---------------------------------------------------------------------------

DATABASE_DIR = BASE_DIR / "data" / "database"
DATABASE_NAME = "oanda_s5_data.db"
DATABASE_PATH = DATABASE_DIR / DATABASE_NAME

WEIGHTS_DIR = BASE_DIR / "weights"
LOGS_DIR = BASE_DIR / "logs"
MMAP_DATA_DIR = BASE_DIR / "data" / "mmap_s5_universal"

for directory in (DATABASE_DIR, WEIGHTS_DIR, LOGS_DIR, MMAP_DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------

DEFAULT_SYMBOLS: Final[list[str]] = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "XAU_USD"]
GRANULARITY: Final[str] = os.getenv("TRAINING_GRANULARITY", "S5")
MAX_SYMBOLS_ALLOWED: Final[int] = int(os.getenv("MAX_SYMBOLS_ALLOWED", "10"))
PRICE_COLUMNS: Final[list[str]] = [
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
    "volume",
]
PRICE_TYPES: Final[dict[str, list[str]]] = {
    "open": ["bid_open", "ask_open"],
    "high": ["bid_high", "ask_high"],
    "low": ["bid_low", "ask_low"],
    "close": ["bid_close", "ask_close"],
}

# OANDA API constants
OANDA_MAX_BATCH_CANDLES: Final[int] = int(os.getenv("OANDA_MAX_BATCH_CANDLES", "4800"))
OANDA_REQUEST_INTERVAL: Final[float] = float(os.getenv("OANDA_REQUEST_INTERVAL", "0.1"))

INITIAL_CAPITAL: Final[float] = float(os.getenv("INITIAL_CAPITAL", "100000.0"))
USE_AMP: Final[bool] = os.getenv("USE_AMP", "true").strip().lower() == "true"
DEVICE: Final[str] = "cuda" if torch.cuda.is_available() else "cpu"

# Risk parameters
OANDA_MARGIN_CLOSEOUT_LEVEL: Final[float] = float(os.getenv("MARGIN_CLOSEOUT_LEVEL", "0.5"))
TRADE_COMMISSION_PERCENTAGE: Final[float] = float(os.getenv("TRADE_COMMISSION_PERCENTAGE", "0.0"))
MAX_ACCOUNT_RISK_PERCENTAGE: Final[float] = float(os.getenv("MAX_ACCOUNT_RISK_PERCENTAGE", "0.02"))  # 2% max account risk per trade
ATR_STOP_LOSS_MULTIPLIER: Final[float] = float(os.getenv("ATR_STOP_LOSS_MULTIPLIER", "2.0"))
MAX_POSITION_SIZE_PERCENTAGE_OF_EQUITY: Final[float] = float(os.getenv("MAX_POSITION_SIZE_PERCENTAGE_OF_EQUITY", "0.1"))  # 10% max per position

# SAC hyperparameters (kept for compatibility)
SAC_LEARNING_RATE: Final[float] = float(os.getenv("SAC_LEARNING_RATE", "0.0003"))
SAC_BATCH_SIZE: Final[int] = int(os.getenv("SAC_BATCH_SIZE", "64"))
SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR: Final[int] = int(os.getenv("SAC_BUFFER_SIZE_PER_SYMBOL_FACTOR", "5000"))
SAC_LEARNING_STARTS_FACTOR: Final[int] = int(os.getenv("SAC_LEARNING_STARTS_FACTOR", "50"))
SAC_GAMMA: Final[float] = float(os.getenv("SAC_GAMMA", "0.99"))
SAC_ENT_COEF: Final[str] = os.getenv("SAC_ENT_COEF", "auto")
SAC_TRAIN_FREQ_STEPS: Final[int] = int(os.getenv("SAC_TRAIN_FREQ_STEPS", "1"))
SAC_GRADIENT_STEPS: Final[int] = int(os.getenv("SAC_GRADIENT_STEPS", "1"))
SAC_TAU: Final[float] = float(os.getenv("SAC_TAU", "0.005"))

# Training Loop Parameters
TIMESTEPS: Final[int] = int(os.getenv("TIMESTEPS", "100000"))
TRAINER_SAVE_FREQ_STEPS: Final[int] = int(os.getenv("TRAINER_SAVE_FREQ_STEPS", "10000"))
TRAINER_EVAL_FREQ_STEPS: Final[int] = int(os.getenv("TRAINER_EVAL_FREQ_STEPS", "10000"))

# Model configuration paths
ENHANCED_TRANSFORMER_CONFIG_PATH = BASE_DIR / "configs" / "enhanced_transformer_config.json"
QUANTUM_STRATEGY_CONFIG_PATH = BASE_DIR / "configs" / "quantum_strategy_config.json"
_model_config_primary = BASE_DIR / "configs" / "training" / "enhanced_model_config.json"
_model_config_legacy = BASE_DIR / "configs" / "enhanced_model_config.json"
ENHANCED_MODEL_CONFIG_PATH = _model_config_primary if _model_config_primary.exists() else _model_config_legacy

# Quantum Strategy Configuration
QUANTUM_STRATEGY_NUM_STRATEGIES: Final[int] = int(os.getenv("QUANTUM_STRATEGY_NUM_STRATEGIES", "8"))

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def get_granularity_seconds(granularity_str: str) -> int:
    head = granularity_str[0].upper()
    value = granularity_str[1:]
    if head == "S":
        return int(value)
    if head == "M":
        return int(value) * 60
    if head == "H":
        return int(value) * 3600
    if granularity_str == "D":
        return 24 * 3600
    if granularity_str == "W":
        return 7 * 24 * 3600
    if granularity_str == "MO":
        return 30 * 24 * 3600
    raise ValueError(f"Unsupported granularity: {granularity_str}")


GRANULARITY_SECONDS: Final[int] = get_granularity_seconds(GRANULARITY)


# ---------------------------------------------------------------------------
# Logging bootstrap
# ---------------------------------------------------------------------------

try:
    from .logger_setup import logger  # noqa: F401
except Exception:  # pragma: no cover - fallback to avoid circular failures
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TrainingSystemConfig")
    logger.warning("Falling back to basic logging configuration.")
