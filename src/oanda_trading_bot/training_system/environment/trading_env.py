# src/environment/trading_env.py
"""
通用多資產強化學習交易環境 - (V5.0 - 包含詳細的 step 方法)
"""
# ... (頂部的導入和後備導入邏輯與 V4.9 版本相同) ...
# <在此處粘貼您上一個版本 trading_env.py 中從文件頂部到 UniversalTradingEnvV4 類定義之前的全部內容>
# 我將重新提供頂部導入，確保所有內容都在
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, getcontext, ROUND_DOWN, ROUND_CEILING, ROUND_FLOOR
import sys
import os
import warnings
import traceback
import time
import logging
from pathlib import Path

# Add matplotlib import with backend setting for headless environments
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Correctly adjust sys.path when the script is run directly
# This ensures that imports like 'from oanda_trading_bot.training_system.common import ...' work.
if __name__ == "__main__": # Only run this when script is executed directly
    # Get the directory of the current script (e.g., Oanda_Trading_Bot/src/environment)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up two levels to get the project root (e.g., Oanda_Trading_Bot)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    # Add the project root to sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from collections import deque, defaultdict

getcontext().prec = 30
logger: logging.Logger = logging.getLogger("trading_env_module_init") # type: ignore
# ... (logger 初始化和 try-except import 塊與 V4.9 版本相同，這裡省略以節省篇幅) ...
_logger_initialized_by_common_env_v5 = False

# Flag to prevent duplicate import logging
_import_logged = False

try:
    from oanda_trading_bot.training_system.common.logger_setup import logger as common_configured_logger; logger = common_configured_logger; _logger_initialized_by_common_env_v5 = True
    if not _import_logged:
        logger.debug("trading_env.py (V5.0): Successfully imported logger from common.logger_setup.")
    from oanda_trading_bot.training_system.common.config import (TIMESTEPS as _TIMESTEPS, MAX_SYMBOLS_ALLOWED as _MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY as _ACCOUNT_CURRENCY, INITIAL_CAPITAL as _DEFAULT_INITIAL_CAPITAL, OANDA_MARGIN_CLOSEOUT_LEVEL as _OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE as _TRADE_COMMISSION_PERCENTAGE, OANDA_API_KEY as _OANDA_API_KEY, ATR_PERIOD as _ATR_PERIOD, STOP_LOSS_ATR_MULTIPLIER as _STOP_LOSS_ATR_MULTIPLIER, MAX_ACCOUNT_RISK_PERCENTAGE as _MAX_ACCOUNT_RISK_PERCENTAGE)
    _config_values_env_v5 = {"TIMESTEPS": _TIMESTEPS, "MAX_SYMBOLS_ALLOWED": _MAX_SYMBOLS_ALLOWED, "ACCOUNT_CURRENCY": _ACCOUNT_CURRENCY, "DEFAULT_INITIAL_CAPITAL": _DEFAULT_INITIAL_CAPITAL, "OANDA_MARGIN_CLOSEOUT_LEVEL": _OANDA_MARGIN_CLOSEOUT_LEVEL, "TRADE_COMMISSION_PERCENTAGE": _TRADE_COMMISSION_PERCENTAGE, "OANDA_API_KEY": _OANDA_API_KEY, "ATR_PERIOD": _ATR_PERIOD, "STOP_LOSS_ATR_MULTIPLIER": _STOP_LOSS_ATR_MULTIPLIER, "MAX_ACCOUNT_RISK_PERCENTAGE": _MAX_ACCOUNT_RISK_PERCENTAGE}
    if not _import_logged:
        logger.info("trading_env.py (V5.0): Successfully imported and stored common.config values.")
    from oanda_trading_bot.training_system.data_manager.mmap_dataset import UniversalMemoryMappedDataset; from oanda_trading_bot.training_system.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols; from oanda_trading_bot.common.instrument_info_manager import InstrumentDetails, InstrumentInfoManager;
    from oanda_trading_bot.training_system.data_manager.currency_route_planner import (
        compute_required_pairs_for_training, compute_conversion_rate_along_route,
    )
    if not _import_logged:
        logger.info("trading_env.py (V5.0): Successfully imported other dependencies.")
        _import_logged = True
except ImportError as e_initial_import_v5:
    logger_temp_v5 = logging.getLogger("trading_env_v5_fallback_initial"); logger_temp_v5.addHandler(logging.StreamHandler(sys.stdout)); logger_temp_v5.setLevel(logging.DEBUG); logger = logger_temp_v5
    logger.warning(f"trading_env.py (V5.0): Initial import failed: {e_initial_import_v5}. Assuming PYTHONPATH is set correctly or this is a critical issue.")
    try:
        # 假設 PYTHONPATH 已設定，這些導入應該能工作
        from oanda_trading_bot.training_system.common.logger_setup import logger as common_logger_retry_v5; logger = common_logger_retry_v5; _logger_initialized_by_common_env_v5 = True; logger.info("trading_env.py (V5.0): Successfully re-imported common_logger.")
        from oanda_trading_bot.training_system.common.config import (TIMESTEPS as _TIMESTEPS_R, MAX_SYMBOLS_ALLOWED as _MAX_SYMBOLS_ALLOWED_R, ACCOUNT_CURRENCY as _ACCOUNT_CURRENCY_R, INITIAL_CAPITAL as _DEFAULT_INITIAL_CAPITAL_R, OANDA_MARGIN_CLOSEOUT_LEVEL as _OANDA_MARGIN_CLOSEOUT_LEVEL_R, TRADE_COMMISSION_PERCENTAGE as _TRADE_COMMISSION_PERCENTAGE_R, OANDA_API_KEY as _OANDA_API_KEY_R, ATR_PERIOD as _ATR_PERIOD_R, STOP_LOSS_ATR_MULTIPLIER as _STOP_LOSS_ATR_MULTIPLIER_R, MAX_ACCOUNT_RISK_PERCENTAGE as _MAX_ACCOUNT_RISK_PERCENTAGE_R)
        _config_values_env_v5 = {"TIMESTEPS": _TIMESTEPS_R, "MAX_SYMBOLS_ALLOWED": _MAX_SYMBOLS_ALLOWED_R, "ACCOUNT_CURRENCY": _ACCOUNT_CURRENCY_R, "DEFAULT_INITIAL_CAPITAL": _DEFAULT_INITIAL_CAPITAL_R, "OANDA_MARGIN_CLOSEOUT_LEVEL": _OANDA_MARGIN_CLOSEOUT_LEVEL_R, "TRADE_COMMISSION_PERCENTAGE": _TRADE_COMMISSION_PERCENTAGE_R, "OANDA_API_KEY": _OANDA_API_KEY_R, "ATR_PERIOD": _ATR_PERIOD_R, "STOP_LOSS_ATR_MULTIPLIER": _STOP_LOSS_ATR_MULTIPLIER_R, "MAX_ACCOUNT_RISK_PERCENTAGE": _MAX_ACCOUNT_RISK_PERCENTAGE_R}
        logger.info("trading_env.py (V5.0): Successfully re-imported and stored common.config after path adjustment.")
        from oanda_trading_bot.training_system.data_manager.mmap_dataset import UniversalMemoryMappedDataset; from oanda_trading_bot.training_system.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols; from oanda_trading_bot.common.instrument_info_manager import InstrumentDetails, InstrumentInfoManager;
        from oanda_trading_bot.training_system.data_manager.currency_route_planner import (
            compute_required_pairs_for_training, compute_conversion_rate_along_route,
        )
        # Import enhanced reward calculator
        try:
            from oanda_trading_bot.training_system.environment.enhanced_reward_calculator import EnhancedRewardCalculator
            from oanda_trading_bot.training_system.environment.progressive_reward_calculator import ProgressiveRewardCalculator
            from oanda_trading_bot.training_system.environment.adaptive_reward_optimizer import AdaptiveRewardOptimizer
            from oanda_trading_bot.training_system.environment.advanced_reward_strategies import AdvancedRewardStrategies
            logger.info("trading_env.py (V5.0): Successfully imported all reward calculators")
        except ImportError as e_reward_calc:
            logger.warning(f"trading_env.py (V5.0): Failed to import reward calculators: {e_reward_calc}")
            EnhancedRewardCalculator = None
            ProgressiveRewardCalculator = None
            AdaptiveRewardOptimizer = None
            AdvancedRewardStrategies = None
        logger.info("trading_env.py (V5.0): Successfully re-imported other dependencies after path adjustment.")
    except ImportError as e_retry_critical_v5:
        logger.error(f"trading_env.py (V5.0): Critical import error after path adjustment: {e_retry_critical_v5}", exc_info=True); logger.warning("trading_env.py (V5.0): Using fallback values for config (critical error during import).")
        _config_values_env_v5 = {"TIMESTEPS": 128, "MAX_SYMBOLS_ALLOWED": 20, "ACCOUNT_CURRENCY": "AUD", "DEFAULT_INITIAL_CAPITAL": 100000.0, "OANDA_MARGIN_CLOSEOUT_LEVEL": Decimal('0.50'), "TRADE_COMMISSION_PERCENTAGE": Decimal('0.0001'), "OANDA_API_KEY": None, "ATR_PERIOD": 14, "STOP_LOSS_ATR_MULTIPLIER": Decimal('2.0'), "MAX_ACCOUNT_RISK_PERCENTAGE": Decimal('0.01')}
        for k_fallback, v_fallback in _config_values_env_v5.items(): globals()[k_fallback] = v_fallback
        if 'UniversalMemoryMappedDataset' not in globals(): UniversalMemoryMappedDataset = type('DummyDataset', (), {'__init__': lambda self, **kwargs: setattr(self, 'symbols', []), '__len__': lambda self: 0, 'timesteps_history': 128, 'num_features_per_symbol': 9, 'aligned_timestamps': pd.Series()}) # type: ignore
        if 'InstrumentDetails' not in globals(): InstrumentDetails = type('DummyInstrumentDetails', (), {}) # type: ignore
        if 'InstrumentInfoManager' not in globals(): InstrumentInfoManager = type('DummyInfoManager', (), {'get_details': lambda self, sym: InstrumentDetails(symbol=sym, quote_currency="USD", base_currency=sym, margin_rate=0.05, minimum_trade_size=1, trade_units_precision=0, pip_location=-4, type="CURRENCY", display_name=sym,is_forex=True)}) # type: ignore
        if 'format_datetime_for_oanda' not in globals():
            def format_datetime_for_oanda(dt):
                return dt.isoformat()
        if 'manage_data_download_for_symbols' not in globals():
            def manage_data_download_for_symbols(*args, **kwargs):
                logger.error("Downloader not available in fallback.")
        logger.info("trading_env.py (V5.0): Fallback definitions applied.")

TIMESTEPS = _config_values_env_v5.get("TIMESTEPS", 128); MAX_SYMBOLS_ALLOWED = _config_values_env_v5.get("MAX_SYMBOLS_ALLOWED", 20); ACCOUNT_CURRENCY = _config_values_env_v5.get("ACCOUNT_CURRENCY", "AUD"); DEFAULT_INITIAL_CAPITAL = _config_values_env_v5.get("DEFAULT_INITIAL_CAPITAL", 100000.0); OANDA_MARGIN_CLOSEOUT_LEVEL = _config_values_env_v5.get("OANDA_MARGIN_CLOSEOUT_LEVEL", Decimal('0.50')); TRADE_COMMISSION_PERCENTAGE = _config_values_env_v5.get("TRADE_COMMISSION_PERCENTAGE", Decimal('0.0001')); OANDA_API_KEY = _config_values_env_v5.get("OANDA_API_KEY", None); ATR_PERIOD = _config_values_env_v5.get("ATR_PERIOD", 14); STOP_LOSS_ATR_MULTIPLIER = _config_values_env_v5.get("STOP_LOSS_ATR_MULTIPLIER", Decimal('2.0')); MAX_ACCOUNT_RISK_PERCENTAGE = _config_values_env_v5.get("MAX_ACCOUNT_RISK_PERCENTAGE", Decimal('0.01')); MAX_POSITION_SIZE_PERCENTAGE_OF_EQUITY = _config_values_env_v5.get("MAX_POSITION_SIZE_PERCENTAGE_OF_EQUITY", Decimal('0.10'))


class UniversalTradingEnvV4(gym.Env): # 保持類名為V4，但內部是V5邏輯
    metadata = {'render_modes': ['human', 'array'], 'render_fps': 10}
    
    def __init__(self, dataset: UniversalMemoryMappedDataset, instrument_info_manager: InstrumentInfoManager, active_symbols_for_episode: List[str], # type: ignore
                 initial_capital: float = float(DEFAULT_INITIAL_CAPITAL), max_episode_steps: Optional[int] = None,
                 commission_percentage_override: Optional[float] = None, reward_config: Optional[Dict[str, Union[float, Decimal]]] = None,
                 max_account_risk_per_trade: float = float(MAX_ACCOUNT_RISK_PERCENTAGE),
                 stop_loss_atr_multiplier: float = float(STOP_LOSS_ATR_MULTIPLIER),
                 atr_period: int = ATR_PERIOD, render_mode: Optional[str] = None,
                 shared_data_manager=None, training_step_offset: int = 0,
                 max_position_size_percentage: float = float(MAX_POSITION_SIZE_PERCENTAGE_OF_EQUITY)):
        super().__init__()
        self.dataset = dataset

        # NEW: Add a check for dataset compatibility to prevent ATR calculation errors
        required_cols = ['bid_close', 'ask_close', 'bid_high', 'bid_low', 'ask_high', 'ask_low']
        if not hasattr(self.dataset, 'raw_price_columns_ordered') or not all(c in self.dataset.raw_price_columns_ordered for c in required_cols):
            error_msg = (
                "Dataset is missing 'raw_price_columns_ordered' metadata or required columns for ATR calculation "
                f"(needs {required_cols}). Please delete the existing mmap data folder and rebuild the dataset."
            )
            logger.critical(error_msg)
            raise ValueError(error_msg)
            
        self.instrument_info_manager = instrument_info_manager
        self.initial_capital = Decimal(str(initial_capital))

        # Create a mapping from symbol string to a unique integer ID based on the master list from InstrumentInfoManager
        # This is crucial for the symbol embedding layer in the feature extractor.
        if not hasattr(self.instrument_info_manager, 'all_symbols') or not self.instrument_info_manager.all_symbols:
             # Attempt to get it via the method if the property doesn't exist for some reason.
             self.universe_symbols = self.instrument_info_manager.get_all_available_symbols()
             if not self.universe_symbols:
                raise ValueError("InstrumentInfoManager does not have 'all_symbols' and get_all_available_symbols() returned empty. Cannot create symbol-ID mapping.")
        else:
            self.universe_symbols = self.instrument_info_manager.all_symbols

        # Define number of symbols and padding ID consistently with the feature extractor
        self.num_universe_symbols = len(self.universe_symbols)
        if self.num_universe_symbols == 0:
            logger.critical("InstrumentInfoManager returned an empty list of universe symbols. Cannot proceed.")
            raise ValueError("InstrumentInfoManager returned an empty list of universe symbols.")

        self.symbol_to_global_id_map: Dict[str, int] = {symbol: i for i, symbol in enumerate(self.universe_symbols)}
        
        # The padding ID is the index right after the last valid symbol ID (0 to num_universe_symbols-1).
        # This matches the feature extractor's nn.Embedding(num_symbols + 1, padding_idx=num_symbols).
        self.padding_symbol_id = self.num_universe_symbols
        logger.info(f"Symbol mapping created. Universe size: {self.num_universe_symbols}. Padding ID set to: {self.padding_symbol_id}")

        # 初始化統一的貨幣轉換管理器
        from oanda_trading_bot.training_system.data_manager.currency_manager import CurrencyDependencyManager
        self.currency_manager = CurrencyDependencyManager(ACCOUNT_CURRENCY, apply_oanda_markup=True)
        # Precompute conversion routes for active symbols' quote currencies to account currency
        self._conversion_routes_map = {}
        try:
            required_symbols, routes = compute_required_pairs_for_training(
                trading_symbols=list(self.active_symbols_for_episode) if hasattr(self, 'active_symbols_for_episode') else [],
                account_currency=ACCOUNT_CURRENCY,
                instrument_info_manager=self.instrument_info_manager,
            )
            for sym, route in routes.items():
                self._conversion_routes_map[str(route.from_ccy).upper()] = route
        except Exception as _e_routes_init:
            logger.debug(f"Route precompute skipped: {_e_routes_init}")
        if commission_percentage_override is not None:
            self.commission_percentage = Decimal(str(commission_percentage_override))
        else:
            self.commission_percentage = Decimal(str(TRADE_COMMISSION_PERCENTAGE))
        self.render_mode = render_mode
        self.max_account_risk_per_trade = Decimal(str(max_account_risk_per_trade))
        self.stop_loss_atr_multiplier = Decimal(str(stop_loss_atr_multiplier))
        self.atr_period = atr_period
        self.max_position_size_percentage = Decimal(str(max_position_size_percentage))
        
        # Shared data manager integration for real-time monitoring
        self.shared_data_manager = shared_data_manager
        self.training_step_offset = training_step_offset  # Global training step offset for current episode
        self.instrument_details_map: Dict[str, InstrumentDetails] = {} # type: ignore
        for sym in self.dataset.symbols:
            details = self.instrument_info_manager.get_details(sym)
            if not details: msg = f"無法獲取數據集中交易對象 {sym} 的詳細信息"; logger.error(msg); raise ValueError(msg)
            self.instrument_details_map[sym] = details
        self.active_symbols_for_episode = sorted(list(set(active_symbols_for_episode)))
        if not all(s in self.instrument_details_map for s in self.active_symbols_for_episode):
            missing = [s for s in self.active_symbols_for_episode if s not in self.instrument_details_map]
            msg = f"部分活躍交易對象缺少詳細信息 (可能未從manager獲取到): {missing}"; logger.error(msg); raise ValueError(msg)
        self.num_env_slots = MAX_SYMBOLS_ALLOWED
        self.symbol_to_slot_map: Dict[str, int] = {}; self.slot_to_symbol_map: Dict[int, Optional[str]] = {i: None for i in range(self.num_env_slots)}
        self.current_episode_tradable_slot_indices: List[int] = []
        for i, sym in enumerate(self.active_symbols_for_episode):
            if i < self.num_env_slots: self.symbol_to_slot_map[sym] = i; self.slot_to_symbol_map[i] = sym; self.current_episode_tradable_slot_indices.append(i)
            else: logger.warning(f"本次episode的活躍交易對象 {sym} 超過最大槽位數，將被忽略。")
        self.num_tradable_symbols_this_episode = len(self.current_episode_tradable_slot_indices)
        logger.info(f"Environment initialized: {self.num_tradable_symbols_this_episode} trading symbols mapped to {self.num_env_slots} slots.")
        self.current_step_in_dataset = 0; self.episode_step_count = 0
        if max_episode_steps is None: self.max_episode_steps = len(self.dataset)
        else: self.max_episode_steps = min(max_episode_steps, len(self.dataset))
        self.cash: Decimal = Decimal('0.0')
        self.current_positions_units: np.ndarray = np.array([Decimal('0.0')] * self.num_env_slots, dtype=object)
        self.avg_entry_prices_qc: np.ndarray = np.array([Decimal('0.0')] * self.num_env_slots, dtype=object)
        self.unrealized_pnl_ac: np.ndarray = np.array([Decimal('0.0')] * self.num_env_slots, dtype=object)
        self.margin_used_per_position_ac: np.ndarray = np.array([Decimal('0.0')] * self.num_env_slots, dtype=object)
        self.atr_values_qc: np.ndarray = np.array([Decimal('0.0')] * self.num_env_slots, dtype=object)
        self.stop_loss_prices_qc: np.ndarray = np.array([Decimal('0.0')] * self.num_env_slots, dtype=object)
        self.last_trade_step_per_slot: np.ndarray = np.full(self.num_env_slots, -1, dtype=np.int32)
        self.position_entry_step_per_slot: np.ndarray = np.full(self.num_env_slots, -1, dtype=np.int32)
        self.total_margin_used_ac: Decimal = Decimal('0.0'); self.portfolio_value_ac: Decimal = Decimal('0.0'); self.equity_ac: Decimal = Decimal('0.0')
        self.portfolio_value_history: List[float] = []; self.reward_history: List[float] = []; self.trade_log: List[Dict[str, Any]] = []
        
        # 風險調整後收益計算所需的歷史數據
        self.returns_history: List[Decimal] = []  # 收益序列，用於計算標準差
        self.returns_window_size: int = 20  # 滾動窗口大小
        self.atr_penalty_threshold: Decimal = Decimal('0.02')  # ATR懲罰閾值（2%）        # 獎勵配置        # 獎勵配置
        default_reward_config_decimal = {"portfolio_log_return_factor": Decimal('1.0'), "risk_adjusted_return_factor": Decimal('0.5'), "max_drawdown_penalty_factor": Decimal('2.0'), "commission_penalty_factor": Decimal('1.0'), "margin_call_penalty": Decimal('-100.0'), "profit_target_bonus": Decimal('0.1'), "hold_penalty_factor": Decimal('0.001')}
        if reward_config:
            for key, value in reward_config.items():
                if key in default_reward_config_decimal: default_reward_config_decimal[key] = Decimal(str(value))
        self.reward_config = default_reward_config_decimal
          # Initialize Reward Calculators
        self.enhanced_reward_calculator = None
        self.progressive_reward_calculator = None
        self.use_enhanced_rewards = False
        self.use_progressive_rewards = False
        
        # Check reward type preference from config
        reward_type = reward_config.get('reward_type', 'progressive') if reward_config else 'progressive'
        
        # Try to initialize Progressive Reward Calculator first (new default)
        try:
            from .progressive_reward_calculator import ProgressiveRewardCalculator
            
            progressive_config = {}
            if reward_config and 'progressive_config' in reward_config:
                progressive_config = reward_config['progressive_config']
            
            self.progressive_reward_calculator = ProgressiveRewardCalculator(
                initial_capital=self.initial_capital,
                config=progressive_config
            )
            self.use_progressive_rewards = True
            logger.info("ProgressiveRewardCalculator successfully initialized - using 3-stage training approach")
            
        except Exception as e:
            logger.warning(f"Failed to initialize ProgressiveRewardCalculator: {e}. Trying EnhancedRewardCalculator...")
            
            # Fallback to Enhanced Reward Calculator
            if 'EnhancedRewardCalculator' in globals() and EnhancedRewardCalculator is not None:
                try:
                    # Enhanced reward configuration with improved parameters
                    enhanced_config = {
                        "portfolio_log_return_factor": Decimal('0.8'),
                        "risk_adjusted_return_factor": Decimal('1.2'),  # Increased from 0.5
                        "max_drawdown_penalty_factor": Decimal('1.5'),  # Decreased from 2.0
                        "commission_penalty_factor": Decimal('0.8'),    # Decreased from 1.0
                        "margin_call_penalty": Decimal('-50.0'),         # Less harsh than -100
                        "profit_target_bonus": Decimal('0.3'),          # Increased from 0.1
                        "hold_penalty_factor": Decimal('0.0005'),       # Reduced from 0.001
                        "win_rate_incentive_factor": Decimal('1.0'),    # New parameter
                        "trend_following_bonus": Decimal('0.5'),        # New parameter
                        "quick_stop_loss_bonus": Decimal('0.1'),        # New parameter
                        "compound_holding_factor": Decimal('0.2'),      # New parameter
                    }
                    
                    # Override with any user-provided config
                    if reward_config:
                        for key, value in reward_config.items():
                            if key in enhanced_config:
                                enhanced_config[key] = Decimal(str(value))
                            elif key == "use_enhanced_rewards":
                                self.use_enhanced_rewards = bool(value)
                    
                    self.enhanced_reward_calculator = EnhancedRewardCalculator(
                        initial_capital=self.initial_capital,
                        config=enhanced_config
                    )
                    self.use_enhanced_rewards = True
                    logger.info("EnhancedRewardCalculator successfully initialized as fallback")
                except Exception as e2:
                    logger.warning(f"Failed to initialize EnhancedRewardCalculator: {e2}. Using standard rewards.")
                    self.enhanced_reward_calculator = None
                    self.use_enhanced_rewards = False
            else:
                logger.info("No advanced reward calculators available. Using standard reward calculation.")
        
        self.peak_portfolio_value_episode: Decimal = self.initial_capital; self.max_drawdown_episode: Decimal = Decimal('0.0')

        # NEW observation space for EnhancedTransformerFeatureExtractor
        # It expects 'market_features' (last step), 'context_features' (state info), and 'symbol_id'
        self.num_context_features = 5 # pos_ratio, pnl_ratio, time_since_trade, volatility, margin_level
        
        # The number of symbols for the embedding layer is the size of our symbol universe.
        num_embedding_symbols = self.num_universe_symbols

        obs_spaces = {
            # Last-step snapshot per symbol (kept for UI/backward compatibility)
            "market_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_env_slots, self.dataset.num_features_per_symbol),
                dtype=np.float32
            ),
            # Rolling sequence per symbol for Transformer/strategies
            "features_from_dataset": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_env_slots, self.dataset.timesteps_history, self.dataset.num_features_per_symbol),
                dtype=np.float32
            ),
            # Lightweight per-symbol context (positions, pnl, etc.)
            "context_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_env_slots, self.num_context_features),
                dtype=np.float32
            ),
            # Symbol token ids for embeddings and deriving padding
            "symbol_id": spaces.Box(
                low=0,
                high=num_embedding_symbols,  # The highest valid ID is the padding_symbol_id
                shape=(self.num_env_slots,),
                dtype=np.int32
            ),
            # Padding mask to indicate active (1) vs dummy (0) slots
            "padding_mask": spaces.MultiBinary(self.num_env_slots)
        }
        self.observation_space = spaces.Dict(obs_spaces)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_env_slots,), dtype=np.float32)
        if self.render_mode == 'human': self._init_render_figure()
        logger.info(f"UniversalTradingEnvV4 (整合量子策略層) 初始化完成。")

    def _update_portfolio_and_equity_value(self, all_prices_map: Dict[str, Tuple[Decimal, Decimal]]):
        """
        Updates unrealized PnL for all positions, total portfolio value, and equity.
        This is a critical method called at each step.
        """
        total_unrealized_pnl_ac = Decimal('0.0')
        
        for slot_idx in range(self.num_env_slots):
            symbol = self.slot_to_symbol_map.get(slot_idx)
            units = self.current_positions_units[slot_idx]
            
            if not symbol or abs(units) < Decimal('1e-9'):
                self.unrealized_pnl_ac[slot_idx] = Decimal('0.0')
                continue
                
            current_bid_qc, current_ask_qc = all_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
            if current_bid_qc <= Decimal('0') or current_ask_qc <= Decimal('0'):
                # Cannot calculate PnL without a valid price, keep the last known PnL.
                total_unrealized_pnl_ac += self.unrealized_pnl_ac[slot_idx]
                continue

            entry_price_qc = self.avg_entry_prices_qc[slot_idx]
            
            # Determine the current price to use for PnL calculation (the price to close the position)
            current_price_for_pnl = current_bid_qc if units > 0 else current_ask_qc
                
            pnl_per_unit_qc = current_price_for_pnl - entry_price_qc
            unrealized_pnl_qc = units * pnl_per_unit_qc
            
            exchange_rate = self._get_exchange_rate_to_account_currency(self.instrument_details_map[symbol].quote_currency, all_prices_map)
            unrealized_pnl_ac = unrealized_pnl_qc * exchange_rate
            
            self.unrealized_pnl_ac[slot_idx] = unrealized_pnl_ac
            total_unrealized_pnl_ac += unrealized_pnl_ac

        self.equity_ac = self.cash + total_unrealized_pnl_ac
        self.portfolio_value_ac = self.equity_ac # For a margin account, portfolio value is equity
        
        self.total_margin_used_ac = sum(self.margin_used_per_position_ac)
        
        self.peak_portfolio_value_episode = max(self.peak_portfolio_value_episode, self.portfolio_value_ac)
        if self.peak_portfolio_value_episode > Decimal('0'):
            drawdown = (self.peak_portfolio_value_episode - self.portfolio_value_ac) / self.peak_portfolio_value_episode
            self.max_drawdown_episode = max(self.max_drawdown_episode, drawdown)
        
        self.portfolio_value_history.append(float(self.portfolio_value_ac))

    def _update_atr_values(self, all_prices_map: Dict[str, Tuple[Decimal, Decimal]]):
        """
        Calculates and updates the ATR values for all active symbols.
        """
        try:
            # Dynamically get price column indices from dataset metadata
            try:
                price_cols = self.dataset.raw_price_columns_ordered
                idx_bid_close = price_cols.index('bid_close')
                idx_ask_close = price_cols.index('ask_close')
                idx_bid_high = price_cols.index('bid_high')
                idx_ask_high = price_cols.index('ask_high')
                idx_bid_low = price_cols.index('bid_low')
                idx_ask_low = price_cols.index('ask_low')
            except (ValueError, AttributeError) as e:
                logger.error(f"Could not find required price columns in dataset metadata. Error: {e}", exc_info=True)
                # Set all ATRs to zero as a fallback
                for slot_idx in self.current_episode_tradable_slot_indices:
                    self.atr_values_qc[slot_idx] = Decimal('0.0')
                return

            safe_step_index = min(self.current_step_in_dataset, len(self.dataset) - 1)
            if safe_step_index < self.dataset.timesteps_history -1: # Need enough history for ATR
                return

            # We need historical data, so we get the full sample from the dataset
            dataset_sample = self.dataset[safe_step_index - (self.dataset.timesteps_history - 1)]
            historical_raw_prices_np = dataset_sample["raw_prices"].numpy().astype(np.float64)
            
            dataset_symbol_to_idx_map = {symbol: i for i, symbol in enumerate(self.dataset.symbols)}

            for slot_idx in self.current_episode_tradable_slot_indices:
                symbol = self.slot_to_symbol_map.get(slot_idx)
                if not symbol or symbol not in dataset_symbol_to_idx_map:
                    continue

                dataset_idx = dataset_symbol_to_idx_map[symbol]
                
                # Use dynamic indices to get the correct columns
                high_prices = (historical_raw_prices_np[dataset_idx, :, idx_bid_high] + historical_raw_prices_np[dataset_idx, :, idx_ask_high]) / 2
                low_prices = (historical_raw_prices_np[dataset_idx, :, idx_bid_low] + historical_raw_prices_np[dataset_idx, :, idx_ask_low]) / 2
                close_prices = (historical_raw_prices_np[dataset_idx, :, idx_bid_close] + historical_raw_prices_np[dataset_idx, :, idx_ask_close]) / 2

                if len(close_prices) < self.atr_period:
                    self.atr_values_qc[slot_idx] = Decimal('0.0')
                    continue

                df = pd.DataFrame({'high': high_prices, 'low': low_prices, 'close': close_prices})

                high_low = df['high'] - df['low']
                high_prev_close = np.abs(df['high'] - df['close'].shift(1))
                low_prev_close = np.abs(df['low'] - df['close'].shift(1))
                
                tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1, ignore_index=True).max(axis=1, skipna=True)
                atr = tr.rolling(window=self.atr_period, min_periods=self.atr_period).mean().iloc[-1]

                self.atr_values_qc[slot_idx] = Decimal(str(atr)) if pd.notna(atr) and np.isfinite(atr) else Decimal('0.0')

        except Exception as e:
            logger.error(f"Error during ATR calculation: {e}", exc_info=True)
            for slot_idx in self.current_episode_tradable_slot_indices:
                self.atr_values_qc[slot_idx] = Decimal('0.0')

    def _get_exchange_rate_to_account_currency(self, currency: str, all_prices_map: Dict[str, Tuple[Decimal, Decimal]]) -> Decimal:
        """
        Gets the exchange rate to convert from a given currency to the account currency.
        """
        try:
            ccy = str(currency).upper()
            route = getattr(self, '_conversion_routes_map', {}).get(ccy)
            if route is not None:
                rate = compute_conversion_rate_along_route(route, all_prices_map, apply_oanda_markup=True)
                if rate is not None and rate > 0:
                    return rate
        except Exception as _e_conv:
            logger.debug(f"Route-based conversion failed, falling back: {_e_conv}")
        return self.currency_manager.convert_to_account_currency(currency, all_prices_map)

    def _round_trade_units(self, units: Decimal, precision: int) -> Decimal:
        """Rounds trade units to the instrument's specified precision."""
        return units.quantize(Decimal('1e-' + str(precision)), rounding=ROUND_HALF_UP)

    def _update_margin_for_position(self, slot_idx: int, all_prices_map: Dict[str, Tuple[Decimal, Decimal]]):
        """
        Updates the margin used for a single position.
        """
        symbol = self.slot_to_symbol_map.get(slot_idx)
        if not symbol:
            self.margin_used_per_position_ac[slot_idx] = Decimal('0.0')
            return

        units = self.current_positions_units[slot_idx]
        if abs(units) < Decimal('1e-9'):
            self.margin_used_per_position_ac[slot_idx] = Decimal('0.0')
            return
            
        details = self.instrument_details_map[symbol]
        _, ask_price_qc = all_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
        
        if ask_price_qc <= 0:
            return

        position_value_qc = abs(units) * ask_price_qc
        
        # Corrected currency conversion logic
        # To convert a value from quote currency to base currency, we need the base/quote rate.
        # get_specific_rate(base, quote) returns quote/base. We need get_specific_rate(quote, base) to get base/quote.
        exchange_rate_quote_to_base = self.currency_manager.get_specific_rate(
            details.quote_currency, details.base_currency, all_prices_map, is_for_conversion=True
        )
        
        if exchange_rate_quote_to_base is None or exchange_rate_quote_to_base <= 0:
            logger.warning(f"Could not get exchange rate for {details.quote_currency}/{details.base_currency} for margin calculation. Setting margin to 0.")
            self.margin_used_per_position_ac[slot_idx] = Decimal('0.0')
            return

        position_value_base = position_value_qc * exchange_rate_quote_to_base
        margin_required_base = position_value_base * Decimal(str(details.margin_rate))
        
        # This converts from base currency to account currency.
        exchange_rate_base_to_ac = self.currency_manager.convert_to_account_currency(
            details.base_currency, all_prices_map
        )
        margin_required_ac = margin_required_base * exchange_rate_base_to_ac
        self.margin_used_per_position_ac[slot_idx] = margin_required_ac

    def _execute_trade(self, slot_idx: int, units_to_trade: Decimal, trade_price_qc: Decimal, timestamp: pd.Timestamp, all_prices_map: Dict[str, Tuple[Decimal, Decimal]]) -> Tuple[Decimal, Decimal]:
        """
        Core logic to execute a single trade. Returns (realized_pnl_ac, commission_ac).
        """
        symbol = self.slot_to_symbol_map[slot_idx]
        if not symbol: return Decimal('0.0'), Decimal('0.0')

        details = self.instrument_details_map[symbol]
        
        trade_value_qc = abs(units_to_trade) * trade_price_qc
        # Commissions are debits; apply debit conversion (mid*(1+markup)) per OANDA convention
        commission_rate_ac = self.currency_manager.convert_to_account_currency(
            details.quote_currency, all_prices_map, is_credit=False
        )
        trade_value_ac = trade_value_qc * commission_rate_ac
        commission = trade_value_ac * self.commission_percentage
        
        self.cash -= commission

        current_units = self.current_positions_units[slot_idx]
        new_units = current_units + units_to_trade
        avg_entry_price = self.avg_entry_prices_qc[slot_idx]
        realized_pnl_qc = Decimal('0.0')
        hold_duration = 0

        # Determine if this trade closes or reduces an existing position
        is_closing_trade = (abs(new_units) < abs(current_units)) or (np.sign(new_units) != np.sign(current_units) and abs(current_units) > 0)

        if is_closing_trade:
            entry_step = self.position_entry_step_per_slot[slot_idx]
            if entry_step != -1:
                hold_duration = self.episode_step_count - entry_step
            
            if np.sign(new_units) != np.sign(current_units): # Position is flipped
                units_closed = -current_units
            else: # Position is reduced
                units_closed = -units_to_trade
            
            realized_pnl_qc = units_closed * (trade_price_qc - avg_entry_price)

        # Update position entry step
        if abs(current_units) < Decimal('1e-9') and abs(new_units) > Decimal('1e-9'):
            # This is a new position opening
            self.position_entry_step_per_slot[slot_idx] = self.episode_step_count
        elif abs(new_units) < Decimal('1e-9'):
            # Position is fully closed
            self.position_entry_step_per_slot[slot_idx] = -1
        elif np.sign(new_units) != np.sign(current_units):
            # Position is flipped, so it's a new entry
            self.position_entry_step_per_slot[slot_idx] = self.episode_step_count
        
        if abs(new_units) > Decimal('1e-9'):
            if abs(current_units) < Decimal('1e-9') or np.sign(new_units) != np.sign(current_units):
                self.avg_entry_prices_qc[slot_idx] = trade_price_qc
            else:
                # Ensure no division by zero if new_units is somehow zero
                if abs(new_units) > Decimal('1e-9'):
                    new_avg_price = ((current_units * avg_entry_price) + (units_to_trade * trade_price_qc)) / new_units
                    self.avg_entry_prices_qc[slot_idx] = new_avg_price
                else:
                    self.avg_entry_prices_qc[slot_idx] = Decimal('0.0')
        else:
            self.avg_entry_prices_qc[slot_idx] = Decimal('0.0')

        self.current_positions_units[slot_idx] = new_units
        
        # PnL conversion uses credit for gains and debit for losses
        pnl_is_credit = realized_pnl_qc >= 0
        pnl_rate_ac = self.currency_manager.convert_to_account_currency(
            details.quote_currency, all_prices_map, is_credit=pnl_is_credit
        )
        realized_pnl_ac = realized_pnl_qc * pnl_rate_ac
        self.cash += realized_pnl_ac
        
        self.trade_log.append({
            "step": self.episode_step_count, "timestamp": timestamp, "symbol": symbol,
            "action": "buy" if units_to_trade > 0 else "sell", "units": float(units_to_trade),
            "price_qc": float(trade_price_qc), "realized_pnl_ac": float(realized_pnl_ac),
            "commission_ac": float(commission), "cash_ac": float(self.cash),
            "portfolio_value_ac": float(self.portfolio_value_ac),
            "hold_duration": hold_duration, # Add hold duration to the log
            "trade_type": "long" if current_units > 0 else "short" if current_units < 0 else "flat" # Add trade type for reward calc
        })
        
        # NEW: Send trade record to the shared data manager for UI display
        if self.shared_data_manager:
            try:
                # Determine detailed action string for UI display
                action_str = ""
                TOLERANCE = Decimal('1e-9')
                is_current_flat = abs(current_units) < TOLERANCE

                if is_current_flat:
                    action_str = "Open - Long" if units_to_trade > 0 else "Open - Short"
                else:  # Position exists
                    is_long_position = current_units > 0
                    
                    # Case 1: Adding to existing position
                    if np.sign(units_to_trade) == np.sign(current_units):
                        action_str = "Add - Long" if is_long_position else "Add - Short"
                    # Case 2: Reducing, closing, or flipping position
                    else:
                        is_full_close = abs(abs(units_to_trade) - abs(current_units)) < TOLERANCE
                        is_flip = abs(units_to_trade) > abs(current_units)

                        if is_flip or is_full_close:
                            # For both flips and full closes, the action is to close the original position
                            action_str = "Close - Long" if is_long_position else "Close - Short"
                        else:  # is_reduce
                            action_str = "Reduce - Long" if is_long_position else "Reduce - Short"

                self.shared_data_manager.add_trade_record(
                    symbol=symbol,
                    action=action_str,
                    price=float(trade_price_qc),
                    quantity=float(abs(units_to_trade)),
                    profit_loss=float(realized_pnl_ac),
                    training_step=self.training_step_offset + self.episode_step_count,
                    timestamp=timestamp.to_pydatetime()
                )
            except Exception as e:
                logger.error(f"Failed to add trade record to shared manager: {e}", exc_info=True)

        self.last_trade_step_per_slot[slot_idx] = self.episode_step_count
        self.position_entry_step_per_slot[slot_idx] = self.episode_step_count
        self._update_margin_for_position(slot_idx, all_prices_map)
        self._set_stop_loss_price(slot_idx) # Set SL after trade
        
        return realized_pnl_ac, commission

    def _execute_agent_actions(self, action: np.ndarray, all_prices_map: Dict[str, Tuple[Decimal, Decimal]], current_timestamp: pd.Timestamp) -> Decimal:
        """
        Interprets the agent's action vector and executes trades.
        """
        total_commission = Decimal('0.0')
        
        for slot_idx, desired_position_ratio in enumerate(action):
            symbol = self.slot_to_symbol_map.get(slot_idx)
            if not symbol or slot_idx not in self.current_episode_tradable_slot_indices:
                continue

            details = self.instrument_details_map[symbol]
            current_bid_qc, current_ask_qc = all_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
            if current_bid_qc <= 0: continue
                 
            mid_price_qc = (current_bid_qc + current_ask_qc) / Decimal('2')
             
            
            # 1) Value cap by max_position_size_percentage (AC -> QC -> units)
            max_pos_value_ac = self.equity_ac * self.max_position_size_percentage
            exchange_rate_to_ac = self._get_exchange_rate_to_account_currency(details.quote_currency, all_prices_map)
            if exchange_rate_to_ac <= 0:
                continue
            max_pos_value_qc = max_pos_value_ac / exchange_rate_to_ac
            target_value_qc = max_pos_value_qc * Decimal(str(desired_position_ratio))
            value_cap_units = (target_value_qc / mid_price_qc) if mid_price_qc > 0 else Decimal('0')

            # 2) Risk-based cap using ATR stop distance (per-trade risk sizing)
            risk_cap_units = None
            atr_qc = self.atr_values_qc[slot_idx]
            if atr_qc > 0 and self.stop_loss_atr_multiplier > 0:
                sl_distance_qc = atr_qc * self.stop_loss_atr_multiplier
                # Risk per unit in account currency = stop distance (QC) * quote->AC rate
                risk_per_unit_ac = sl_distance_qc * exchange_rate_to_ac
                if risk_per_unit_ac > 0:
                    risk_amount_ac = self.equity_ac * self.max_account_risk_per_trade
                    risk_cap_units = risk_amount_ac / risk_per_unit_ac

            # 3) Combine caps: take the minimum absolute units while preserving sign of desired ratio
            desired_sign = Decimal('1.0') if Decimal(str(desired_position_ratio)) >= 0 else Decimal('-1.0')
            abs_units_value = abs(value_cap_units)
            if risk_cap_units is not None and risk_cap_units > 0:
                abs_units_final = min(abs_units_value, risk_cap_units)
            else:
                abs_units_final = abs_units_value
            target_units = desired_sign * abs_units_final
            target_units = self._round_trade_units(target_units, details.trade_units_precision)

            current_units = self.current_positions_units[slot_idx]
            units_to_trade = target_units - current_units
            
            if abs(units_to_trade) > Decimal('1e-9'):
                trade_price_qc = current_ask_qc if units_to_trade > 0 else current_bid_qc
                _, commission = self._execute_trade(slot_idx, units_to_trade, trade_price_qc, current_timestamp, all_prices_map)
                total_commission += commission

        return total_commission

    def _get_positions_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Gathers detailed data for all current positions, required for advanced reward calculation.
        Returns a dictionary mapping symbol to its position data.
        """
        positions_data: Dict[str, Dict[str, Any]] = {}
        for slot_idx in range(self.num_env_slots):
            symbol = self.slot_to_symbol_map.get(slot_idx)
            units = self.current_positions_units[slot_idx]
            
            if not symbol or abs(units) < Decimal('1e-9'):
                continue

            entry_step = self.position_entry_step_per_slot[slot_idx]
            hold_duration = self.episode_step_count - entry_step if entry_step != -1 else 0

            positions_data[symbol] = {
                "symbol": symbol,
                "units": units,
                "avg_entry_price_qc": self.avg_entry_prices_qc[slot_idx],
                "unrealized_pnl_ac": self.unrealized_pnl_ac[slot_idx],
                "margin_used_ac": self.margin_used_per_position_ac[slot_idx],
                "stop_loss_price_qc": self.stop_loss_prices_qc[slot_idx],
                "hold_duration": hold_duration,
                "trade_type": "long" if units > 0 else "short"
            }
        return positions_data

    def _get_market_data(self, all_prices_map: Dict[str, Tuple[Decimal, Decimal]]) -> Dict[str, Dict[str, Any]]:
        """
        Gathers current market data (prices, ATR) for all tradable symbols.
        """
        market_data = {}
        for slot_idx in self.current_episode_tradable_slot_indices:
            symbol = self.slot_to_symbol_map.get(slot_idx)
            if not symbol:
                continue
            
            bid, ask = all_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
            
            market_data[symbol] = {
                "bid_price_qc": bid,
                "ask_price_qc": ask,
                "mid_price_qc": (bid + ask) / Decimal('2') if bid > 0 and ask > 0 else Decimal('0'),
                "atr_qc": self.atr_values_qc[slot_idx]
            }
        return market_data

    def _calculate_reward(self, portfolio_value_before_action: Decimal, commission_this_step: Decimal, positions_data: Dict[str, Dict[str, Any]], market_data: Dict[str, Dict[str, Any]]) -> Decimal:
        """
        Calculates the reward for the current step using the appropriate calculator.
        """
        if self.portfolio_value_ac <= 0 or portfolio_value_before_action <= 0:
            log_return = Decimal('0.0')
        else:
            log_return = (self.portfolio_value_ac / portfolio_value_before_action).ln()

        if self.use_progressive_rewards and self.progressive_reward_calculator:
            # The progressive calculator expects individual arguments.
            reward_float = self.progressive_reward_calculator.calculate_reward(
                current_portfolio_value=self.portfolio_value_ac,
                prev_portfolio_value=portfolio_value_before_action,
                commission_this_step=commission_this_step,
                trade_log=self.trade_log,  # Pass the full log
                positions_data=positions_data,
                market_data=market_data,
                episode_step=self.episode_step_count
            )
            return Decimal(str(reward_float))
            
        if self.use_enhanced_rewards and self.enhanced_reward_calculator:
            # Note: The enhanced calculator might also need an update if we want to use it.
            return self.enhanced_reward_calculator.calculate_reward(log_return, self.trade_log, self.max_drawdown_episode)

        # Fallback to simple reward
        reward = log_return * self.reward_config["portfolio_log_return_factor"]
        reward -= self.max_drawdown_episode * self.reward_config["max_drawdown_penalty_factor"]
        return reward

    def _set_stop_loss_price(self, slot_idx: int):
        """
        Sets the stop-loss price for a new or modified position based on ATR.
        """
        units = self.current_positions_units[slot_idx]
        if abs(units) < Decimal('1e-9'):
            self.stop_loss_prices_qc[slot_idx] = Decimal('0.0')
            return

        atr = self.atr_values_qc[slot_idx]
        if atr <= 0:
            # Cannot set SL without a valid ATR value.
            self.stop_loss_prices_qc[slot_idx] = Decimal('0.0')
            return

        entry_price = self.avg_entry_prices_qc[slot_idx]
        sl_distance = atr * self.stop_loss_atr_multiplier

        if units > 0:  # Long position
            self.stop_loss_prices_qc[slot_idx] = entry_price - sl_distance
        else:  # Short position
            self.stop_loss_prices_qc[slot_idx] = entry_price + sl_distance

    def _apply_stop_loss(self, all_prices_map: Dict[str, Tuple[Decimal, Decimal]], timestamp: pd.Timestamp):
        """
        Checks for and executes stop-loss orders for all open positions.
        """
        for slot_idx in self.current_episode_tradable_slot_indices:
            units = self.current_positions_units[slot_idx]
            if abs(units) < Decimal('1e-9'):
                continue

            symbol = self.slot_to_symbol_map.get(slot_idx)
            if not symbol: continue

            stop_loss_price = self.stop_loss_prices_qc[slot_idx]
            if stop_loss_price <= 0: continue

            current_bid_qc, current_ask_qc = all_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
            
            should_close = False
            trade_price = Decimal('0')
            if units > 0 and current_bid_qc <= stop_loss_price: # Long position stop-loss triggered
                should_close = True
                trade_price = current_bid_qc
            elif units < 0 and current_ask_qc >= stop_loss_price: # Short position stop-loss triggered
                should_close = True
                trade_price = current_ask_qc

            if should_close:
                logger.info(f"STOP-LOSS TRIGGERED for {symbol} at price {trade_price}. SL Price was {stop_loss_price}.")
                self._execute_trade(slot_idx, -units, trade_price, timestamp, all_prices_map)
                # Stop loss is reset to 0 inside _set_stop_loss_price called by _execute_trade

    def _handle_margin_call(self, all_prices_map: Dict[str, Tuple[Decimal, Decimal]], timestamp: pd.Timestamp):
        """
        Checks for margin call conditions and liquidates positions if necessary.
        OANDA's margin closeout happens when Margin Level (Equity / Margin Used) drops to 50%.
        """
        if self.total_margin_used_ac <= 0:
            return

        margin_level = self.equity_ac / self.total_margin_used_ac
        
        if margin_level < OANDA_MARGIN_CLOSEOUT_LEVEL:
            logger.warning(f"MARGIN CALL! Equity: {self.equity_ac:.2f}, Margin Used: {self.total_margin_used_ac:.2f}, Margin Level: {margin_level:.2%}. Liquidating all positions.")
            
            # Liquidate all positions
            for slot_idx in range(self.num_env_slots):
                units = self.current_positions_units[slot_idx]
                if abs(units) > Decimal('1e-9'):
                    symbol = self.slot_to_symbol_map.get(slot_idx)
                    if not symbol: continue
                    
                    current_bid_qc, current_ask_qc = all_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
                    trade_price = current_bid_qc if units > 0 else current_ask_qc
                    
                    if trade_price > 0:
                        self._execute_trade(slot_idx, -units, trade_price, timestamp, all_prices_map)
            
            # After liquidation, update portfolio value to reflect realized losses
            self._update_portfolio_and_equity_value(all_prices_map)

    def _check_termination_truncation(self) -> Tuple[bool, bool]:
        """
        Checks if the episode should terminate or be truncated.
        """
        terminated = False
        # Termination condition: Portfolio value drops to a very low level (e.g., 10% of initial)
        if self.portfolio_value_ac < self.initial_capital * Decimal('0.1'):
            logger.warning(f"Episode terminated: Portfolio value ({self.portfolio_value_ac:.2f}) dropped below 10% of initial capital.")
            terminated = True

        truncated = False
        # Truncation condition: Reached max episode steps
        if self.episode_step_count >= self.max_episode_steps:
            logger.info(f"Episode truncated: Reached max steps ({self.max_episode_steps}).")
            truncated = True
        
        # Truncation condition: Dataset ends
        if self.current_step_in_dataset >= len(self.dataset) -1:
            truncated = True

        return terminated, truncated

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment.
        """
        self.episode_step_count += 1
        
        # 1. Get current market data for this step
        all_prices_map, current_timestamp = self._get_current_raw_prices_for_all_dataset_symbols()
        
        # 2. Store state before any actions are taken
        portfolio_value_before_action = self.portfolio_value_ac

        # 3. Apply pre-action market mechanics (Stop-Loss, Margin Call)
        self._apply_stop_loss(all_prices_map, current_timestamp)
        self._handle_margin_call(all_prices_map, current_timestamp)
        
        # 4. Update portfolio value after potential automatic liquidations
        self._update_portfolio_and_equity_value(all_prices_map)

        # 5. Let the agent execute its intended actions, generating trades and commissions
        commission_this_step = self._execute_agent_actions(action, all_prices_map, current_timestamp)
        
        # 6. Advance time to the next state in the dataset
        self.current_step_in_dataset += 1
        
        # 7. Get market data for the *next* state to calculate PnL and define the new observation
        all_prices_map_next, _ = self._get_current_raw_prices_for_all_dataset_symbols()
        
        # 8. Update portfolio value, equity, and ATR based on the new prices
        self._update_portfolio_and_equity_value(all_prices_map_next)
        self._update_atr_values(all_prices_map_next) # ATR for the *next* state
        
        # 9. Gather all data required for the reward calculation
        positions_data = self._get_positions_data()
        market_data = self._get_market_data(all_prices_map_next)
        
        # 10. Calculate reward based on the outcome of the step
        reward = self._calculate_reward(
            portfolio_value_before_action=portfolio_value_before_action,
            commission_this_step=commission_this_step,
            positions_data=positions_data,
            market_data=market_data
        )
        self.reward_history.append(float(reward))

        # 11. Check if the episode has ended (terminated or truncated)
        terminated, truncated = self._check_termination_truncation()
        
        # 12. Get the observation and info for the new state
        observation = self._get_observation()
        info = self._get_info()
        
        # 13. Update shared data for external monitoring
        if self.shared_data_manager:
            self.shared_data_manager.update_env_data(self.training_step_offset + self.episode_step_count, info)

        if self.render_mode == 'human':
            self.render()

        return observation, float(reward), terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed); self.current_step_in_dataset = 0; self.episode_step_count = 0
        if len(self.dataset) == 0: raise RuntimeError("Dataset is empty, cannot reset environment.")
        self.current_step_in_dataset = self.np_random.integers(0, len(self.dataset)) # type: ignore
        self.cash = self.initial_capital
        for i in range(self.num_env_slots):
            self.current_positions_units[i] = Decimal('0.0'); self.avg_entry_prices_qc[i] = Decimal('0.0')
            self.unrealized_pnl_ac[i] = Decimal('0.0'); self.margin_used_per_position_ac[i] = Decimal('0.0')
            self.atr_values_qc[i] = Decimal('0.0'); self.stop_loss_prices_qc[i] = Decimal('0.0')
        self.last_trade_step_per_slot.fill(0)
        self.position_entry_step_per_slot.fill(-1)
        self.total_margin_used_ac = Decimal('0.0')
        self.equity_ac = self.initial_capital; self.portfolio_value_ac = self.initial_capital
        self.portfolio_value_history = [float(self.initial_capital)]; self.reward_history = []; self.trade_log = []
        # 初始化價格歷史記錄
        self.price_history = {symbol: deque(maxlen=20) for symbol in self.active_symbols_for_episode}  # 20步波動率窗口
        # 重置收益歷史
        self.returns_history = []
        self.peak_portfolio_value_episode = self.initial_capital; self.max_drawdown_episode = Decimal('0.0')
        logger.debug(f"Env reset. Initial capital: {self.cash} {ACCOUNT_CURRENCY}. Start step: {self.current_step_in_dataset}")
        all_prices_map, _ = self._get_current_raw_prices_for_all_dataset_symbols()
        # 初始化價格歷史
        for symbol in self.active_symbols_for_episode:
            if symbol in all_prices_map:
                bid, ask = all_prices_map[symbol]
                mid_price = (bid + ask) / Decimal('2')
                self.price_history[symbol].append(mid_price)
        self._update_atr_values(all_prices_map); self._update_portfolio_and_equity_value(all_prices_map)
        return self._get_observation(), self._get_info()

    def _get_current_raw_prices_for_all_dataset_symbols(self) -> Tuple[Dict[str, Tuple[Decimal, Decimal]], pd.Timestamp]:
        """
        Safely retrieves the latest raw bid/ask prices for all symbols in the dataset for the current step.
        This method is designed to be resilient to empty datasets or out-of-bounds indices.
        """
        try:
            # Check for an empty or invalid dataset early.
            if len(self.dataset) == 0 or not hasattr(self.dataset, 'symbols') or not self.dataset.symbols:
                logger.warning("Dataset is empty or has no symbols. Returning empty price map.")
                return {}, pd.Timestamp.now(tz=timezone.utc)

            # Ensure the index is always within the valid range [0, len(dataset)-1]
            safe_step_index = min(max(0, self.current_step_in_dataset), len(self.dataset) - 1)

            # Retrieve the data sample and timestamp
            dataset_sample = self.dataset[safe_step_index]
            current_timestamp = self.dataset.aligned_timestamps[safe_step_index]

            # Shape of raw_prices is expected to be (num_symbols, timesteps, num_raw_features)
            # We take the prices from the very last timestep in the history window.
            latest_raw_prices_np = dataset_sample["raw_prices"][:, -1, :].numpy().astype(np.float64)
            
            prices_map: Dict[str, Tuple[Decimal, Decimal]] = {}
            
            # raw_price_columns_ordered: ['bid_close', 'ask_close', 'bid_high', 'bid_low', 'ask_high', 'ask_low']
            # We need bid_close (index 0) and ask_close (index 1)
            for i, symbol_name in enumerate(self.dataset.symbols):
                bid_price = Decimal(str(latest_raw_prices_np[i, 0]))
                ask_price = Decimal(str(latest_raw_prices_np[i, 1]))
                
                if bid_price > 0 and ask_price > 0 and ask_price >= bid_price:
                    prices_map[symbol_name] = (bid_price, ask_price)
                else:
                    # Assign a default of (0, 0) for invalid prices and log it for debugging.
                    prices_map[symbol_name] = (Decimal('0.0'), Decimal('0.0'))
                    logger.debug(f"Symbol {symbol_name} at step {self.current_step_in_dataset} (safe_index: {safe_step_index}) has invalid prices: bid={latest_raw_prices_np[i, 0]}, ask={latest_raw_prices_np[i, 1]}")

            return prices_map, current_timestamp

        except IndexError as e:
            logger.error(f"IndexError in _get_current_raw_prices_for_all_dataset_symbols at step {self.current_step_in_dataset}: {e}", exc_info=True)
            # Fallback for index errors (e.g., empty aligned_timestamps)
            return {}, pd.Timestamp.now(tz=timezone.utc)
        except Exception as e:
            logger.error(f"Unexpected error in _get_current_raw_prices_for_all_dataset_symbols: {e}", exc_info=True)
            # General fallback for any other unexpected errors to ensure stability.
            return {}, pd.Timestamp.now(tz=timezone.utc)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Constructs the observation dictionary for the agent.
        This includes market features, context features, and symbol IDs.
        """
        safe_step_index = min(max(0, self.current_step_in_dataset), len(self.dataset) - 1)
        
        # 1. Get Market Features from the dataset
        # This should be the pre-calculated features for the *current* step
        try:
            dataset_sample = self.dataset[safe_step_index]
            # The shape of features is (num_dataset_symbols, timesteps, num_features)
            # We need the features of the last point in time for each symbol
            market_features_all_symbols = dataset_sample["features"][:, -1, :].numpy().astype(np.float32)
        except IndexError:
            # This can happen if the dataset is shorter than the history length.
            market_features_all_symbols = np.zeros((len(self.dataset.symbols), self.dataset.num_features_per_symbol), dtype=np.float32)


        # Create a mapping from dataset symbol order to env slot order
        market_features_padded = np.zeros((self.num_env_slots, self.dataset.num_features_per_symbol), dtype=np.float32)
        
        dataset_symbol_to_idx_map = {symbol: i for i, symbol in enumerate(self.dataset.symbols)}

        for slot_idx in range(self.num_env_slots):
            symbol = self.slot_to_symbol_map.get(slot_idx)
            if symbol and symbol in dataset_symbol_to_idx_map:
                dataset_idx = dataset_symbol_to_idx_map[symbol]
                market_features_padded[slot_idx, :] = market_features_all_symbols[dataset_idx, :]

        # 2. Construct Context Features
        context_features = np.zeros((self.num_env_slots, self.num_context_features), dtype=np.float32)
        
        all_prices_map, _ = self._get_current_raw_prices_for_all_dataset_symbols()

        for slot_idx in range(self.num_env_slots):
            symbol = self.slot_to_symbol_map.get(slot_idx)
            if not symbol:
                continue

            # a) Position Ratio (normalized by max allowed size)
            current_units = self.current_positions_units[slot_idx]
            _, ask_price_qc = all_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
            
            pos_ratio = Decimal('0.0')
            if ask_price_qc > 0 and self.equity_ac > 0:
                details = self.instrument_details_map[symbol]
                exchange_rate = self._get_exchange_rate_to_account_currency(details.quote_currency, all_prices_map)
                position_value_ac = abs(current_units) * ask_price_qc * exchange_rate
                if self.equity_ac > 0:
                    pos_ratio = position_value_ac / self.equity_ac
            
            normalized_pos_ratio = pos_ratio / self.max_position_size_percentage if self.max_position_size_percentage > 0 else Decimal('0.0')
            final_pos_ratio = float(normalized_pos_ratio.copy_sign(current_units))

            # b) PnL Ratio
            pnl_ac = self.unrealized_pnl_ac[slot_idx]
            pnl_ratio = float(pnl_ac / self.equity_ac) if self.equity_ac > 0 else 0.0

            # c) Time Since Last Trade (normalized)
            time_since_trade = (self.episode_step_count - self.last_trade_step_per_slot[slot_idx]) / self.max_episode_steps if self.max_episode_steps > 0 else 0.0

            # d) Volatility (ATR as a percentage of price)
            atr_qc = self.atr_values_qc[slot_idx]
            bid_price_qc, _ = all_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
            mid_price_qc = (bid_price_qc + ask_price_qc) / Decimal('2')
            volatility = float(atr_qc / mid_price_qc) if mid_price_qc > 0 else 0.0

            # e) Margin Level Proxy (margin for this position / total equity)
            margin_used_ac = self.margin_used_per_position_ac[slot_idx]
            margin_level_proxy = float(margin_used_ac / self.equity_ac) if self.equity_ac > 0 else 0.0

            context_features[slot_idx, :] = [
                np.clip(final_pos_ratio, -1.0, 1.0),
                np.clip(pnl_ratio, -1.0, 1.0),
                np.clip(time_since_trade, 0.0, 1.0),
                np.clip(volatility, 0.0, 1.0),
                np.clip(margin_level_proxy, 0.0, 1.0)
            ]

        # 3. Get Symbol IDs
        symbol_ids = np.full(self.num_env_slots, self.padding_symbol_id, dtype=np.int32)
        for slot_idx in range(self.num_env_slots):
            symbol = self.slot_to_symbol_map.get(slot_idx)
            if symbol and symbol in self.symbol_to_global_id_map:
                symbol_id = self.symbol_to_global_id_map[symbol]
                # Defensive check: ensure the generated ID is within the expected range.
                if symbol_id >= self.num_universe_symbols:
                    logger.warning(
                        f"Generated symbol_id {symbol_id} for symbol '{symbol}' is out of bounds "
                        f"(>= num_universe_symbols {self.num_universe_symbols}). This indicates a mapping mismatch. "
                        f"Using padding ID {self.padding_symbol_id} as a fallback for this step."
                    )
                    symbol_ids[slot_idx] = self.padding_symbol_id
                else:
                    symbol_ids[slot_idx] = symbol_id
            # For non-active slots, the ID remains the padding_symbol_id, which is correct.

        # 4. Build features_from_dataset (sequence) padded to env slots
        # dataset_sample provides features for the episode's active symbols only, in dataset order
        # We must place each dataset symbol into the correct env slot
        seq_len = self.dataset.timesteps_history if hasattr(self.dataset, 'timesteps_history') else 1
        features_from_dataset_padded = np.zeros((self.num_env_slots, seq_len, self.dataset.num_features_per_symbol), dtype=np.float32)
        if 'dataset_idx_to_slot_idx_map' in dir(self) and isinstance(self.dataset_idx_to_slot_idx_map, dict):
            # If there is an explicit mapping prepared earlier, use it
            for dataset_idx, slot_idx in self.dataset_idx_to_slot_idx_map.items():
                if 0 <= dataset_idx < len(self.dataset.symbols) and 0 <= slot_idx < self.num_env_slots:
                    try:
                        features_from_dataset_padded[slot_idx, :, :] = dataset_sample["features"][dataset_idx, :, :].numpy().astype(np.float32)
                    except Exception:
                        # Fall back to zeros if any issue arises
                        pass
        else:
            # Fallback: build a map using slot_to_symbol_map and dataset.symbols
            sym_to_dataset_idx = {sym: i for i, sym in enumerate(getattr(self.dataset, 'symbols', []))}
            for slot_idx in range(self.num_env_slots):
                symbol = self.slot_to_symbol_map.get(slot_idx)
                if symbol in sym_to_dataset_idx:
                    dataset_idx = sym_to_dataset_idx[symbol]
                    try:
                        features_from_dataset_padded[slot_idx, :, :] = dataset_sample["features"][dataset_idx, :, :].numpy().astype(np.float32)
                    except Exception:
                        pass

        # 5. Padding mask: 1 for active slots, 0 for dummy
        padding_mask = np.zeros((self.num_env_slots,), dtype=np.int8)
        for slot_idx in range(self.num_env_slots):
            symbol = self.slot_to_symbol_map.get(slot_idx)
            if symbol is not None and symbol != "":
                padding_mask[slot_idx] = 1

        observation = {
            "market_features": market_features_padded,
            "features_from_dataset": features_from_dataset_padded,
            "context_features": context_features,
            "symbol_id": symbol_ids,
            "padding_mask": padding_mask,
        }
        
        return observation

    def _get_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary with auxiliary information about the environment's state.
        """
        active_positions = {}
        for slot_idx in self.current_episode_tradable_slot_indices:
            symbol = self.slot_to_symbol_map.get(slot_idx)
            if symbol and abs(self.current_positions_units[slot_idx]) > 0:
                active_positions[symbol] = {
                    "units": float(self.current_positions_units[slot_idx]),
                    "avg_entry_price_qc": float(self.avg_entry_prices_qc[slot_idx]),
                    "unrealized_pnl_ac": float(self.unrealized_pnl_ac[slot_idx]),
                    "margin_used_ac": float(self.margin_used_per_position_ac[slot_idx]),
                    "atr_qc": float(self.atr_values_qc[slot_idx]),
                    "stop_loss_price_qc": float(self.stop_loss_prices_qc[slot_idx]),
                }

        info = {
            "episode_step": self.episode_step_count,
            "dataset_step": self.current_step_in_dataset,
            "timestamp": self.dataset.aligned_timestamps[min(self.current_step_in_dataset, len(self.dataset)-1)].isoformat() if len(self.dataset) > 0 and self.current_step_in_dataset < len(self.dataset.aligned_timestamps) else None,
            "portfolio_value_ac": float(self.portfolio_value_ac),
            "equity_ac": float(self.equity_ac),
            "cash_ac": float(self.cash),
            "total_margin_used_ac": float(self.total_margin_used_ac),
            "margin_level": float(self.equity_ac / self.total_margin_used_ac) if self.total_margin_used_ac > 0 else float('inf'),
            "max_drawdown_episode": float(self.max_drawdown_episode),
            "active_positions_count": len(active_positions),
            "active_positions": active_positions,
            "total_trades_in_episode": len(self.trade_log),
            "active_symbols_for_episode": self.active_symbols_for_episode,
            "num_tradable_symbols_this_episode": self.num_tradable_symbols_this_episode,
        }
        return info
