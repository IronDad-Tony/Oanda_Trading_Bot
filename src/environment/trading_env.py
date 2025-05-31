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
# This ensures that imports like 'from src.common import ...' work.
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
    from src.common.logger_setup import logger as common_configured_logger; logger = common_configured_logger; _logger_initialized_by_common_env_v5 = True
    if not _import_logged:
        logger.debug("trading_env.py (V5.0): Successfully imported logger from common.logger_setup.")
    from src.common.config import (TIMESTEPS as _TIMESTEPS, MAX_SYMBOLS_ALLOWED as _MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY as _ACCOUNT_CURRENCY, INITIAL_CAPITAL as _DEFAULT_INITIAL_CAPITAL, OANDA_MARGIN_CLOSEOUT_LEVEL as _OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE as _TRADE_COMMISSION_PERCENTAGE, OANDA_API_KEY as _OANDA_API_KEY, ATR_PERIOD as _ATR_PERIOD, STOP_LOSS_ATR_MULTIPLIER as _STOP_LOSS_ATR_MULTIPLIER, MAX_ACCOUNT_RISK_PERCENTAGE as _MAX_ACCOUNT_RISK_PERCENTAGE)
    _config_values_env_v5 = {"TIMESTEPS": _TIMESTEPS, "MAX_SYMBOLS_ALLOWED": _MAX_SYMBOLS_ALLOWED, "ACCOUNT_CURRENCY": _ACCOUNT_CURRENCY, "DEFAULT_INITIAL_CAPITAL": _DEFAULT_INITIAL_CAPITAL, "OANDA_MARGIN_CLOSEOUT_LEVEL": _OANDA_MARGIN_CLOSEOUT_LEVEL, "TRADE_COMMISSION_PERCENTAGE": _TRADE_COMMISSION_PERCENTAGE, "OANDA_API_KEY": _OANDA_API_KEY, "ATR_PERIOD": _ATR_PERIOD, "STOP_LOSS_ATR_MULTIPLIER": _STOP_LOSS_ATR_MULTIPLIER, "MAX_ACCOUNT_RISK_PERCENTAGE": _MAX_ACCOUNT_RISK_PERCENTAGE}
    if not _import_logged:
        logger.info("trading_env.py (V5.0): Successfully imported and stored common.config values.")
    from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset; from src.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols; from src.data_manager.instrument_info_manager import InstrumentDetails, InstrumentInfoManager;
    if not _import_logged:
        logger.info("trading_env.py (V5.0): Successfully imported other dependencies.")
        _import_logged = True
except ImportError as e_initial_import_v5:
    logger_temp_v5 = logging.getLogger("trading_env_v5_fallback_initial"); logger_temp_v5.addHandler(logging.StreamHandler(sys.stdout)); logger_temp_v5.setLevel(logging.DEBUG); logger = logger_temp_v5
    logger.warning(f"trading_env.py (V5.0): Initial import failed: {e_initial_import_v5}. Assuming PYTHONPATH is set correctly or this is a critical issue.")
    # project_root_env_v5 = Path(__file__).resolve().parent.parent.parent # 移除
    # if str(project_root_env_v5) not in sys.path: sys.path.insert(0, str(project_root_env_v5)); logger.info(f"trading_env.py (V5.0): Added project root to sys.path: {project_root_env_v5}") # 移除
    try:
        # 假設 PYTHONPATH 已設定，這些導入應該能工作
        from src.common.logger_setup import logger as common_logger_retry_v5; logger = common_logger_retry_v5; _logger_initialized_by_common_env_v5 = True; logger.info("trading_env.py (V5.0): Successfully re-imported common_logger.")
        from src.common.config import (TIMESTEPS as _TIMESTEPS_R, MAX_SYMBOLS_ALLOWED as _MAX_SYMBOLS_ALLOWED_R, ACCOUNT_CURRENCY as _ACCOUNT_CURRENCY_R, INITIAL_CAPITAL as _DEFAULT_INITIAL_CAPITAL_R, OANDA_MARGIN_CLOSEOUT_LEVEL as _OANDA_MARGIN_CLOSEOUT_LEVEL_R, TRADE_COMMISSION_PERCENTAGE as _TRADE_COMMISSION_PERCENTAGE_R, OANDA_API_KEY as _OANDA_API_KEY_R, ATR_PERIOD as _ATR_PERIOD_R, STOP_LOSS_ATR_MULTIPLIER as _STOP_LOSS_ATR_MULTIPLIER_R, MAX_ACCOUNT_RISK_PERCENTAGE as _MAX_ACCOUNT_RISK_PERCENTAGE_R)
        _config_values_env_v5 = {"TIMESTEPS": _TIMESTEPS_R, "MAX_SYMBOLS_ALLOWED": _MAX_SYMBOLS_ALLOWED_R, "ACCOUNT_CURRENCY": _ACCOUNT_CURRENCY_R, "DEFAULT_INITIAL_CAPITAL": _DEFAULT_INITIAL_CAPITAL_R, "OANDA_MARGIN_CLOSEOUT_LEVEL": _OANDA_MARGIN_CLOSEOUT_LEVEL_R, "TRADE_COMMISSION_PERCENTAGE": _TRADE_COMMISSION_PERCENTAGE_R, "OANDA_API_KEY": _OANDA_API_KEY_R, "ATR_PERIOD": _ATR_PERIOD_R, "STOP_LOSS_ATR_MULTIPLIER": _STOP_LOSS_ATR_MULTIPLIER_R, "MAX_ACCOUNT_RISK_PERCENTAGE": _MAX_ACCOUNT_RISK_PERCENTAGE_R}
        logger.info("trading_env.py (V5.0): Successfully re-imported and stored common.config after path adjustment.")
        from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset; from src.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols; from src.data_manager.instrument_info_manager import InstrumentDetails, InstrumentInfoManager; 
        # Import enhanced reward calculator
        try:
            from src.environment.enhanced_reward_calculator import EnhancedRewardCalculator
            logger.info("trading_env.py (V5.0): Successfully imported EnhancedRewardCalculator")
        except ImportError as e_reward_calc:
            logger.warning(f"trading_env.py (V5.0): Failed to import EnhancedRewardCalculator: {e_reward_calc}")
            EnhancedRewardCalculator = None
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

TIMESTEPS = _config_values_env_v5.get("TIMESTEPS", 128); MAX_SYMBOLS_ALLOWED = _config_values_env_v5.get("MAX_SYMBOLS_ALLOWED", 20); ACCOUNT_CURRENCY = _config_values_env_v5.get("ACCOUNT_CURRENCY", "AUD"); DEFAULT_INITIAL_CAPITAL = _config_values_env_v5.get("DEFAULT_INITIAL_CAPITAL", 100000.0); OANDA_MARGIN_CLOSEOUT_LEVEL = _config_values_env_v5.get("OANDA_MARGIN_CLOSEOUT_LEVEL", Decimal('0.50')); TRADE_COMMISSION_PERCENTAGE = _config_values_env_v5.get("TRADE_COMMISSION_PERCENTAGE", Decimal('0.0001')); OANDA_API_KEY = _config_values_env_v5.get("OANDA_API_KEY", None); ATR_PERIOD = _config_values_env_v5.get("ATR_PERIOD", 14); STOP_LOSS_ATR_MULTIPLIER = _config_values_env_v5.get("STOP_LOSS_ATR_MULTIPLIER", Decimal('2.0')); MAX_ACCOUNT_RISK_PERCENTAGE = _config_values_env_v5.get("MAX_ACCOUNT_RISK_PERCENTAGE", Decimal('0.01'))


class UniversalTradingEnvV4(gym.Env): # 保持類名為V4，但內部是V5邏輯
    metadata = {'render_modes': ['human', 'array'], 'render_fps': 10}
    
    def __init__(self, dataset: UniversalMemoryMappedDataset, instrument_info_manager: InstrumentInfoManager, active_symbols_for_episode: List[str], # type: ignore
                 initial_capital: float = float(DEFAULT_INITIAL_CAPITAL), max_episode_steps: Optional[int] = None,
                 commission_percentage_override: Optional[float] = None, reward_config: Optional[Dict[str, Union[float, Decimal]]] = None,
                 max_account_risk_per_trade: float = float(MAX_ACCOUNT_RISK_PERCENTAGE),
                 stop_loss_atr_multiplier: float = float(STOP_LOSS_ATR_MULTIPLIER),
                 atr_period: int = ATR_PERIOD, render_mode: Optional[str] = None,
                 shared_data_manager=None, training_step_offset: int = 0):
        super().__init__()
        self.dataset = dataset
        self.instrument_info_manager = instrument_info_manager
        self.initial_capital = Decimal(str(initial_capital))
        if commission_percentage_override is not None:
            self.commission_percentage = Decimal(str(commission_percentage_override))
        else:
            self.commission_percentage = Decimal(str(TRADE_COMMISSION_PERCENTAGE))
        self.render_mode = render_mode
        self.max_account_risk_per_trade = Decimal(str(max_account_risk_per_trade))
        self.stop_loss_atr_multiplier = Decimal(str(stop_loss_atr_multiplier))
        self.atr_period = atr_period
        
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
        
        # Initialize Enhanced Reward Calculator if available
        self.enhanced_reward_calculator = None
        self.use_enhanced_rewards = False
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
                logger.info("EnhancedRewardCalculator successfully initialized with improved parameters")
            except Exception as e:
                logger.warning(f"Failed to initialize EnhancedRewardCalculator: {e}. Falling back to standard rewards.")
                self.enhanced_reward_calculator = None
                self.use_enhanced_rewards = False
        else:
            logger.info("EnhancedRewardCalculator not available. Using standard reward calculation.")
        
        self.peak_portfolio_value_episode: Decimal = self.initial_capital; self.max_drawdown_episode: Decimal = Decimal('0.0')
        obs_spaces = {
            "features_from_dataset": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_env_slots, self.dataset.timesteps_history, self.dataset.num_features_per_symbol), dtype=np.float32),
            "current_positions_nominal_ratio_ac": spaces.Box(low=-5.0, high=5.0, shape=(self.num_env_slots,), dtype=np.float32),
            "unrealized_pnl_ratio_ac": spaces.Box(low=-1.0, high=5.0, shape=(self.num_env_slots,), dtype=np.float32),
            "margin_level": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            "time_since_last_trade_ratio": spaces.Box(low=0.0, high=1.0, shape=(self.num_env_slots,), dtype=np.float32),
            "volatility": spaces.Box(low=0.0, high=1.0, shape=(self.num_env_slots,), dtype=np.float32),  # 新增波動率特徵
            "padding_mask": spaces.Box(low=0, high=1, shape=(self.num_env_slots,), dtype=np.bool_)}
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_env_slots,), dtype=np.float32)
        if self.render_mode == 'human': self._init_render_figure()
        logger.info(f"UniversalTradingEnvV4 (整合量子策略層) 初始化完成。")

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
        dataset_sample = self.dataset[min(self.current_step_in_dataset, len(self.dataset)-1)]
        latest_raw_prices_np = dataset_sample["raw_prices"][:, -1, :].numpy().astype(np.float64)
        prices_map: Dict[str, Tuple[Decimal, Decimal]] = {}
        for i, symbol_name in enumerate(self.dataset.symbols):
            bid_price = Decimal(str(latest_raw_prices_np[i, 0])); ask_price = Decimal(str(latest_raw_prices_np[i, 1]))
            if bid_price > 0 and ask_price > 0 and ask_price >= bid_price: prices_map[symbol_name] = (bid_price, ask_price)
            else: prices_map[symbol_name] = (Decimal('0.0'), Decimal('0.0')); logger.debug(f"Symbol {symbol_name} invalid prices: bid={latest_raw_prices_np[i, 0]}, ask={latest_raw_prices_np[i, 1]}")
        timestamp_index = self.current_step_in_dataset + self.dataset.timesteps_history - 1
        timestamp_index = min(timestamp_index, len(self.dataset.aligned_timestamps)-1)
        return prices_map, self.dataset.aligned_timestamps[timestamp_index]
    
    def _get_specific_rate(self, base_curr: str, quote_curr: str, current_prices_map: Dict[str, Tuple[Decimal, Decimal]]) -> Optional[Decimal]:
        base_curr_upper = base_curr.upper(); quote_curr_upper = quote_curr.upper()
        if base_curr_upper == quote_curr_upper: return Decimal('1.0')
        pair1 = f"{base_curr_upper}_{quote_curr_upper}"; pair2 = f"{quote_curr_upper}_{base_curr_upper}"
        price_pair1_tuple = current_prices_map.get(pair1)
        if price_pair1_tuple and price_pair1_tuple[1] > 0: return price_pair1_tuple[1]
        price_pair2_tuple = current_prices_map.get(pair2)
        if price_pair2_tuple and price_pair2_tuple[0] > 0: return Decimal('1.0') / price_pair2_tuple[0]
        return None

    def _get_exchange_rate_to_account_currency(self, from_currency: str, current_prices_map: Dict[str, Tuple[Decimal, Decimal]]) -> Decimal:
        """
        根據Oanda實際轉換規則實現的匯率轉換邏輯：
        1. 優先使用直接貨幣對報價
        2. 次選通過USD中轉（Oanda標準做法）
        3. 最後嘗試其他主要貨幣中轉
        """
        from_currency_upper = from_currency.upper()
        account_currency_upper = ACCOUNT_CURRENCY.upper()
        
        # 1. 相同貨幣直接返回1.0
        if from_currency_upper == account_currency_upper:
            return Decimal('1.0')
        
        # 2. 嘗試直接匯率 (AUD_CAD)
        direct_pair = f"{from_currency_upper}_{account_currency_upper}"
        direct_rate = self._get_specific_rate(from_currency_upper, account_currency_upper, current_prices_map)
        if direct_rate and direct_rate > 0:
            return direct_rate
        
        # 3. 嘗試反向匯率 (CAD_AUD)
        reverse_pair = f"{account_currency_upper}_{from_currency_upper}"
        reverse_rate = self._get_specific_rate(account_currency_upper, from_currency_upper, current_prices_map)
        if reverse_rate and reverse_rate > 0:
            return Decimal('1.0') / reverse_rate
        
        # 4. 標準Oanda轉換：通過USD中轉
        # 符合Oanda實際結算規則
        if from_currency_upper != "USD":
            # 源貨幣→USD
            rate_from_usd = self._get_specific_rate(from_currency_upper, "USD", current_prices_map)
        else:
            rate_from_usd = Decimal('1.0')
            
        if account_currency_upper != "USD":
            # USD→目標貨幣
            rate_usd_to_ac = self._get_specific_rate("USD", account_currency_upper, current_prices_map)
        else:
            rate_usd_to_ac = Decimal('1.0')
            
        if rate_from_usd and rate_usd_to_ac and rate_from_usd > 0 and rate_usd_to_ac > 0:
            return rate_from_usd * rate_usd_to_ac
        
        # 5. 加強版：非標準貨幣對通過EUR中轉
        if from_currency_upper != "EUR":
            rate_from_eur = self._get_specific_rate(from_currency_upper, "EUR", current_prices_map)
        else:
            rate_from_eur = Decimal('1.0')
            
        if account_currency_upper != "EUR":
            rate_eur_to_ac = self._get_specific_rate("EUR", account_currency_upper, current_prices_map)
        else:
            rate_eur_to_ac = Decimal('1.0')
            
        if rate_from_eur and rate_eur_to_ac and rate_from_eur > 0 and rate_eur_to_ac > 0:
            return rate_from_eur * rate_eur_to_ac
        
        # 6. 終極備援：嘗試其他主要貨幣中轉
        for intermediate in ["GBP", "JPY", "CHF", "CAD", "AUD"]:
            if intermediate == from_currency_upper or intermediate == account_currency_upper:
                continue
                
            # 獲取中轉匯率
            rate_to_intermediate = self._get_specific_rate(from_currency_upper, intermediate, current_prices_map)
            rate_from_intermediate = self._get_specific_rate(intermediate, account_currency_upper, current_prices_map)
                
            if rate_to_intermediate and rate_from_intermediate and rate_to_intermediate > 0 and rate_from_intermediate > 0:
                return rate_to_intermediate * rate_from_intermediate
        
        # 7. 安全後備
        available_pairs = ", ".join(current_prices_map.keys())
        logger.warning(f"Oanda標準轉換失敗: {from_currency}→{ACCOUNT_CURRENCY}，可用的貨幣對: [{available_pairs}]，使用安全值1.0")
        return Decimal('1.0')
    
    def _update_atr_values(self, current_prices_map: Dict[str, Tuple[Decimal, Decimal]]):
        for slot_idx in range(self.num_env_slots):
            symbol = self.slot_to_symbol_map.get(slot_idx)
            if not symbol: self.atr_values_qc[slot_idx] = Decimal('0.0'); continue
            details = self.instrument_details_map.get(symbol); price_tuple = current_prices_map.get(symbol)
            if not details or not price_tuple: self.atr_values_qc[slot_idx] = Decimal('0.0'); continue
            bid_qc, ask_qc = price_tuple
            if bid_qc <=0 or ask_qc <=0 or ask_qc < bid_qc: self.atr_values_qc[slot_idx] = details.pip_value_in_quote_currency_per_unit * Decimal('10'); continue
            spread = ask_qc - bid_qc; self.atr_values_qc[slot_idx] = max(spread * Decimal('5'), details.pip_value_in_quote_currency_per_unit * Decimal('10'))

    def _update_stop_loss_prices(self, current_prices_map: Dict[str, Tuple[Decimal, Decimal]]):
        sl_multiplier = self.stop_loss_atr_multiplier
        for slot_idx in range(self.num_env_slots):
            units = self.current_positions_units[slot_idx]
            if abs(units) > Decimal('1e-9'):
                avg_entry = self.avg_entry_prices_qc[slot_idx]; atr = self.atr_values_qc[slot_idx]
                if atr <= Decimal(0): self.stop_loss_prices_qc[slot_idx] = Decimal('0.0'); continue
                if units > 0: self.stop_loss_prices_qc[slot_idx] = avg_entry - (atr * sl_multiplier)
                else: self.stop_loss_prices_qc[slot_idx] = avg_entry + (atr * sl_multiplier)
            else: self.stop_loss_prices_qc[slot_idx] = Decimal('0.0')

    def _update_portfolio_and_equity_value(self, current_prices_map: Dict[str, Tuple[Decimal, Decimal]]):
        self.equity_ac = self.cash
        for slot_idx in range(self.num_env_slots):
            self.unrealized_pnl_ac[slot_idx] = Decimal('0.0')
            units = self.current_positions_units[slot_idx]
            if abs(units) > Decimal('1e-9'):
                symbol = self.slot_to_symbol_map.get(slot_idx); avg_entry_qc = self.avg_entry_prices_qc[slot_idx]
                if not symbol or not avg_entry_qc or avg_entry_qc <= Decimal('0'): continue # avg_entry_qc 在平倉後為0
                details = self.instrument_details_map[symbol]; price_tuple = current_prices_map.get(symbol)
                if not price_tuple: continue
                current_bid_qc, current_ask_qc = price_tuple   # 提取買價和賣價
                current_price_qc = current_bid_qc if units > 0 else current_ask_qc # Bid for long, Ask for short
                if current_price_qc <= Decimal('0'): continue
                # Oanda精確損益計算（考慮點差成本）
                if units > 0:  # 多頭倉位
                    pnl_per_unit_qc = current_bid_qc - avg_entry_qc  # 平倉用Bid價
                else:  # 空頭倉位
                    pnl_per_unit_qc = avg_entry_qc - current_ask_qc  # 平倉用Ask價
                
                # 點差成本（Oanda實際收取） - 使用已定義的變量
                spread_cost = (current_ask_qc - current_bid_qc) * abs(units) * Decimal('0.5')  # 50%點差成本
                total_pnl_qc = pnl_per_unit_qc * abs(units) - spread_cost
                pnl_in_ac = total_pnl_qc
                if details.quote_currency != ACCOUNT_CURRENCY:
                    exchange_rate_qc_to_ac = self.currency_manager.convert_to_account_currency(details.quote_currency, current_prices_map)
                    pnl_in_ac = total_pnl_qc * exchange_rate_qc_to_ac
                self.unrealized_pnl_ac[slot_idx] = pnl_in_ac
                self.equity_ac += pnl_in_ac
        self.portfolio_value_ac = self.equity_ac # 淨值等於權益

    def _execute_trade(self, slot_idx: int, units_to_trade: Decimal, trade_price_qc: Decimal, current_timestamp: pd.Timestamp, all_prices_map: Dict[str, Tuple[Decimal, Decimal]]) -> Tuple[Decimal, Decimal]:
        symbol = self.slot_to_symbol_map[slot_idx]
        if not symbol:
            logger.error(f"嘗試在無效槽位 {slot_idx} 執行交易。")
            return Decimal('0.0'), Decimal('0.0')

        details = self.instrument_details_map[symbol]
        current_units = self.current_positions_units[slot_idx]
        avg_entry_qc = self.avg_entry_prices_qc[slot_idx]
        commission_ac = Decimal('0.0')
        realized_pnl_ac = Decimal('0.0')
        trade_type = "UNKNOWN"
        
        # 計算交易名義價值 (報價貨幣和賬戶貨幣)
        nominal_value_qc = abs(units_to_trade) * trade_price_qc
        exchange_rate_qc_to_ac = self._get_exchange_rate_to_account_currency(details.quote_currency, all_prices_map)
        if exchange_rate_qc_to_ac <= Decimal('0'):
            logger.warning(f"無法獲取 {details.quote_currency} 到 {ACCOUNT_CURRENCY} 的匯率，取消交易。")
            return Decimal('0.0'), Decimal('0.0')
        nominal_value_ac = nominal_value_qc * exchange_rate_qc_to_ac

        # 計算手續費並從 cash 中扣除
        commission_ac = nominal_value_ac * self.commission_percentage
        if self.cash < commission_ac:
            logger.warning(f"現金不足支付手續費 {commission_ac:.2f} AC，取消交易。")
            return Decimal('0.0'), Decimal('0.0')
        self.cash -= commission_ac

        # 處理開倉/平倉/加倉/減倉/反向開倉
        new_units = current_units + units_to_trade

        if abs(current_units) < Decimal('1e-9'): # 當前無倉位
            trade_type = "OPEN"
            self.avg_entry_prices_qc[slot_idx] = trade_price_qc
        elif current_units.copy_sign(Decimal('1')) == units_to_trade.copy_sign(Decimal('1')): # 同向加倉
            trade_type = "ADD"
            # 重新計算加權平均價
            total_value_at_old_avg = current_units * avg_entry_qc
            total_value_at_new_trade = units_to_trade * trade_price_qc
            
            # 防止除零錯誤：檢查new_units是否為零
            if abs(new_units) > Decimal('1e-9'):
                self.avg_entry_prices_qc[slot_idx] = (total_value_at_old_avg + total_value_at_new_trade) / new_units
            else:
                # 如果new_units接近零，保持原有的平均入場價格
                logger.warning(f"new_units接近零 ({new_units})，保持原有平均入場價格: {self.avg_entry_prices_qc[slot_idx]}")
        else: # 反向交易 (平倉或反向開倉)
            if abs(units_to_trade) >= abs(current_units): # 完全平倉或反向開倉
                trade_type = "CLOSE_AND_REVERSE" if abs(units_to_trade) > abs(current_units) else "CLOSE"
                # 計算已實現盈虧
                pnl_per_unit_qc = (trade_price_qc - avg_entry_qc) if current_units > 0 else (avg_entry_qc - trade_price_qc)
                realized_pnl_qc = pnl_per_unit_qc * abs(current_units)
                realized_pnl_ac = realized_pnl_qc * exchange_rate_qc_to_ac
                self.cash += realized_pnl_ac
                if abs(new_units) < Decimal('1e-9'): # 完全平倉
                    self.avg_entry_prices_qc[slot_idx] = Decimal('0.0')
                else: # 反向開倉
                    self.avg_entry_prices_qc[slot_idx] = trade_price_qc
            else: # 部分平倉
                trade_type = "REDUCE"
                # 計算已實現盈虧 (只針對平倉部分)
                pnl_per_unit_qc = (trade_price_qc - avg_entry_qc) if current_units > 0 else (avg_entry_qc - trade_price_qc)
                realized_pnl_qc = pnl_per_unit_qc * abs(units_to_trade)
                realized_pnl_ac = realized_pnl_qc * exchange_rate_qc_to_ac
                self.cash += realized_pnl_ac
                # 平均入場價不變

        self.current_positions_units[slot_idx] = new_units

        # 更新該倉位的已用保證金 (根據新的持倉單位、實際交易價格和保證金率)
        current_bid_qc, current_ask_qc = all_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
        
        # 修復：使用實際交易價格而非中間價計算保證金
        if abs(new_units) > Decimal('1e-9') and trade_price_qc > Decimal('0'):
            # Oanda標準保證金計算: 單位數 * 合約大小 * 市場價格 * 保證金率
            contract_size = Decimal('100000')  # Oanda標準合約大小
            margin_required_qc = abs(new_units) * contract_size * trade_price_qc * Decimal(str(details.margin_rate))
            
            # 根據波動性增加額外保證金要求
            volatility_factor = Decimal('1.0') + (self.atr_values_qc[slot_idx] / trade_price_qc) * Decimal('5.0')
            margin_required_qc *= volatility_factor
            
            # 轉換為賬戶貨幣
            self.margin_used_per_position_ac[slot_idx] = margin_required_qc * exchange_rate_qc_to_ac
            
            # 添加2%緩衝區（Oanda實際會根據波動性動態調整）
            margin_buffer = Decimal('0.02')
            self.margin_used_per_position_ac[slot_idx] *= (Decimal('1.0') + margin_buffer)
        else:
            self.margin_used_per_position_ac[slot_idx] = Decimal('0.0')

        # 記錄到 trade_log
        trade_direction = "LONG" if units_to_trade > 0 else "SHORT"
        position_direction = "LONG" if new_units > 0 else ("SHORT" if new_units < 0 else "FLAT")
        self.trade_log.append({
            "step": self.episode_step_count,
            "timestamp": current_timestamp.isoformat(),
            "symbol": symbol,
            "trade_type": trade_type,
            "trade_direction": trade_direction,
            "position_direction": position_direction,
            "units_traded": float(units_to_trade),
            "trade_price_qc": float(trade_price_qc),
            "trade_price_ac": float(trade_price_qc * exchange_rate_qc_to_ac), # 這裡的交易價格是QC，轉換為AC
            "realized_pnl_ac": float(realized_pnl_ac),
            "commission_ac": float(commission_ac),
            "current_position_units": float(new_units),
            "avg_entry_price_qc": float(self.avg_entry_prices_qc[slot_idx]),
            "margin_used_ac": float(self.margin_used_per_position_ac[slot_idx]),
            "cash_after_trade": float(self.cash),
            "equity_after_trade": float(self.equity_ac + realized_pnl_ac - commission_ac) # 這裡的equity_after_trade是預估值，最終會在_update_portfolio_and_equity_value更新
        })
        self.last_trade_step_per_slot[slot_idx] = self.episode_step_count
          # Log trade to shared data manager for real-time monitoring
        if self.shared_data_manager is not None:
            # Calculate global training step
            global_training_step = self.training_step_offset + self.episode_step_count
            
            # Determine action for shared data manager with detailed format: [Long/Short] - [Trade Type]
            position_direction = "Long" if units_to_trade > 0 else "Short"
            
            # Map trade_type to user-friendly terms
            trade_type_mapping = {
                "OPEN": "Open",
                "ADD": "Add", 
                "REDUCE": "Reduce",
                "CLOSE": "Close",
                "CLOSE_AND_REVERSE": "Close & Reverse"
            }
            trade_action = trade_type_mapping.get(trade_type, trade_type)
            
            action_str = f"{position_direction} - {trade_action}"
            
            # Convert price to account currency for consistency
            trade_price_ac = float(trade_price_qc * exchange_rate_qc_to_ac)
            
            # Log trade with training step as primary time axis
            self.shared_data_manager.add_trade_record(
                symbol=symbol,
                action=action_str,
                price=trade_price_ac,  # Price in account currency
                quantity=float(abs(units_to_trade)),
                profit_loss=float(realized_pnl_ac),
                training_step=global_training_step,  # Primary time axis
                timestamp=current_timestamp.to_pydatetime()  # Auxiliary time information
            )
        
        logger.info(f"執行交易: {symbol}, 類型: {trade_type}, 單位: {units_to_trade:.2f}, 價格: {trade_price_qc:.5f} QC, 手續費: {commission_ac:.2f} AC, 實現盈虧: {realized_pnl_ac:.2f} AC, 現金: {self.cash:.2f} AC, 新倉位: {new_units:.2f}")
        return units_to_trade, commission_ac

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        _step_time_start = time.perf_counter()
        logger.debug(f"--- Step {self.episode_step_count} Start ---")

        self.episode_step_count += 1
        
        _t_get_prices = time.perf_counter()
        all_prices_map, current_timestamp = self._get_current_raw_prices_for_all_dataset_symbols()
        logger.debug(f"Step {self.episode_step_count}: _get_current_raw_prices_for_all_dataset_symbols took {time.perf_counter() - _t_get_prices:.6f}s")
        
        # 更新價格歷史
        for symbol in self.active_symbols_for_episode:
            if symbol in all_prices_map:
                bid, ask = all_prices_map[symbol]
                mid_price = (bid + ask) / Decimal('2')
                self.price_history[symbol].append(mid_price)
        
        prev_portfolio_value_ac = self.portfolio_value_ac
        
        _t_update_atr_sl = time.perf_counter()
        self._update_atr_values(all_prices_map); self._update_stop_loss_prices(all_prices_map)
        logger.debug(f"Step {self.episode_step_count}: _update_atr_values & _update_stop_loss_prices took {time.perf_counter() - _t_update_atr_sl:.6f}s")
        
        total_commission_this_step_ac = Decimal('0.0')

        # 1. 處理止損
        _t_apply_sl = time.perf_counter()
        commission_from_sl = self._apply_stop_loss(all_prices_map, current_timestamp)
        total_commission_this_step_ac += commission_from_sl
        logger.debug(f"Step {self.episode_step_count}: _apply_stop_loss took {time.perf_counter() - _t_apply_sl:.6f}s, commission: {commission_from_sl}")

        # 2. 處理保證金追繳 (如果止損後仍觸發)
        _t_handle_mc = time.perf_counter()
        commission_from_mc = self._handle_margin_call(all_prices_map, current_timestamp)
        total_commission_this_step_ac += commission_from_mc
        logger.debug(f"Step {self.episode_step_count}: _handle_margin_call took {time.perf_counter() - _t_handle_mc:.6f}s, commission: {commission_from_mc}")

        # 3. 執行智能體動作
        _t_agent_actions_start = time.perf_counter()
        for slot_idx in self.current_episode_tradable_slot_indices:
            _t_slot_action_start = time.perf_counter()
            symbol = self.slot_to_symbol_map.get(slot_idx)
            if not symbol: continue

            details = self.instrument_info_manager.get_details(symbol)
            current_units = self.current_positions_units[slot_idx]
            current_bid_qc, current_ask_qc = all_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
            
            if current_bid_qc <= Decimal('0') or current_ask_qc <= Decimal('0'):
                logger.warning(f"Step {self.episode_step_count} Symbol {symbol}: Skipping trade due to invalid current prices.")
                continue

            # 計算目標單位數 (target_units)
            risk_per_unit_qc = self.atr_values_qc[slot_idx] * self.stop_loss_atr_multiplier
            if risk_per_unit_qc <= Decimal('0'):
                logger.debug(f"Step {self.episode_step_count} Symbol {symbol}: Skipping trade due to zero ATR risk.")
                continue

            exchange_rate_qc_to_ac = self._get_exchange_rate_to_account_currency(details.quote_currency, all_prices_map)
            if exchange_rate_qc_to_ac <= Decimal('0'):
                logger.warning(f"Step {self.episode_step_count} Symbol {symbol}: Cannot get exchange rate for {details.quote_currency} to {ACCOUNT_CURRENCY}, skipping trade.")
                continue
            
            risk_per_unit_ac = risk_per_unit_qc * exchange_rate_qc_to_ac
            if risk_per_unit_ac <= Decimal('0'):
                logger.debug(f"Step {self.episode_step_count} Symbol {symbol}: Skipping trade due to zero risk per unit in AC.")
                continue

            max_risk_capital = self.equity_ac * self.max_account_risk_per_trade
            max_units_by_risk = (max_risk_capital / risk_per_unit_ac).quantize(Decimal('1'), rounding=ROUND_DOWN)

            target_position_ratio = Decimal(str(action[slot_idx]))
            target_nominal_value_ac = abs(target_position_ratio) * self.equity_ac
            current_mid_price_qc = (current_bid_qc + current_ask_qc) / Decimal('2')

            if current_mid_price_qc <= Decimal('0'):
                logger.warning(f"Step {self.episode_step_count} Symbol {symbol}: Skipping trade due to invalid mid price.")
                continue

            target_units_raw = (target_nominal_value_ac / (current_mid_price_qc * exchange_rate_qc_to_ac)).quantize(Decimal('1e-9'), rounding=ROUND_HALF_UP)
            target_units_final = min(target_units_raw, max_units_by_risk)
            target_units_final = target_units_final.copy_sign(target_position_ratio)
            target_units = details.round_units(target_units_final)
            units_to_trade = target_units - current_units
            
            # Oanda倉位控制規則
            min_size = details.minimum_trade_size
            max_size = details.max_trade_units if details.max_trade_units is not None else Decimal('1000000')  # 修正屬性名並處理None情況
            
            if abs(units_to_trade) < min_size:
                logger.debug(f"交易單位 {units_to_trade} 低於最小值 {min_size}，取消交易")
                continue
                
            if abs(units_to_trade) > max_size:
                logger.info(f"交易單位 {units_to_trade} 超過最大值 {max_size}，自動調整")
                units_to_trade = max_size.copy_sign(units_to_trade)
                
            # 確定交易價格（用於保證金計算）
            trade_price_qc = current_ask_qc if units_to_trade > 0 else current_bid_qc
            if trade_price_qc <= Decimal('0'):
                logger.warning(f"Step {self.episode_step_count} Symbol {symbol}: Skipping trade due to invalid trade price.")
                continue

            # 精確保證金檢查（Oanda實時風控）
            # 使用Oanda實際合約大小計算保證金
            contract_size = Decimal('100000')  # Oanda標準合約大小
            margin_required_qc = abs(units_to_trade) * contract_size * trade_price_qc * Decimal(str(details.margin_rate))
            # 根據波動性增加額外保證金要求
            volatility_factor = Decimal('1.0') + (self.atr_values_qc[slot_idx] / trade_price_qc) * Decimal('5.0')
            margin_required_qc *= volatility_factor
            # 轉換為賬戶貨幣
            margin_required_ac = margin_required_qc * exchange_rate_qc_to_ac
            
            if margin_required_ac > self.cash * Decimal('0.9'):  # 保留10%現金緩衝
                logger.warning(f"保證金不足: 需要{margin_required_ac:.2f} AC, 可用{self.cash:.2f} AC")
                continue
            
            _t_margin_check_start = time.perf_counter()
            projected_new_units = current_units + units_to_trade
            projected_margin_required_qc = abs(projected_new_units) * trade_price_qc * Decimal(str(details.margin_rate))
            projected_margin_required_ac = projected_margin_required_qc * exchange_rate_qc_to_ac
            current_margin_for_slot_ac = self.margin_used_per_position_ac[slot_idx]
            projected_total_margin_used_ac = self.total_margin_used_ac - current_margin_for_slot_ac + projected_margin_required_ac
            # projected_commission_ac = abs(units_to_trade) * trade_price_qc * exchange_rate_qc_to_ac * self.commission_percentage # Already calculated in _execute_trade

            if units_to_trade.copy_sign(Decimal('1')) == projected_new_units.copy_sign(Decimal('1')) or abs(current_units) < Decimal('1e-9'):
                margin_safety_buffer = Decimal('0.1')
                safe_margin_threshold = Decimal(str(OANDA_MARGIN_CLOSEOUT_LEVEL)) + margin_safety_buffer
                
                if projected_total_margin_used_ac > self.equity_ac * (Decimal('1.0') - safe_margin_threshold):
                    logger.warning(f"Step {self.episode_step_count} Symbol {symbol}: Insufficient margin for {units_to_trade:.2f} units. Projected total margin {projected_total_margin_used_ac:.2f} AC, Equity {self.equity_ac:.2f} AC. Reducing trade size.")
                    max_affordable_margin_ac = self.equity_ac * (Decimal('1.0') - safe_margin_threshold)
                    if max_affordable_margin_ac <= Decimal('0'):
                        logger.warning(f"Step {self.episode_step_count} Symbol {symbol}: Cannot afford any margin, cancelling trade.")
                        continue
                    
                    max_units_by_margin_ac = (max_affordable_margin_ac / (trade_price_qc * Decimal(str(details.margin_rate)) * exchange_rate_qc_to_ac)).quantize(Decimal('1'), rounding=ROUND_DOWN)
                    
                    if abs(projected_new_units) > max_units_by_margin_ac:
                        if abs(current_units) < Decimal('1e-9'):
                            units_to_trade = max_units_by_margin_ac.copy_sign(units_to_trade)
                        else:
                            units_can_add = max_units_by_margin_ac - abs(current_units)
                            units_to_trade = units_can_add.copy_sign(units_to_trade)
                        
                        units_to_trade = details.round_units(units_to_trade)
                        if abs(units_to_trade) < details.minimum_trade_size:
                            logger.warning(f"Step {self.episode_step_count} Symbol {symbol}: Reduced units {units_to_trade:.2f} less than min trade size, cancelling trade.")
                            continue
                        logger.info(f"Step {self.episode_step_count} Symbol {symbol}: Trade size reduced to {units_to_trade:.2f} due to margin requirements.")
            logger.debug(f"Step {self.episode_step_count} Symbol {symbol}: Margin check took {time.perf_counter() - _t_margin_check_start:.6f}s")

            if abs(units_to_trade) > Decimal('0'):
                _t_execute_trade_start = time.perf_counter()
                traded_units, commission = self._execute_trade(slot_idx, units_to_trade, trade_price_qc, current_timestamp, all_prices_map)
                total_commission_this_step_ac += commission
                logger.debug(f"Step {self.episode_step_count} Symbol {symbol}: _execute_trade for {traded_units} units took {time.perf_counter() - _t_execute_trade_start:.6f}s")
            # else:
                # logger.debug(f"Step {self.episode_step_count} Slot {slot_idx} ({symbol}): No trade needed.")
            logger.debug(f"Step {self.episode_step_count} Symbol {symbol}: Action processing for slot took {time.perf_counter() - _t_slot_action_start:.6f}s")
        logger.debug(f"Step {self.episode_step_count}: Agent actions loop took {time.perf_counter() - _t_agent_actions_start:.6f}s")

        _t_update_portfolio_final = time.perf_counter()
        self.total_margin_used_ac = sum(self.margin_used_per_position_ac) # Recalculate after all trades in loop
        self._update_portfolio_and_equity_value(all_prices_map)
        self.portfolio_value_history.append(float(self.portfolio_value_ac))
        logger.debug(f"Step {self.episode_step_count}: Final _update_portfolio_and_equity_value took {time.perf_counter() - _t_update_portfolio_final:.6f}s")
        
        _t_calc_reward = time.perf_counter()
        reward = self._calculate_reward(prev_portfolio_value_ac, total_commission_this_step_ac)
        logger.debug(f"Step {self.episode_step_count}: _calculate_reward took {time.perf_counter() - _t_calc_reward:.6f}s")
        
        # Get the next observation FIRST
        _t_get_obs_early = time.perf_counter()
        next_observation = self._get_observation()
        logger.debug(f"Step {self.episode_step_count}: Early _get_observation took {time.perf_counter() - _t_get_obs_early:.6f}s")

        # Corrected check for NaN/Inf in observation components.
        for key, obs_component_value in next_observation.items():
            if isinstance(obs_component_value, np.ndarray) and np.issubdtype(obs_component_value.dtype, np.floating):
                if np.any(np.isnan(obs_component_value)) or np.any(np.isinf(obs_component_value)):
                    logger.error(f"NaN or Inf detected in floating-point observation component '{key}' within step method. Data sample (first 100 chars): {str(obs_component_value)[:100]}")
                    # Optional: Implement imputation here if desired
            # No explicit check for non-float/non-bool numpy arrays with np.isnan/isinf, as these functions
            # would typically raise an error or return False for integer arrays.
            # Boolean arrays don't carry NaN/Inf in the same way.

        # Check for termination or truncation
        self.current_step_in_dataset += 1
        
        _t_check_term = time.perf_counter()
        terminated, truncated = self._check_termination_truncation()
        logger.debug(f"Step {self.episode_step_count}: _check_termination_truncation took {time.perf_counter() - _t_check_term:.6f}s")
          # next_observation was already fetched, no need to call _get_observation() again here
        # _t_get_obs = time.perf_counter()
        # next_observation = self._get_observation() # THIS LINE IS REMOVED / COMMENTED OUT
        # logger.debug(f"Step {self.episode_step_count}: _get_observation took {time.perf_counter() - _t_get_obs:.6f}s")
        info = self._get_info()
        info["reward_this_step"] = reward
        logger.debug(f"--- Step {self.episode_step_count} End. Total time: {time.perf_counter() - _step_time_start:.6f}s ---")
        return next_observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, prev_portfolio_value_ac: Decimal, commission_this_step_ac: Decimal) -> float:
        """
        計算強化學習獎勵函數，支援傳統與增強版獎勵計算
        
        如果啟用增強版獎勵計算器，則使用更先進的獎勵機制；
        否則使用傳統的風險調整獎勵計算
        """
        
        # 使用增強版獎勵計算器（如果可用且啟用）
        if self.use_enhanced_rewards and self.enhanced_reward_calculator is not None:
            try:
                # 準備增強版獎勵計算所需的數據
                reward_data = {
                    'current_portfolio_value': self.portfolio_value_ac,
                    'previous_portfolio_value': prev_portfolio_value_ac,
                    'commission_this_step': commission_this_step_ac,
                    'returns_history': self.returns_history,
                    'unrealized_pnl_per_slot': self.unrealized_pnl_ac,
                    'current_positions': self.current_positions_units,
                    'last_trade_steps': self.last_trade_step_per_slot,
                    'current_step': self.episode_step_count,
                    'max_episode_steps': self.max_episode_steps,
                    'peak_portfolio_value': self.peak_portfolio_value_episode,
                    'max_drawdown': self.max_drawdown_episode,
                    'equity': self.equity_ac,
                    'total_margin_used': self.total_margin_used_ac,
                    'trade_log': self.trade_log,
                    'active_slot_indices': self.current_episode_tradable_slot_indices,
                    'atr_values': self.atr_values_qc,
                    'atr_penalty_threshold': self.atr_penalty_threshold
                }
                
                enhanced_reward = self.enhanced_reward_calculator.calculate_reward(reward_data)
                
                # 記錄增強版獎勵組件（如果有的話）
                if hasattr(self.enhanced_reward_calculator, 'last_reward_components'):
                    if not hasattr(self, 'enhanced_reward_components_history'):
                        self.enhanced_reward_components_history = []
                    
                    self.enhanced_reward_components_history.append({
                        'step': self.episode_step_count,
                        **self.enhanced_reward_calculator.last_reward_components
                    })
                
                return enhanced_reward
                
            except Exception as e:
                logger.error(f"Enhanced reward calculation failed: {e}. Falling back to standard calculation.")
                # 如果增強版計算失敗，回退到傳統計算
                self.use_enhanced_rewards = False
          # 傳統獎勵計算（改進版）
        return self._calculate_standard_reward(prev_portfolio_value_ac, commission_this_step_ac)
    
    def _calculate_standard_reward(self, prev_portfolio_value_ac: Decimal, commission_this_step_ac: Decimal) -> float:
        """
        傳統獎勵計算方法，包含一些改進的參數調整
        
        核心理念：
        1. 風險控制優先：重視風險調整後的收益而非絕對收益
        2. 穩定淨值增長：獎勵平穩的收益，懲罰過度波動
        3. 讓利潤奔跑，快速止損：實現交易原則
        4. 避免人為限制，讓模型探索更優策略
        """
        
        # === 1. 基礎收益計算 ===
        log_return = Decimal('0.0')
        if prev_portfolio_value_ac > Decimal('0'): 
            log_return = (self.portfolio_value_ac / prev_portfolio_value_ac).ln()
        
        # 更新收益歷史序列（用於風險調整計算）- 使用更大的窗口
        self.returns_history.append(log_return)
        enhanced_window_size = 50  # 增加窗口大小以提高穩定性
        if len(self.returns_history) > enhanced_window_size:
            self.returns_history.pop(0)
        
        # === 2. 風險調整後收益（改進的夏普比率） ===
        risk_adjusted_reward = Decimal('0.0')
        if len(self.returns_history) >= 10:  # 減少最小數據點要求
            returns_array = [float(r) for r in self.returns_history]
            mean_return = Decimal(str(sum(returns_array) / len(returns_array)))
            
            # 計算標準差
            variance = sum([(Decimal(str(r)) - mean_return) ** 2 for r in returns_array]) / Decimal(str(len(returns_array)))
            std_return = variance.sqrt() if variance > Decimal('0') else Decimal('1e-6')
            
            # 風險調整收益 = 平均收益 / 標準差（簡化夏普比率）
            risk_adjusted_return = mean_return / (std_return + Decimal('1e-6'))
            # 使用更積極的風險調整因子
            enhanced_risk_factor = Decimal('1.2')  # 從0.5提升到1.2
            risk_adjusted_reward = enhanced_risk_factor * risk_adjusted_return
        else:
            # 數據不足時使用基礎對數收益
            risk_adjusted_reward = self.reward_config["portfolio_log_return_factor"] * log_return
        
        reward_val = risk_adjusted_reward
        
        # === 3. 手續費懲罰（鼓勵減少過度交易） - 調整為動態懲罰 ===
        # 根據近期收益調整手續費懲罰強度
        recent_performance = Decimal('1.0')
        if len(self.returns_history) >= 5:
            recent_returns = self.returns_history[-5:]
            recent_performance = max(Decimal('0.5'), sum(recent_returns) / Decimal('5') + Decimal('1.0'))
        
        dynamic_commission_factor = self.reward_config["commission_penalty_factor"] / recent_performance
        commission_penalty = dynamic_commission_factor * (commission_this_step_ac / self.initial_capital)
        reward_val -= commission_penalty
        
        # === 4. 最大回撤懲罰（強化風險控制） - 調整懲罰強度 ===
        self.peak_portfolio_value_episode = max(self.peak_portfolio_value_episode, self.portfolio_value_ac)
        current_dd = (self.peak_portfolio_value_episode - self.portfolio_value_ac) / (self.peak_portfolio_value_episode + Decimal('1e-9'))
        
        enhanced_dd_factor = Decimal('1.5')  # 從2.0降低到1.5，減少過度懲罰
        if current_dd > self.max_drawdown_episode:
            # 新的最大回撤發生時給予較重懲罰
            dd_penalty = enhanced_dd_factor * (current_dd - self.max_drawdown_episode)
            reward_val -= dd_penalty
            self.max_drawdown_episode = current_dd
        elif current_dd > Decimal('0'):
            # 持續回撤時給予輕微懲罰
            reward_val -= enhanced_dd_factor * current_dd * Decimal('0.05')  # 從0.1降低到0.05
        
        # === 5. 持倉時間獎勵機制（讓利潤奔跑） - 增強獎勵 ===
        position_hold_reward = Decimal('0.0')
        for slot_idx in self.current_episode_tradable_slot_indices:
            units = self.current_positions_units[slot_idx]
            if abs(units) > Decimal('1e-9'):  # 有持倉
                unrealized_pnl = self.unrealized_pnl_ac[slot_idx]
                last_trade_step = self.last_trade_step_per_slot[slot_idx]
                
                if last_trade_step >= 0:
                    hold_duration = self.episode_step_count - last_trade_step
                    
                    # 如果持倉時間較長且盈利，給予獎勵（讓利潤奔跑）
                    if unrealized_pnl > Decimal('0') and hold_duration > 3:  # 降低持倉要求
                        duration_factor = min(Decimal(str(hold_duration)) / Decimal('15'), Decimal('3.0'))  # 調整因子
                        profit_ratio = unrealized_pnl / self.initial_capital
                        enhanced_profit_bonus = Decimal('0.3')  # 從0.1提升到0.3
                        position_hold_reward += enhanced_profit_bonus * profit_ratio * duration_factor
                    
                    # 如果持倉時間較短但虧損較大，輕微懲罰（快速止損相關）
                    elif unrealized_pnl < Decimal('0') and hold_duration > 8:  # 稍微降低止損要求
                        loss_ratio = abs(unrealized_pnl) / self.initial_capital
                        if loss_ratio > Decimal('0.003'):  # 從0.005降低到0.003
                            reduced_hold_penalty = Decimal('0.0005')  # 從0.001減半
                            position_hold_reward -= reduced_hold_penalty * loss_ratio
        
        reward_val += position_hold_reward
        
        # === 6. ATR波動性調整（控制過度風險） ===
        volatility_penalty = Decimal('0.0')
        active_positions = 0
        total_atr_ratio = Decimal('0.0')
        
        for slot_idx in self.current_episode_tradable_slot_indices:
            units = self.current_positions_units[slot_idx]
            if abs(units) > Decimal('1e-9'):
                active_positions += 1
                symbol = self.slot_to_symbol_map.get(slot_idx)
                if symbol:
                    atr_qc = self.atr_values_qc[slot_idx]
                    details = self.instrument_details_map[symbol]
                    avg_entry = self.avg_entry_prices_qc[slot_idx]
                    
                    if atr_qc > Decimal('0') and avg_entry > Decimal('0'):
                        atr_ratio = atr_qc / avg_entry  # ATR相對於入場價格的比例
                        total_atr_ratio += atr_ratio
        
        if active_positions > 0:
            avg_atr_ratio = total_atr_ratio / Decimal(str(active_positions))
            if avg_atr_ratio > self.atr_penalty_threshold:
                # 當平均ATR比例過高時（市場過於波動），給予懲罰
                volatility_penalty = enhanced_risk_factor * (avg_atr_ratio - self.atr_penalty_threshold) * Decimal('0.3')  # 從0.5降低到0.3
                reward_val -= volatility_penalty
        
        # === 7. 保證金追繳懲罰（強化風險管理） ===
        if self.total_margin_used_ac > Decimal('0'):
            margin_level = self.equity_ac / self.total_margin_used_ac
            margin_warning_level = Decimal(str(OANDA_MARGIN_CLOSEOUT_LEVEL)) * Decimal('1.2')  # 60%水平開始警告
            
            if margin_level < margin_warning_level:
                margin_risk_penalty = (margin_warning_level - margin_level) * Decimal('0.05')  # 從0.1降低到0.05
                reward_val -= margin_risk_penalty
        
        # === 記錄詳細信息用於監控 ===
        if not hasattr(self, 'reward_components_history'):
            self.reward_components_history = []
        
        self.reward_components_history.append({
            'step': self.episode_step_count,
            'risk_adjusted_reward': float(risk_adjusted_reward),
            'commission_penalty': float(commission_penalty),
            'position_hold_reward': float(position_hold_reward),
            'volatility_penalty': float(volatility_penalty),
            'total_reward': float(reward_val),
            'reward_type': 'standard_enhanced'
        })
        
        return float(reward_val)

    def _apply_stop_loss(self, all_prices_map: Dict[str, Tuple[Decimal, Decimal]], current_timestamp: pd.Timestamp) -> Decimal:
        """
        檢查並執行止損。
        返回因止損而產生的總手續費。
        """
        commission_from_sl = Decimal('0.0')
        for slot_idx in self.current_episode_tradable_slot_indices:
            units = self.current_positions_units[slot_idx]
            stop_loss_price_qc = self.stop_loss_prices_qc[slot_idx]
            symbol = self.slot_to_symbol_map.get(slot_idx)

            if symbol and abs(units) > Decimal('1e-9') and stop_loss_price_qc > Decimal('0'):
                current_bid_qc, current_ask_qc = all_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
                
                if current_bid_qc <= Decimal('0') or current_ask_qc <= Decimal('0'):
                    logger.warning(f"止損檢查: {symbol} 的當前價格無效。")
                    continue

                closed_by_sl = False
                trade_price_for_sl = Decimal('0.0')

                if units > 0: # 多頭倉位
                    if current_bid_qc <= stop_loss_price_qc:
                        closed_by_sl = True
                        trade_price_for_sl = current_bid_qc # 賣出平倉用 Bid
                elif units < 0: # 空頭倉位
                    if current_ask_qc >= stop_loss_price_qc:
                        closed_by_sl = True
                        trade_price_for_sl = current_ask_qc # 買入平倉用 Ask
                
                if closed_by_sl and trade_price_for_sl > Decimal('0'):
                    logger.info(f"止損觸發 for {symbol} at step {self.episode_step_count}. 價格: {trade_price_for_sl:.5f} QC, 止損價: {stop_loss_price_qc:.5f} QC.")
                    # 調用 _execute_trade 進行平倉
                    # units_to_trade 應該是 -current_units (反向平倉所有單位)
                    traded_units, commission = self._execute_trade(slot_idx, -units, trade_price_for_sl, current_timestamp, all_prices_map)
                    commission_from_sl += commission
                    # 止損後，該倉位的保證金會被釋放，_execute_trade 會處理
                    # 平均入場價也會被重置為 0，_execute_trade 會處理
        return commission_from_sl

    def _handle_margin_call(self, all_prices_map: Dict[str, Tuple[Decimal, Decimal]], current_timestamp: pd.Timestamp) -> Decimal:
        """
        處理保證金追繳。如果保證金水平低於 OANDA_MARGIN_CLOSEOUT_LEVEL，則強制平倉。
        返回因強平而產生的總手續費。
        """
        commission_from_mc = Decimal('0.0')
        oanda_closeout_level_decimal = Decimal(str(OANDA_MARGIN_CLOSEOUT_LEVEL))
        
        # 重新計算總保證金使用量和權益，因為止損可能已經改變了它們
        self._update_portfolio_and_equity_value(all_prices_map)
        self.total_margin_used_ac = sum(self.margin_used_per_position_ac)

        if self.total_margin_used_ac <= Decimal('0'):
            return Decimal('0.0') # 沒有保證金使用，無需處理強平

        margin_level = self.equity_ac / self.total_margin_used_ac

        if margin_level < oanda_closeout_level_decimal:
            logger.warning(f"強制平倉觸發! Equity={self.equity_ac:.2f}, MarginUsed={self.total_margin_used_ac:.2f}, Level={margin_level:.2%}. 開始平倉。")
            
            # 獲取所有持倉，按未實現虧損排序 (虧損最大的先平)
            positions_to_close = []
            for slot_idx in self.current_episode_tradable_slot_indices:
                units = self.current_positions_units[slot_idx]
                if abs(units) > Decimal('1e-9'):
                    symbol = self.slot_to_symbol_map[slot_idx]
                    unrealized_pnl = self.unrealized_pnl_ac[slot_idx] # 這是負數表示虧損
                    positions_to_close.append((unrealized_pnl, slot_idx, symbol, units))
            
            # 按未實現盈虧升序排序 (虧損最大的在前面)
            positions_to_close.sort(key=lambda x: x[0])

            for pnl, slot_idx, symbol, units_to_close in positions_to_close:
                # 重新檢查保證金水平，如果已經恢復則停止平倉
                self._update_portfolio_and_equity_value(all_prices_map)
                self.total_margin_used_ac = sum(self.margin_used_per_position_ac)
                if self.total_margin_used_ac > Decimal('0') and (self.equity_ac / self.total_margin_used_ac) >= oanda_closeout_level_decimal:
                    logger.info(f"保證金水平已恢復 ({self.equity_ac / self.total_margin_used_ac:.2%})，停止強平。")
                    break
                
                # 確定平倉價格
                current_bid_qc, current_ask_qc = all_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
                trade_price_for_mc = Decimal('0.0')
                if units_to_close > 0: # 多頭倉位，賣出平倉
                    trade_price_for_mc = current_bid_qc
                elif units_to_close < 0: # 空頭倉位，買入平倉
                    trade_price_for_mc = current_ask_qc
                
                if trade_price_for_mc <= Decimal('0'):
                    logger.warning(f"強平: {symbol} 的交易價格無效，無法平倉。")
                    continue

                logger.info(f"強平 {symbol}: 平倉 {units_to_close:.2f} 單位，價格 {trade_price_for_mc:.5f} QC。")
                traded_units, commission = self._execute_trade(slot_idx, -units_to_close, trade_price_for_mc, current_timestamp, all_prices_map)
                commission_from_mc += commission
                
                # 每次平倉後更新權益和保證金使用情況
                self._update_portfolio_and_equity_value(all_prices_map)
                self.total_margin_used_ac = sum(self.margin_used_per_position_ac)
                
                # 如果所有倉位都平了，也停止
                if all(abs(u) < Decimal('1e-9') for u in self.current_positions_units):
                    logger.info("所有倉位已平倉，停止強平。")
                    break
            
            # 如果強平後保證金水平仍未恢復，則可能需要額外處理 (例如，直接終止 episode)
            self._update_portfolio_and_equity_value(all_prices_map)
            self.total_margin_used_ac = sum(self.margin_used_per_position_ac)
            if self.total_margin_used_ac > Decimal('0') and (self.equity_ac / self.total_margin_used_ac) < oanda_closeout_level_decimal:
                logger.error(f"強平後保證金水平仍未恢復! Equity={self.equity_ac:.2f}, MarginUsed={self.total_margin_used_ac:.2f}, Level={self.equity_ac / self.total_margin_used_ac:.2%}. Episode 將終止。")
                # 環境將在 _check_termination_truncation 中被標記為 terminated
            
            # 懲罰：強平會導致一個大的負獎勵
            # 這裡不直接返回懲罰，而是讓 _check_termination_truncation 觸發 terminated，然後在 _calculate_reward 中處理
            # 但可以在 info 中標記強平事件
            # self.reward_config["margin_call_penalty"] 可以在 _calculate_reward 中使用
        return commission_from_mc

    def _check_termination_truncation(self) -> Tuple[bool, bool]:
        terminated = False
        oanda_closeout_level_decimal = Decimal(str(OANDA_MARGIN_CLOSEOUT_LEVEL))

        # 檢查權益是否過低
        if self.portfolio_value_ac < self.initial_capital * oanda_closeout_level_decimal * Decimal('0.4'):
            logger.warning(f"Episode terminated: Portfolio value ({self.portfolio_value_ac:.2f}) too low."); terminated = True
        
        # 檢查保證金水平是否觸發強平
        self.total_margin_used_ac = sum(self.margin_used_per_position_ac)
        if self.total_margin_used_ac > Decimal('0'):
            margin_level = self.equity_ac / self.total_margin_used_ac
            if margin_level < oanda_closeout_level_decimal:
                logger.warning(f"強制平倉觸發! Equity={self.equity_ac:.2f}, MarginUsed={self.total_margin_used_ac:.2f}, Level={margin_level:.2%}. Episode 將終止。")
                terminated = True
        
        truncated = self.episode_step_count >= self.max_episode_steps
        if self.current_step_in_dataset >= len(self.dataset):
            truncated = True
            terminated = True if not terminated else True # 如果已經終止，保持終止狀態
        
        return terminated, truncated

    def _get_observation(self) -> Dict[str, np.ndarray]:
        dataset_sample = self.dataset[min(self.current_step_in_dataset, len(self.dataset)-1)]
        features_raw = dataset_sample["features"].numpy()
        obs_f = np.zeros((self.num_env_slots, self.dataset.timesteps_history, self.dataset.num_features_per_symbol), dtype=np.float32)
        obs_pr = np.zeros(self.num_env_slots, dtype=np.float32); obs_upl_r = np.zeros(self.num_env_slots, dtype=np.float32)
        obs_tslt_ratio = np.zeros(self.num_env_slots, dtype=np.float32); obs_pm = np.ones(self.num_env_slots, dtype=np.bool_)
        obs_volatility = np.zeros(self.num_env_slots, dtype=np.float32)   # 新增波動率特徵
        current_prices_map, _ = self._get_current_raw_prices_for_all_dataset_symbols()
        for slot_idx in range(self.num_env_slots):
            symbol = self.slot_to_symbol_map.get(slot_idx)
            if symbol:
                if symbol in self.dataset.symbols:
                    try: dataset_symbol_idx = self.dataset.symbols.index(symbol)
                    except ValueError: logger.error(f"Symbol {symbol} in slot_map but not in dataset.symbols for observation."); continue
                    feature_slice = features_raw[dataset_symbol_idx, :, :]
                    expected_shape = (self.dataset.timesteps_history, self.dataset.num_features_per_symbol)
                    if feature_slice.shape != expected_shape:
                        logger.warning(f"維度不匹配修復: Symbol {symbol} 特徵維度 {feature_slice.shape} != 預期 {expected_shape}")
                        if feature_slice.shape[0] != self.dataset.timesteps_history:
                            if feature_slice.shape[0] > self.dataset.timesteps_history:
                                feature_slice = feature_slice[-self.dataset.timesteps_history:, :]
                            else:
                                padded_slice = np.zeros(expected_shape, dtype=np.float32)
                                padded_slice[-feature_slice.shape[0]:, :] = feature_slice
                                feature_slice = padded_slice
                        if feature_slice.shape[1] != self.dataset.num_features_per_symbol:
                            if feature_slice.shape[1] > self.dataset.num_features_per_symbol:
                                feature_slice = feature_slice[:, :self.dataset.num_features_per_symbol]
                            else:
                                padded_slice = np.zeros(expected_shape, dtype=np.float32)
                                padded_slice[:, :feature_slice.shape[1]] = feature_slice
                                feature_slice = padded_slice
                    obs_f[slot_idx, :, :] = feature_slice
                units = self.current_positions_units[slot_idx]; details = self.instrument_details_map[symbol]
                current_bid_qc, current_ask_qc = current_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
                # 新增波動率計算
                current_mid_price = (current_bid_qc + current_ask_qc) / Decimal('2')
                atr_value = self.atr_values_qc[slot_idx]
                if current_mid_price > Decimal('0') and atr_value > Decimal('0'):
                    rel_volatility = atr_value / current_mid_price
                    normalized_vol = min(1.0, float(rel_volatility / Decimal('0.1')))   # 假設0.1是最大波動閾值
                else:
                    normalized_vol = 0.0
                obs_volatility[slot_idx] = normalized_vol
                # 原特徵計算
                price_for_value_calc_qc = (current_bid_qc + current_ask_qc) / Decimal('2') if current_bid_qc > 0 and current_ask_qc > 0 else Decimal('0.0')
                nominal_value_qc = abs(units) * price_for_value_calc_qc; nominal_value_ac = nominal_value_qc
                if details.quote_currency != ACCOUNT_CURRENCY:
                    rate = self._get_exchange_rate_to_account_currency(details.quote_currency, current_prices_map)
                    if rate > 0: nominal_value_ac *= rate
                    else: nominal_value_ac = Decimal('0.0')
                obs_pr[slot_idx] = float( (nominal_value_ac / self.initial_capital) * units.copy_sign(Decimal('1')) )
                obs_upl_r[slot_idx] = float(self.unrealized_pnl_ac[slot_idx] / self.initial_capital)
                obs_pm[slot_idx] = False
                if self.last_trade_step_per_slot[slot_idx] == -1 or self.episode_step_count == 0 : obs_tslt_ratio[slot_idx] = 1.0
                else: steps_since_last = self.episode_step_count - self.last_trade_step_per_slot[slot_idx]; obs_tslt_ratio[slot_idx] = min(1.0, steps_since_last / (self.max_episode_steps / 10.0 if self.max_episode_steps > 0 else 100.0) )
            else:
                obs_volatility[slot_idx] = 0.0   # 無交易品種，波動率設為0
        margin_level_val = float(self.equity_ac / (self.total_margin_used_ac + Decimal('1e-9')))
        return {"features_from_dataset": obs_f, "current_positions_nominal_ratio_ac": np.clip(obs_pr, -5.0, 5.0).astype(np.float32), "unrealized_pnl_ratio_ac": np.clip(obs_upl_r, -1.0, 5.0).astype(np.float32), "margin_level": np.clip(np.array([margin_level_val]), 0.0, 100.0).astype(np.float32), "time_since_last_trade_ratio": obs_tslt_ratio.astype(np.float32), "volatility": obs_volatility.astype(np.float32), "padding_mask": obs_pm}   # 添加波動率特徵

    def _init_render_figure(self):
        """初始化matplotlib圖表用於渲染"""
        try:
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
            self.fig.suptitle('交易環境監控', fontsize=14)
            
            # 上圖：投資組合價值
            self.ax1.set_title('投資組合價值歷史')
            self.ax1.set_ylabel(f'價值 ({ACCOUNT_CURRENCY})')
            self.ax1.grid(True, alpha=0.3)
            
            # 下圖：持倉狀況
            self.ax2.set_title('當前持倉狀況')
            self.ax2.set_ylabel('持倉單位')
            self.ax2.set_xlabel('交易對象')
            self.ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            logger.info("渲染圖表初始化完成")
        except Exception as e:
            logger.warning(f"渲染圖表初始化失敗: {e}")
            self.fig = None
            self.ax1 = None
            self.ax2 = None

    def render(self):
        """渲染當前環境狀態"""
        if self.render_mode != 'human' or not hasattr(self, 'fig') or self.fig is None:
            return
        
        try:
            # 清除之前的圖表內容
            self.ax1.clear()
            self.ax2.clear()
            
            # 重新設置標題和標籤
            self.ax1.set_title('投資組合價值歷史')
            self.ax1.set_ylabel(f'價值 ({ACCOUNT_CURRENCY})')
            self.ax1.grid(True, alpha=0.3)
            
            # 繪製投資組合價值歷史
            if len(self.portfolio_value_history) > 1:
                steps = list(range(len(self.portfolio_value_history)))
                self.ax1.plot(steps, self.portfolio_value_history, 'b-', linewidth=2, label='投資組合價值')
                self.ax1.axhline(y=float(self.initial_capital), color='r', linestyle='--', alpha=0.7, label='初始資本')
                self.ax1.legend()
            
            # 繪製當前持倉狀況
            self.ax2.set_title('當前持倉狀況')
            self.ax2.set_ylabel('持倉單位')
            self.ax2.set_xlabel('交易對象')
            self.ax2.grid(True, alpha=0.3)
            
            # 收集有效持倉數據
            symbols = []
            positions = []
            colors = []
            
            for slot_idx in self.current_episode_tradable_slot_indices:
                symbol = self.slot_to_symbol_map.get(slot_idx)
                if symbol:
                    units = float(self.current_positions_units[slot_idx])
                    if abs(units) > 1e-9:  # 只顯示有意義的持倉
                        symbols.append(symbol)
                        positions.append(units)
                        colors.append('green' if units > 0 else 'red')
            
            if symbols:
                bars = self.ax2.bar(symbols, positions, color=colors, alpha=0.7)
                self.ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                # 添加數值標籤
                for bar, pos in zip(bars, positions):
                    height = bar.get_height()
                    self.ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{pos:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
            else:
                self.ax2.text(0.5, 0.5, '無持倉', ha='center', va='center', transform=self.ax2.transAxes)
            
            # 添加環境信息
            info_text = f"步驟: {self.episode_step_count}\n"
            info_text += f"現金: {float(self.cash):.2f} {ACCOUNT_CURRENCY}\n"
            info_text += f"權益: {float(self.equity_ac):.2f} {ACCOUNT_CURRENCY}\n"
            info_text += f"已用保證金: {float(self.total_margin_used_ac):.2f} {ACCOUNT_CURRENCY}"
            
            self.fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.pause(0.01)  # 短暫暫停以更新顯示
            
        except Exception as e:
            logger.warning(f"渲染過程中發生錯誤: {e}")
    def close(self):
        """關閉環境並清理資源"""
        logger.info("關閉 TradingEnvV4。")
        try:
            if hasattr(self, 'fig') and self.fig is not None:
                plt.close(self.fig)
                self.fig = None
                self.ax1 = None
                self.ax2 = None
                logger.debug("已關閉matplotlib圖表")
        except Exception as e:
            logger.warning(f"關閉圖表時發生錯誤: {e}")

    def _get_info(self) -> Dict[str, Any]:
        """獲取環境信息"""
        return {
            "cash_ac": float(self.cash),
            "portfolio_value_ac": float(self.portfolio_value_ac),
            "equity_ac": float(self.equity_ac),
            "total_margin_used_ac": float(self.total_margin_used_ac),
            "episode_step": self.episode_step_count,
            "max_drawdown": float(self.max_drawdown_episode),
            "peak_portfolio_value": float(self.peak_portfolio_value_episode),
            "active_positions": sum(1 for units in self.current_positions_units if abs(units) > Decimal('1e-9')),
            "trade_count": len(self.trade_log)
        }
    
    def get_current_info(self) -> Dict[str, Any]:
        """
        獲取當前環境的詳細信息，供訓練監控使用
        
        Returns:
            包含當前狀態詳細信息的字典
        """
        info = {
            'episode_reward': float(sum(self.reward_history)) if self.reward_history else 0.0,
            'portfolio_value_ac': float(self.portfolio_value_ac),
            'equity_ac': float(self.equity_ac),
            'cash_ac': float(self.cash),
            'total_margin_used_ac': float(self.total_margin_used_ac),
            'episode_step': self.episode_step_count,
            'max_drawdown': float(self.max_drawdown_episode),
            'peak_portfolio_value': float(self.peak_portfolio_value_episode),
            'active_positions': sum(1 for units in self.current_positions_units if abs(units) > Decimal('1e-9')),
            'trade_count': len(self.trade_log),
            'symbol_stats': {}
        }
        
        # 計算每個symbol的統計信息
        symbol_trades = {}
        for trade in self.trade_log:
            symbol = trade['symbol']
            if symbol not in symbol_trades:
                symbol_trades[symbol] = {
                    'trades': [],
                    'pnl': []
                }
            symbol_trades[symbol]['trades'].append(trade)
            if trade['realized_pnl_ac'] != 0:
                symbol_trades[symbol]['pnl'].append(trade['realized_pnl_ac'])
        
        # 計算統計指標
        for symbol, data in symbol_trades.items():
            trades = data['trades']
            pnl_list = data['pnl']
            
            stats = {
                'trades': len(trades),
                'win_rate': 0,
                'avg_return': 0,
                'max_return': 0,
                'max_loss': 0,
                'sharpe_ratio': 0,
                'returns': []
            }
            
            if pnl_list:
                wins = sum(1 for p in pnl_list if p > 0)
                stats['win_rate'] = (wins / len(pnl_list)) * 100 if pnl_list else 0
                
                # 計算收益率
                returns = []
                for i, trade in enumerate(trades):
                    if trade['trade_type'] in ['CLOSE', 'REDUCE', 'CLOSE_AND_REVERSE']:
                        # 計算收益率
                        if trade['avg_entry_price_qc'] > 0:
                            if trade['position_direction'] == 'LONG':
                                ret = ((trade['trade_price_qc'] - trade['avg_entry_price_qc']) / trade['avg_entry_price_qc']) * 100
                            else:
                                ret = ((trade['avg_entry_price_qc'] - trade['trade_price_qc']) / trade['avg_entry_price_qc']) * 100
                            returns.append(ret)
                
                if returns:
                    stats['returns'] = returns
                    stats['avg_return'] = np.mean(returns)
                    stats['max_return'] = max(returns)
                    stats['max_loss'] = min(returns)
                    
                    # 簡化的夏普比率計算
                    if len(returns) > 1:
                        returns_std = np.std(returns)
                        if returns_std > 0:
                            stats['sharpe_ratio'] = stats['avg_return'] / returns_std
            
            info['symbol_stats'][symbol] = stats
        
        return info

# --- if __name__ == "__main__": 測試塊 (與V4.8版本相同) ---
if __name__ == "__main__":
    # ... (與您上一個版本 UniversalTradingEnvV4.8 __main__ 測試塊相同的代碼) ...
    logger.info("正在直接運行 UniversalTradingEnvV4.py 進行測試...")
    if 'OANDA_API_KEY' not in globals() or globals().get('OANDA_API_KEY') is None: logger.error("OANDA_API_KEY 未配置。"); sys.exit(1)
    if 'format_datetime_for_oanda' not in globals() or globals().get('format_datetime_for_oanda') is None: logger.error("format_datetime_for_oanda 未定義。"); sys.exit(1)
    if 'manage_data_download_for_symbols' not in globals() or globals().get('manage_data_download_for_symbols') is None: logger.error("manage_data_download_for_symbols 未定義。"); sys.exit(1)
    if 'UniversalMemoryMappedDataset' not in globals() or globals().get('UniversalMemoryMappedDataset') is None: logger.error("UniversalMemoryMappedDataset 未定義。"); sys.exit(1)
    if 'InstrumentInfoManager' not in globals() or globals().get('InstrumentInfoManager') is None: logger.error("InstrumentInfoManager 未定義。"); sys.exit(1)
    logger.info("MMap數據集和InstrumentInfo準備...")
    test_symbols_list_main = ["EUR_USD", "USD_JPY", "AUD_USD"]
    try:
        test_start_datetime_main = datetime(2024, 5, 22, 10, 0, 0, tzinfo=timezone.utc)
        test_end_datetime_main = datetime(2024, 5, 22, 11, 0, 0, tzinfo=timezone.utc)
    except ValueError as e_date_main: logger.error(f"測試用的固定日期時間無效: {e_date_main}", exc_info=True); sys.exit(1)
    if test_start_datetime_main >= test_end_datetime_main: logger.error("測試時間範圍無效：開始時間必須早於結束時間。"); sys.exit(1)
    test_start_iso_str_main = format_datetime_for_oanda(test_start_datetime_main)
    test_end_iso_str_main = format_datetime_for_oanda(test_end_datetime_main)
    test_granularity_val_main = "S5"; test_timesteps_history_val_main = TIMESTEPS
    logger.info(f"測試參數: symbols={test_symbols_list_main}, start={test_start_iso_str_main}, end={test_end_iso_str_main}, granularity={test_granularity_val_main}, history_len={test_timesteps_history_val_main}")
    logger.info("確保數據庫中有測試時間段的數據 (如果沒有則下載)...")
    manage_data_download_for_symbols(symbols=test_symbols_list_main, overall_start_str=test_start_iso_str_main, overall_end_str=test_end_iso_str_main, granularity=test_granularity_val_main)
    logger.info("數據庫數據準備完成/已檢查。")
    test_dataset_main = UniversalMemoryMappedDataset(symbols=test_symbols_list_main, start_time_iso=test_start_iso_str_main, end_time_iso=test_end_iso_str_main, granularity=test_granularity_val_main, timesteps_history=test_timesteps_history_val_main, force_reload=False)
    if len(test_dataset_main) == 0: logger.error("測試數據集為空!"); sys.exit(1)
    instrument_manager_main = InstrumentInfoManager(force_refresh=False)
    active_episode_symbols_main = ["EUR_USD", "USD_JPY"]
    account_currency_upper_main = ACCOUNT_CURRENCY.upper()
    symbols_needed_for_details = list(set(active_episode_symbols_main + [sym for sym in test_symbols_list_main if account_currency_upper_main in sym.upper().split("_") or "USD" in sym.upper().split("_") or sym == f"{account_currency_upper_main}_USD" or sym == f"USD_{account_currency_upper_main}"]))
    logger.info(f"為環境準備InstrumentDetails的Symbols列表: {symbols_needed_for_details}")
    logger.info("創建測試環境實例 (UniversalTradingEnvV4)...")
    env_main = UniversalTradingEnvV4(dataset=test_dataset_main, instrument_info_manager=instrument_manager_main, active_symbols_for_episode=active_episode_symbols_main)
    logger.info("重置環境...")
    obs_main, info_reset_main = env_main.reset()
    logger.info(f"初始觀察 keys: {list(obs_main.keys())}"); logger.info(f"初始信息: {info_reset_main}")
    logger.info("\n執行一個隨機動作步驟...")
    action_main_raw = env_main.action_space.sample()
    action_to_apply_main = np.zeros_like(action_main_raw)
    for i_main_slot_idx in env_main.current_episode_tradable_slot_indices:
        action_to_apply_main[i_main_slot_idx] = action_main_raw[i_main_slot_idx]
    logger.info(f"執行動作 (僅對活躍槽位，基於槽位索引): {action_to_apply_main.round(3)}")
    obs_main_step, reward_main, terminated_main, truncated_main, info_main = env_main.step(action_to_apply_main)
    logger.info(f"  獎勵: {reward_main:.4f}, 終止: {terminated_main}, 截斷: {truncated_main}"); logger.info(f"  信息: {info_main}")
    env_main.close(); test_dataset_main.close()
    logger.info("UniversalTradingEnvV4.py 測試執行完畢。")