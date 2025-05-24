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
from pathlib import Path
import logging

getcontext().prec = 30
logger: logging.Logger = logging.getLogger("trading_env_module_init") # type: ignore
# ... (logger 初始化和 try-except import 塊與 V4.9 版本相同，這裡省略以節省篇幅) ...
_logger_initialized_by_common_env_v5 = False
try:
    from common.logger_setup import logger as common_configured_logger; logger = common_configured_logger; _logger_initialized_by_common_env_v5 = True
    logger.debug("trading_env.py (V5.0): Successfully imported logger from common.logger_setup.")
    from common.config import (TIMESTEPS as _TIMESTEPS, MAX_SYMBOLS_ALLOWED as _MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY as _ACCOUNT_CURRENCY, INITIAL_CAPITAL as _DEFAULT_INITIAL_CAPITAL, OANDA_MARGIN_CLOSEOUT_LEVEL as _OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE as _TRADE_COMMISSION_PERCENTAGE, OANDA_API_KEY as _OANDA_API_KEY, ATR_PERIOD as _ATR_PERIOD, STOP_LOSS_ATR_MULTIPLIER as _STOP_LOSS_ATR_MULTIPLIER, MAX_ACCOUNT_RISK_PERCENTAGE as _MAX_ACCOUNT_RISK_PERCENTAGE)
    _config_values_env_v5 = {"TIMESTEPS": _TIMESTEPS, "MAX_SYMBOLS_ALLOWED": _MAX_SYMBOLS_ALLOWED, "ACCOUNT_CURRENCY": _ACCOUNT_CURRENCY, "DEFAULT_INITIAL_CAPITAL": _DEFAULT_INITIAL_CAPITAL, "OANDA_MARGIN_CLOSEOUT_LEVEL": _OANDA_MARGIN_CLOSEOUT_LEVEL, "TRADE_COMMISSION_PERCENTAGE": _TRADE_COMMISSION_PERCENTAGE, "OANDA_API_KEY": _OANDA_API_KEY, "ATR_PERIOD": _ATR_PERIOD, "STOP_LOSS_ATR_MULTIPLIER": _STOP_LOSS_ATR_MULTIPLIER, "MAX_ACCOUNT_RISK_PERCENTAGE": _MAX_ACCOUNT_RISK_PERCENTAGE}
    logger.info("trading_env.py (V5.0): Successfully imported and stored common.config values.")
    from data_manager.mmap_dataset import UniversalMemoryMappedDataset; from data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols; from data_manager.instrument_info_manager import InstrumentDetails, InstrumentInfoManager; logger.info("trading_env.py (V5.0): Successfully imported other dependencies.")
except ImportError as e_initial_import_v5:
    logger_temp_v5 = logging.getLogger("trading_env_v5_fallback_initial"); logger_temp_v5.addHandler(logging.StreamHandler(sys.stdout)); logger_temp_v5.setLevel(logging.DEBUG); logger = logger_temp_v5
    logger.warning(f"trading_env.py (V5.0): Initial import failed: {e_initial_import_v5}. Attempting path adjustment...")
    project_root_env_v5 = Path(__file__).resolve().parent.parent.parent
    if str(project_root_env_v5) not in sys.path: sys.path.insert(0, str(project_root_env_v5)); logger.info(f"trading_env.py (V5.0): Added project root to sys.path: {project_root_env_v5}")
    try:
        from src.common.logger_setup import logger as common_logger_retry_v5; logger = common_logger_retry_v5; _logger_initialized_by_common_env_v5 = True; logger.info("trading_env.py (V5.0): Successfully re-imported common_logger after path adj.")
        from src.common.config import (TIMESTEPS as _TIMESTEPS_R, MAX_SYMBOLS_ALLOWED as _MAX_SYMBOLS_ALLOWED_R, ACCOUNT_CURRENCY as _ACCOUNT_CURRENCY_R, INITIAL_CAPITAL as _DEFAULT_INITIAL_CAPITAL_R, OANDA_MARGIN_CLOSEOUT_LEVEL as _OANDA_MARGIN_CLOSEOUT_LEVEL_R, TRADE_COMMISSION_PERCENTAGE as _TRADE_COMMISSION_PERCENTAGE_R, OANDA_API_KEY as _OANDA_API_KEY_R, ATR_PERIOD as _ATR_PERIOD_R, STOP_LOSS_ATR_MULTIPLIER as _STOP_LOSS_ATR_MULTIPLIER_R, MAX_ACCOUNT_RISK_PERCENTAGE as _MAX_ACCOUNT_RISK_PERCENTAGE_R)
        _config_values_env_v5 = {"TIMESTEPS": _TIMESTEPS_R, "MAX_SYMBOLS_ALLOWED": _MAX_SYMBOLS_ALLOWED_R, "ACCOUNT_CURRENCY": _ACCOUNT_CURRENCY_R, "DEFAULT_INITIAL_CAPITAL": _DEFAULT_INITIAL_CAPITAL_R, "OANDA_MARGIN_CLOSEOUT_LEVEL": _OANDA_MARGIN_CLOSEOUT_LEVEL_R, "TRADE_COMMISSION_PERCENTAGE": _TRADE_COMMISSION_PERCENTAGE_R, "OANDA_API_KEY": _OANDA_API_KEY_R, "ATR_PERIOD": _ATR_PERIOD_R, "STOP_LOSS_ATR_MULTIPLIER": _STOP_LOSS_ATR_MULTIPLIER_R, "MAX_ACCOUNT_RISK_PERCENTAGE": _MAX_ACCOUNT_RISK_PERCENTAGE_R}
        logger.info("trading_env.py (V5.0): Successfully re-imported and stored common.config after path adjustment.")
        from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset; from src.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols; from src.data_manager.instrument_info_manager import InstrumentDetails, InstrumentInfoManager; logger.info("trading_env.py (V5.0): Successfully re-imported other dependencies after path adjustment.")
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
    # ... ( __init__, reset, _get_current_raw_prices_for_all_dataset_symbols, 
    #       _get_specific_rate, _get_exchange_rate_to_account_currency,
    #       _update_atr_values, _update_stop_loss_prices, _update_portfolio_and_equity_value,
    #       _get_observation, _get_info, _init_render_figure, render, close
    #       這些方法的實現與您上一個版本 V4.9 的保持一致) ...
    # <在此處粘貼您上一個版本 UniversalTradingEnvV4 (V4.9) 中這些方法的完整實現>
    # 我將重新粘貼它們以確保完整性。
    metadata = {'render_modes': ['human', 'array'], 'render_fps': 10}
    def __init__(self, dataset: UniversalMemoryMappedDataset, instrument_info_manager: InstrumentInfoManager, active_symbols_for_episode: List[str], # type: ignore
                 initial_capital: float = float(DEFAULT_INITIAL_CAPITAL), max_episode_steps: Optional[int] = None,
                 commission_percentage_override: Optional[float] = None, reward_config: Optional[Dict[str, Union[float, Decimal]]] = None,
                 max_account_risk_per_trade: float = float(MAX_ACCOUNT_RISK_PERCENTAGE),
                 stop_loss_atr_multiplier: float = float(STOP_LOSS_ATR_MULTIPLIER),
                 atr_period: int = ATR_PERIOD, render_mode: Optional[str] = None):
        super().__init__(); self.dataset = dataset; self.instrument_info_manager = instrument_info_manager
        self.initial_capital = Decimal(str(initial_capital))
        if commission_percentage_override is not None: self.commission_percentage = Decimal(str(commission_percentage_override))
        else: self.commission_percentage = Decimal(str(TRADE_COMMISSION_PERCENTAGE))
        self.render_mode = render_mode; self.max_account_risk_per_trade = Decimal(str(max_account_risk_per_trade))
        self.stop_loss_atr_multiplier = Decimal(str(stop_loss_atr_multiplier)); self.atr_period = atr_period
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
        logger.info(f"環境初始化: {self.num_tradable_symbols_this_episode} 個可交易對象映射到 {self.num_env_slots} 個槽位。")
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
        self.atr_penalty_threshold: Decimal = Decimal('0.02')  # ATR懲罰閾值（2%）</search>
# 獎勵配置

        default_reward_config_decimal = {"portfolio_log_return_factor": Decimal('1.0'), "risk_adjusted_return_factor": Decimal('0.5'), "max_drawdown_penalty_factor": Decimal('2.0'), "commission_penalty_factor": Decimal('1.0'), "margin_call_penalty": Decimal('-100.0'), "profit_target_bonus": Decimal('0.1'), "hold_penalty_factor": Decimal('0.001')}
        if reward_config:
            for key, value in reward_config.items():
                if key in default_reward_config_decimal: default_reward_config_decimal[key] = Decimal(str(value))
        self.reward_config = default_reward_config_decimal
        self.peak_portfolio_value_episode: Decimal = self.initial_capital; self.max_drawdown_episode: Decimal = Decimal('0.0')
        obs_spaces = {
            "features_from_dataset": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_env_slots, TIMESTEPS, self.dataset.num_features_per_symbol), dtype=np.float32),
            "current_positions_nominal_ratio_ac": spaces.Box(low=-5.0, high=5.0, shape=(self.num_env_slots,), dtype=np.float32),
            "unrealized_pnl_ratio_ac": spaces.Box(low=-1.0, high=5.0, shape=(self.num_env_slots,), dtype=np.float32),
            "margin_level": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            "time_since_last_trade_ratio": spaces.Box(low=0.0, high=1.0, shape=(self.num_env_slots,), dtype=np.float32),
            "padding_mask": spaces.Box(low=0, high=1, shape=(self.num_env_slots,), dtype=np.bool_)}
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_env_slots,), dtype=np.float32)
        if self.render_mode == 'human': self._init_render_figure()
        logger.info(f"UniversalTradingEnvV4 (Detailed Step - V5.0) 初始化完成。")

    # --- (reset, _get_current_raw_prices_for_all_dataset_symbols, _get_specific_rate, _get_exchange_rate_to_account_currency,
    #      _update_atr_values, _update_stop_loss_prices, _update_portfolio_and_equity_value,
    #      _get_observation, _get_info, _init_render_figure, render, close 與 V4.9 基本相同) ---
    # <在此處粘貼您上一個版本 UniversalTradingEnvV4 (V4.9) 中這些方法的完整實現>
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
# 重置收益歷史
        self.returns_history = []
        self.peak_portfolio_value_episode = self.initial_capital; self.max_drawdown_episode = Decimal('0.0')
        logger.debug(f"Env reset. Initial capital: {self.cash} {ACCOUNT_CURRENCY}. Start step: {self.current_step_in_dataset}")
        all_prices_map, _ = self._get_current_raw_prices_for_all_dataset_symbols()
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
        from_currency_upper = from_currency.upper(); account_currency_upper = ACCOUNT_CURRENCY.upper()
        if from_currency_upper == account_currency_upper: return Decimal('1.0')
        direct_rate = self._get_specific_rate(from_currency_upper, account_currency_upper, current_prices_map)
        if direct_rate is not None and direct_rate > 0 : return direct_rate
        if from_currency_upper != "USD" and account_currency_upper != "USD":
            rate_from_per_usd = self._get_specific_rate(from_currency_upper, "USD", current_prices_map)
            rate_usd_per_ac = self._get_specific_rate("USD", account_currency_upper, current_prices_map)
            if rate_from_per_usd is not None and rate_usd_per_ac is not None and rate_from_per_usd > 0 and rate_usd_per_ac > 0: return rate_from_per_usd * rate_usd_per_ac
        if from_currency_upper == "USD" and account_currency_upper != "USD":
            rate_usd_per_ac = self._get_specific_rate("USD", account_currency_upper, current_prices_map)
            if rate_usd_per_ac is not None and rate_usd_per_ac > 0: return rate_usd_per_ac
        if account_currency_upper == "USD" and from_currency_upper != "USD":
            rate_from_per_usd = self._get_specific_rate(from_currency_upper, "USD", current_prices_map)
            if rate_from_per_usd is not None and rate_from_per_usd > 0: return rate_from_per_usd
        logger.warning(f"無法找到匯率將 {from_currency} 轉換到 {ACCOUNT_CURRENCY}。使用後備值 0.0。")
        return Decimal('0.0')
    
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
                current_price_qc = price_tuple[0] if units > 0 else price_tuple[1] # Bid for long, Ask for short
                if current_price_qc <= Decimal('0'): continue
                pnl_per_unit_qc = (current_price_qc - avg_entry_qc) if units > 0 else (avg_entry_qc - current_price_qc)
                total_pnl_qc = pnl_per_unit_qc * abs(units); pnl_in_ac = total_pnl_qc
                if details.quote_currency != ACCOUNT_CURRENCY:
                    exchange_rate_qc_to_ac = self._get_exchange_rate_to_account_currency(details.quote_currency, current_prices_map)
                    if exchange_rate_qc_to_ac > 0: pnl_in_ac = total_pnl_qc * exchange_rate_qc_to_ac
                    else: pnl_in_ac = Decimal('0.0') # 轉換失敗
                self.unrealized_pnl_ac[slot_idx] = pnl_in_ac; self.equity_ac += pnl_in_ac
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
            self.avg_entry_prices_qc[slot_idx] = (total_value_at_old_avg + total_value_at_new_trade) / new_units
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
            margin_required_qc = abs(new_units) * trade_price_qc * Decimal(str(details.margin_rate))
            self.margin_used_per_position_ac[slot_idx] = margin_required_qc * exchange_rate_qc_to_ac
            
            # 添加保證金緩衝區機制，避免邊界情況下的保證金不足
            margin_buffer = Decimal('0.02')  # 2% 緩衝區
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
        logger.info(f"執行交易: {symbol}, 類型: {trade_type}, 單位: {units_to_trade:.2f}, 價格: {trade_price_qc:.5f} QC, 手續費: {commission_ac:.2f} AC, 實現盈虧: {realized_pnl_ac:.2f} AC, 現金: {self.cash:.2f} AC, 新倉位: {new_units:.2f}")
        return units_to_trade, commission_ac

    # --- (詳細的 step, _calculate_reward, _check_termination_truncation, _get_observation, _get_info, render, close 方法將緊隨其後) ---

    # ... (if __name__ == "__main__": 測試塊與V4.8版本相同) ...
    # <在此處粘貼您上一個版本 UniversalTradingEnvV4 (V4.8) 中 if __name__ == "__main__": 塊的全部內容>
    # 為確保完整性，我將再次粘貼並檢查
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.episode_step_count += 1
        all_prices_map, current_timestamp = self._get_current_raw_prices_for_all_dataset_symbols()
        prev_portfolio_value_ac = self.portfolio_value_ac
        self._update_atr_values(all_prices_map); self._update_stop_loss_prices(all_prices_map)
        
        total_commission_this_step_ac = Decimal('0.0') # 用於獎勵計算

        # 1. 處理止損
        commission_from_sl = self._apply_stop_loss(all_prices_map, current_timestamp)
        total_commission_this_step_ac += commission_from_sl

        # 2. 處理保證金追繳 (如果止損後仍觸發)
        commission_from_mc = self._handle_margin_call(all_prices_map, current_timestamp)
        total_commission_this_step_ac += commission_from_mc

        # 3. 執行智能體動作
        for slot_idx in self.current_episode_tradable_slot_indices:
            symbol = self.slot_to_symbol_map.get(slot_idx)
            if not symbol: continue

            details = self.instrument_details_map[symbol]
            current_units = self.current_positions_units[slot_idx]
            current_bid_qc, current_ask_qc = all_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
            
            if current_bid_qc <= Decimal('0') or current_ask_qc <= Decimal('0'):
                logger.warning(f"跳過 {symbol} 的交易，因為當前價格無效。")
                continue

            # 計算目標單位數 (target_units)
            # 使用 self.max_account_risk_per_trade 和 self.atr_values_qc[slot_idx] 以及止損乘數來確定基於風險的最大單位數。
            # 結合智能體動作 action[slot_idx]（目標倉位比例）和 self.equity_ac（或 self.initial_capital）來計算目標名義價值。
            # 將目標名義價值轉換為報價貨幣，再除以當前價格得到目標單位數。
            # 使用 details.round_units() 進行調整。

            # 這裡使用 equity_ac 作為基礎，因為它反映了當前賬戶的真實價值
            risk_per_unit_qc = self.atr_values_qc[slot_idx] * self.stop_loss_atr_multiplier
            if risk_per_unit_qc <= Decimal('0'):
                logger.debug(f"跳過 {symbol} 的交易，因為 ATR 風險為零。")
                continue

            # 計算每單位在賬戶貨幣中的風險
            exchange_rate_qc_to_ac = self._get_exchange_rate_to_account_currency(details.quote_currency, all_prices_map)
            if exchange_rate_qc_to_ac <= Decimal('0'):
                logger.warning(f"無法獲取 {details.quote_currency} 到 {ACCOUNT_CURRENCY} 的匯率，跳過交易。")
                continue
            
            risk_per_unit_ac = risk_per_unit_qc * exchange_rate_qc_to_ac

            if risk_per_unit_ac <= Decimal('0'):
                logger.debug(f"跳過 {symbol} 的交易，因為每單位風險為零。")
                continue

            # 基於風險的最大單位數
            max_risk_capital = self.equity_ac * self.max_account_risk_per_trade
            max_units_by_risk = (max_risk_capital / risk_per_unit_ac).quantize(Decimal('1'), rounding=ROUND_DOWN) # 確保是整數單位

            # 智能體動作 (目標倉位比例)
            target_position_ratio = Decimal(str(action[slot_idx])) # 動作範圍 -1.0 到 1.0

            # 計算目標名義價值 (基於目標倉位比例和賬戶淨值)
            # 這裡使用 self.equity_ac 作為基礎，因為它反映了當前賬戶的真實價值
            target_nominal_value_ac = abs(target_position_ratio) * self.equity_ac

            # 將目標名義價值轉換為報價貨幣
            # 使用中間價作為轉換價格
            current_mid_price_qc = (current_bid_qc + current_ask_qc) / Decimal('2')
            if current_mid_price_qc <= Decimal('0'):
                logger.warning(f"跳過 {symbol} 的交易，因為中間價無效。")
                continue

            # 計算目標單位數 (未調整精度)
            target_units_raw = (target_nominal_value_ac / (current_mid_price_qc * exchange_rate_qc_to_ac)).quantize(Decimal('1e-9'), rounding=ROUND_HALF_UP) # 暫時保留精度

            # 結合風險限制和動作目標
            # 最終目標單位數不應超過基於風險計算的最大單位數
            target_units_final = min(target_units_raw, max_units_by_risk)
            target_units_final = target_units_final.copy_sign(target_position_ratio) # 恢復方向

            # 改進交易單位精度處理 - 使用 InstrumentDetails.round_units() 方法進行精確調整
            target_units = details.round_units(target_units_final)

            # 計算 units_to_trade
            units_to_trade = target_units - current_units
            
            if abs(units_to_trade) < details.minimum_trade_size:
                logger.debug(f"跳過 {symbol} 的交易，因為交易單位 {units_to_trade:.2f} 小於最小交易單位 {details.minimum_trade_size:.2f}。")
                continue

            # 確定交易價格 (買入用 Ask, 賣出用 Bid)
            trade_price_qc = current_ask_qc if units_to_trade > 0 else current_bid_qc
            if trade_price_qc <= Decimal('0'):
                logger.warning(f"跳過 {symbol} 的交易，因為交易價格無效。")
                continue

            # 保證金檢查
            # 計算執行 units_to_trade 預計會改變多少保證金。
            # 檢查 self.equity_ac - self.total_margin_used_ac (現有的) - 預計佣金 >= 新增保證金。
            # 如果不足，則按比例縮減 units_to_trade（確保縮減後的單位數仍符合最小交易單位）或取消該筆交易。記錄此類事件。

            # 預計新倉位
            projected_new_units = current_units + units_to_trade
            
            # 修復保證金計算精度問題 - 使用實際交易價格（bid/ask）而非中間價計算保證金
            projected_margin_required_qc = abs(projected_new_units) * trade_price_qc * Decimal(str(details.margin_rate))
            projected_margin_required_ac = projected_margin_required_qc * exchange_rate_qc_to_ac

            # 當前該槽位的保證金
            current_margin_for_slot_ac = self.margin_used_per_position_ac[slot_idx]

            # 預計總保證金 (假設其他槽位不變)
            projected_total_margin_used_ac = self.total_margin_used_ac - current_margin_for_slot_ac + projected_margin_required_ac

            # 預計交易手續費
            projected_commission_ac = abs(units_to_trade) * trade_price_qc * exchange_rate_qc_to_ac * self.commission_percentage

            # 檢查可用資金是否足夠支付新增保證金和手續費
            # 可用現金 = self.cash - projected_commission_ac
            # 可用權益 = self.equity_ac - projected_commission_ac
            
            # 這裡的邏輯應該是：新的總保證金不能導致保證金水平低於關閉水平
            # 也就是說，(self.equity_ac - projected_commission_ac) / projected_total_margin_used_ac >= OANDA_MARGIN_CLOSEOUT_LEVEL
            # 或者更直接地，確保有足夠的自由保證金
            free_margin_before_trade = self.equity_ac - self.total_margin_used_ac

            # 考慮交易後，新的自由保證金
            # 這裡需要考慮交易對現金和未實現盈虧的影響，但為了保證金檢查的簡潔性，我們主要關注保證金的變化
            # 簡化：只考慮新增保證金和手續費對現金的影響
            margin_change_ac = projected_margin_required_ac - current_margin_for_slot_ac
            
            # 如果是平倉，margin_change_ac 會是負數，表示釋放保證金
            # 如果是開倉/加倉，margin_change_ac 會是正數，表示需要更多保證金

            # 檢查是否會導致保證金追繳
            # 這裡的檢查應該是：執行這筆交易後，我的總權益減去新的總保證金，是否仍然大於一個安全閾值
            # 或者，更直接地，新的總保證金加上預計手續費，是否會超過我的可用現金/權益
            
            # 預計交易後的總權益 (考慮手續費，但不考慮未實現盈虧，因為那是浮動的)
            # 這裡的 equity_ac 已經包含了未實現盈虧，所以我們直接用它來判斷保證金水平
            # 預計交易後的現金 (只考慮手續費)
            projected_cash_after_commission = self.cash - projected_commission_ac

            # 如果是開倉/加倉，需要檢查是否有足夠的自由保證金
            if units_to_trade.copy_sign(Decimal('1')) == projected_new_units.copy_sign(Decimal('1')) or abs(current_units) < Decimal('1e-9'): # 開倉或加倉
                # 添加保證金緩衝區機制，避免邊界情況下的保證金不足
                margin_safety_buffer = Decimal('0.1')  # 10% 安全緩衝區
                safe_margin_threshold = OANDA_MARGIN_CLOSEOUT_LEVEL + margin_safety_buffer
                
                if projected_total_margin_used_ac > self.equity_ac * (Decimal('1.0') - safe_margin_threshold): # 使用安全閾值檢查
                    logger.warning(f"保證金不足以執行 {symbol} 的交易 ({units_to_trade:.2f} 單位)。預計總保證金 {projected_total_margin_used_ac:.2f} AC，權益 {self.equity_ac:.2f} AC。縮減交易單位。")
                    # 按比例縮減 units_to_trade
                    # 計算最大可交易單位
                    max_affordable_margin_ac = self.equity_ac * (Decimal('1.0') - safe_margin_threshold)
                    if max_affordable_margin_ac <= Decimal('0'):
                        logger.warning(f"無法負擔任何保證金，取消 {symbol} 的交易。")
                        continue
                    
                    # 計算基於最大可負擔保證金的最大單位數
                    max_units_by_margin_ac = (max_affordable_margin_ac / (trade_price_qc * Decimal(str(details.margin_rate)) * exchange_rate_qc_to_ac)).quantize(Decimal('1'), rounding=ROUND_DOWN)
                    
                    # 縮減 units_to_trade
                    if abs(projected_new_units) > max_units_by_margin_ac:
                        # 如果是開倉，直接限制為 max_units_by_margin_ac
                        if abs(current_units) < Decimal('1e-9'):
                            units_to_trade = max_units_by_margin_ac.copy_sign(units_to_trade)
                        else: # 加倉，需要計算可以加多少
                            units_can_add = max_units_by_margin_ac - abs(current_units)
                            units_to_trade = units_can_add.copy_sign(units_to_trade)
                        
                        units_to_trade = details.round_units(units_to_trade)
                        if abs(units_to_trade) < details.minimum_trade_size:
                            logger.warning(f"縮減後 {symbol} 的交易單位 {units_to_trade:.2f} 小於最小交易單位，取消交易。")
                            continue
                        logger.info(f"縮減 {symbol} 的交易單位至 {units_to_trade:.2f}，以符合保證金要求。")
            
            # 如果可以交易，則調用 _execute_trade()
            if abs(units_to_trade) > Decimal('0'):
                traded_units, commission = self._execute_trade(slot_idx, units_to_trade, trade_price_qc, current_timestamp, all_prices_map)
                total_commission_this_step_ac += commission
            else:
                logger.debug(f"槽位 {slot_idx} ({symbol}) 無需交易。")

        # 在所有槽位交易循環結束後，重新計算 self.total_margin_used_ac = sum(self.margin_used_per_position_ac)。
        self.total_margin_used_ac = sum(self.margin_used_per_position_ac)

        # 在所有交易和止損執行後，調用 _update_portfolio_and_equity_value() 更新最終狀態。
        self._update_portfolio_and_equity_value(all_prices_map)
        self.portfolio_value_history.append(float(self.portfolio_value_ac))
        
        # 獎勵計算
        reward = self._calculate_reward(prev_portfolio_value_ac, total_commission_this_step_ac)
        self.current_step_in_dataset += 1
        terminated, truncated = self._check_termination_truncation()
        next_observation = self._get_observation()
        info = self._get_info(); info["reward_this_step"] = reward
        return next_observation, reward, terminated, truncated, info

    def _calculate_reward(self, prev_portfolio_value_ac: Decimal, commission_this_step_ac: Decimal) -> float:
        log_return = Decimal('0.0')
        if prev_portfolio_value_ac > Decimal('0'): log_return = (self.portfolio_value_ac / prev_portfolio_value_ac).ln()
        
        reward_val = self.reward_config["portfolio_log_return_factor"] * log_return
        
        # 手續費懲罰
        commission_penalty = self.reward_config["commission_penalty_factor"] * (commission_this_step_ac / self.initial_capital)
        reward_val -= commission_penalty

        # 最大回撤懲罰
        self.peak_portfolio_value_episode = max(self.peak_portfolio_value_episode, self.portfolio_value_ac)
        current_dd = (self.peak_portfolio_value_episode - self.portfolio_value_ac) / (self.peak_portfolio_value_episode + Decimal('1e-9'))
        if current_dd > self.max_drawdown_episode :
            reward_val -= self.reward_config["max_drawdown_penalty_factor"] * (current_dd - self.max_drawdown_episode)
            self.max_drawdown_episode = current_dd
        elif current_dd > Decimal('0'):
            reward_val -= self.reward_config["max_drawdown_penalty_factor"] * current_dd * Decimal('0.1') # 對於持續回撤的輕微懲罰

        # 風險調整後收益 (簡化夏普比率)
        # 這裡可以考慮使用過去一段時間的收益標準差來計算，但為了簡化，先使用一個基於波動性的懲罰
        # 假設我們希望獎勵平穩的收益，懲罰劇烈波動
        # 這裡可以考慮使用 ATR 或其他波動性指標
        # 暫時不引入複雜的風險調整，因為這需要歷史數據，而我們在 step 中只處理當前數據
        # 如果要實現，可能需要在 info 中傳遞更多歷史數據，或者在環境中維護收益序列
        # 這裡可以簡單地懲罰高波動性，或者獎勵低波動性
        # 例如，如果 ATR 很高，可以給予輕微懲罰
        # 為了實現更精確的風險調整後收益指標（如差分夏普比率的簡化版本），我們需要維護一個收益序列。
        # 目前的 log_return 已經是收益的一部分。

        # 其他可能的獎勵/懲罰項 (持倉時間、過度交易等)
        # 持倉時間獎勵：如果一個倉位持有時間較長且盈利，可以給予獎勵
        # 過度交易懲罰：已經通過 commission_penalty 實現了一部分，可以考慮額外懲罰頻繁交易
        # 這裡暫時不引入，保持簡潔。

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
        # (與V4.8版本相同)
        dataset_sample = self.dataset[min(self.current_step_in_dataset, len(self.dataset)-1)]
        features_raw = dataset_sample["features"].numpy()
        obs_f = np.zeros((self.num_env_slots, TIMESTEPS, self.dataset.num_features_per_symbol), dtype=np.float32)
        obs_pr = np.zeros(self.num_env_slots, dtype=np.float32); obs_upl_r = np.zeros(self.num_env_slots, dtype=np.float32)
        obs_tslt_ratio = np.zeros(self.num_env_slots, dtype=np.float32); obs_pm = np.ones(self.num_env_slots, dtype=np.bool_)
        current_prices_map, _ = self._get_current_raw_prices_for_all_dataset_symbols()
        for slot_idx in range(self.num_env_slots):
            symbol = self.slot_to_symbol_map.get(slot_idx)
            if symbol:
                if symbol in self.dataset.symbols:
                    try: dataset_symbol_idx = self.dataset.symbols.index(symbol)
                    except ValueError: logger.error(f"Symbol {symbol} in slot_map but not in dataset.symbols for observation."); continue
                    obs_f[slot_idx, :, :] = features_raw[dataset_symbol_idx, :, :]
                units = self.current_positions_units[slot_idx]; details = self.instrument_details_map[symbol]
                current_bid_qc, current_ask_qc = current_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
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
        margin_level_val = float(self.equity_ac / (self.total_margin_used_ac + Decimal('1e-9')))
        return {"features_from_dataset": obs_f, "current_positions_nominal_ratio_ac": np.clip(obs_pr, -5.0, 5.0).astype(np.float32), "unrealized_pnl_ratio_ac": np.clip(obs_upl_r, -1.0, 5.0).astype(np.float32), "margin_level": np.clip(np.array([margin_level_val]), 0.0, 100.0).astype(np.float32), "time_since_last_trade_ratio": obs_tslt_ratio.astype(np.float32), "padding_mask": obs_pm}

    def _get_info(self) -> Dict[str, Any]:
        return {"cash_ac": float(self.cash), "portfolio_value_ac": float(self.portfolio_value_ac), "equity_ac": float(self.equity_ac), "total_margin_used_ac": float(self.total_margin_used_ac), "episode_step": self.episode_step_count}
    def _init_render_figure(self): pass
    def render(self): pass
    def close(self): logger.info("關閉 TradingEnvV4。"); pass

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