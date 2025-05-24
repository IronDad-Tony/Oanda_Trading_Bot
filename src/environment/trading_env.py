# src/environment/trading_env.py
"""
通用多資產強化學習交易環境 - (V4.8 - 統一價格數據處理)
"""
# ... (頂部的導入和後備導入邏輯與 V4.7 版本相同) ...
# <在此處粘貼您上一個版本 trading_env.py 中從文件頂部到 UniversalTradingEnvV4 類定義之前的全部內容>
# 我將重新提供頂部導入，確保所有內容都在
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, getcontext, ROUND_DOWN
import sys
from pathlib import Path
import logging

getcontext().prec = 30
logger: logging.Logger = logging.getLogger("trading_env_module_init") # type: ignore
# ... (logger 初始化和 try-except import 塊與 V4.7 版本相同，這裡省略以節省篇幅) ...
# <確保V4.7的完整頂部導入邏輯被粘貼於此>
# 為了簡潔，我直接粘貼V4.7的導入部分
_logger_initialized_by_common_env_v48 = False
try:
    from common.logger_setup import logger as common_configured_logger; logger = common_configured_logger; _logger_initialized_by_common_env_v48 = True
    logger.debug("trading_env.py (V4.8): Successfully imported logger from common.logger_setup.")
    from common.config import (TIMESTEPS as _TIMESTEPS, MAX_SYMBOLS_ALLOWED as _MAX_SYMBOLS_ALLOWED, ACCOUNT_CURRENCY as _ACCOUNT_CURRENCY, INITIAL_CAPITAL as _DEFAULT_INITIAL_CAPITAL, OANDA_MARGIN_CLOSEOUT_LEVEL as _OANDA_MARGIN_CLOSEOUT_LEVEL, TRADE_COMMISSION_PERCENTAGE as _TRADE_COMMISSION_PERCENTAGE, OANDA_API_KEY as _OANDA_API_KEY, ATR_PERIOD as _ATR_PERIOD, STOP_LOSS_ATR_MULTIPLIER as _STOP_LOSS_ATR_MULTIPLIER, MAX_ACCOUNT_RISK_PERCENTAGE as _MAX_ACCOUNT_RISK_PERCENTAGE)
    _config_values_env_v48 = {"TIMESTEPS": _TIMESTEPS, "MAX_SYMBOLS_ALLOWED": _MAX_SYMBOLS_ALLOWED, "ACCOUNT_CURRENCY": _ACCOUNT_CURRENCY, "DEFAULT_INITIAL_CAPITAL": _DEFAULT_INITIAL_CAPITAL, "OANDA_MARGIN_CLOSEOUT_LEVEL": _OANDA_MARGIN_CLOSEOUT_LEVEL, "TRADE_COMMISSION_PERCENTAGE": _TRADE_COMMISSION_PERCENTAGE, "OANDA_API_KEY": _OANDA_API_KEY, "ATR_PERIOD": _ATR_PERIOD, "STOP_LOSS_ATR_MULTIPLIER": _STOP_LOSS_ATR_MULTIPLIER, "MAX_ACCOUNT_RISK_PERCENTAGE": _MAX_ACCOUNT_RISK_PERCENTAGE}
    logger.info("trading_env.py (V4.8): Successfully imported and stored common.config values.")
    from data_manager.mmap_dataset import UniversalMemoryMappedDataset
    from data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols
    from data_manager.instrument_info_manager import InstrumentDetails, InstrumentInfoManager
    logger.info("trading_env.py (V4.8): Successfully imported other dependencies.")
except ImportError as e_initial_import_v48:
    logger.warning(f"trading_env.py (V4.8): Initial import failed: {e_initial_import_v48}. Attempting path adjustment...")
    project_root_env_v48 = Path(__file__).resolve().parent.parent.parent
    if str(project_root_env_v48) not in sys.path: sys.path.insert(0, str(project_root_env_v48)); logger.info(f"trading_env.py (V4.8): Added project root to sys.path: {project_root_env_v48}")
    try:
        from src.common.logger_setup import logger as common_logger_retry_v48; logger = common_logger_retry_v48; _logger_initialized_by_common_env_v48 = True
        logger.info("trading_env.py (V4.8): Successfully re-imported common_logger after path adj.")
        from src.common.config import (TIMESTEPS as _TIMESTEPS_R, MAX_SYMBOLS_ALLOWED as _MAX_SYMBOLS_ALLOWED_R, ACCOUNT_CURRENCY as _ACCOUNT_CURRENCY_R, INITIAL_CAPITAL as _DEFAULT_INITIAL_CAPITAL_R, OANDA_MARGIN_CLOSEOUT_LEVEL as _OANDA_MARGIN_CLOSEOUT_LEVEL_R, TRADE_COMMISSION_PERCENTAGE as _TRADE_COMMISSION_PERCENTAGE_R, OANDA_API_KEY as _OANDA_API_KEY_R, ATR_PERIOD as _ATR_PERIOD_R, STOP_LOSS_ATR_MULTIPLIER as _STOP_LOSS_ATR_MULTIPLIER_R, MAX_ACCOUNT_RISK_PERCENTAGE as _MAX_ACCOUNT_RISK_PERCENTAGE_R)
        _config_values_env_v48 = {"TIMESTEPS": _TIMESTEPS_R, "MAX_SYMBOLS_ALLOWED": _MAX_SYMBOLS_ALLOWED_R, "ACCOUNT_CURRENCY": _ACCOUNT_CURRENCY_R, "DEFAULT_INITIAL_CAPITAL": _DEFAULT_INITIAL_CAPITAL_R, "OANDA_MARGIN_CLOSEOUT_LEVEL": _OANDA_MARGIN_CLOSEOUT_LEVEL_R, "TRADE_COMMISSION_PERCENTAGE": _TRADE_COMMISSION_PERCENTAGE_R, "OANDA_API_KEY": _OANDA_API_KEY_R, "ATR_PERIOD": _ATR_PERIOD_R, "STOP_LOSS_ATR_MULTIPLIER": _STOP_LOSS_ATR_MULTIPLIER_R, "MAX_ACCOUNT_RISK_PERCENTAGE": _MAX_ACCOUNT_RISK_PERCENTAGE_R}
        logger.info("trading_env.py (V4.8): Successfully re-imported and stored common.config after path adjustment.")
        from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset
        from src.data_manager.oanda_downloader import format_datetime_for_oanda, manage_data_download_for_symbols
        from src.data_manager.instrument_info_manager import InstrumentDetails, InstrumentInfoManager
        logger.info("trading_env.py (V4.8): Successfully re-imported other dependencies after path adjustment.")
    except ImportError as e_retry_critical_v48:
        logger.error(f"trading_env.py (V4.8): Critical import error after path adjustment: {e_retry_critical_v48}", exc_info=True)
        logger.warning("trading_env.py (V4.8): Using fallback values for config (critical error during import).")
        _config_values_env_v48 = {"TIMESTEPS": 128, "MAX_SYMBOLS_ALLOWED": 20, "ACCOUNT_CURRENCY": "AUD", "DEFAULT_INITIAL_CAPITAL": 100000.0, "OANDA_MARGIN_CLOSEOUT_LEVEL": Decimal('0.50'), "TRADE_COMMISSION_PERCENTAGE": Decimal('0.0001'), "OANDA_API_KEY": None, "ATR_PERIOD": 14, "STOP_LOSS_ATR_MULTIPLIER": Decimal('2.0'), "MAX_ACCOUNT_RISK_PERCENTAGE": Decimal('0.01')}
        for k_fallback, v_fallback in _config_values_env_v48.items(): globals()[k_fallback] = v_fallback
        if 'UniversalMemoryMappedDataset' not in globals(): UniversalMemoryMappedDataset = type('DummyDataset', (), {'__init__': lambda self, **kwargs: setattr(self, 'symbols', []), '__len__': lambda self: 0, 'timesteps_history': 128, 'num_features_per_symbol': 9, 'aligned_timestamps': pd.Series()})
        if 'InstrumentDetails' not in globals(): InstrumentDetails = type('DummyInstrumentDetails', (), {})
        if 'InstrumentInfoManager' not in globals(): InstrumentInfoManager = type('DummyInfoManager', (), {'get_details': lambda self, sym: InstrumentDetails(symbol=sym, quote_currency="USD", base_currency=sym, margin_rate=0.05, minimum_trade_size=1, trade_units_precision=0, pip_location=-4, type="CURRENCY", display_name=sym)}) # type: ignore
        if 'format_datetime_for_oanda' not in globals():
            def format_datetime_for_oanda(dt):
                return dt.isoformat()
        if 'manage_data_download_for_symbols' not in globals():
            def manage_data_download_for_symbols(*args, **kwargs):
                logger.error("Downloader not available in fallback.")
        logger.info("trading_env.py (V4.8): Fallback definitions applied.")

TIMESTEPS = _config_values_env_v48.get("TIMESTEPS", 128); MAX_SYMBOLS_ALLOWED = _config_values_env_v48.get("MAX_SYMBOLS_ALLOWED", 20)
ACCOUNT_CURRENCY = _config_values_env_v48.get("ACCOUNT_CURRENCY", "AUD"); DEFAULT_INITIAL_CAPITAL = _config_values_env_v48.get("DEFAULT_INITIAL_CAPITAL", 100000.0)
OANDA_MARGIN_CLOSEOUT_LEVEL = _config_values_env_v48.get("OANDA_MARGIN_CLOSEOUT_LEVEL", Decimal('0.50')); TRADE_COMMISSION_PERCENTAGE = _config_values_env_v48.get("TRADE_COMMISSION_PERCENTAGE", Decimal('0.0001'))
OANDA_API_KEY = _config_values_env_v48.get("OANDA_API_KEY", None); ATR_PERIOD = _config_values_env_v48.get("ATR_PERIOD", 14)
STOP_LOSS_ATR_MULTIPLIER = _config_values_env_v48.get("STOP_LOSS_ATR_MULTIPLIER", Decimal('2.0')); MAX_ACCOUNT_RISK_PERCENTAGE = _config_values_env_v48.get("MAX_ACCOUNT_RISK_PERCENTAGE", Decimal('0.01'))


class UniversalTradingEnvV4(gym.Env):
    # ... ( __init__ 與 V4.7 版本相同 ) ...
    # <在此處粘貼您上一個版本 UniversalTradingEnvV4 (V4.7) 中 __init__ 方法的完整實現>
    metadata = {'render_modes': ['human', 'array'], 'render_fps': 10}
    def __init__(self, dataset: UniversalMemoryMappedDataset, instrument_info_manager: InstrumentInfoManager, active_symbols_for_episode: List[str], # type: ignore
                 initial_capital: float = float(DEFAULT_INITIAL_CAPITAL), max_episode_steps: Optional[int] = None,
                 commission_percentage_override: Optional[float] = None, reward_config: Optional[Dict[str, Union[float, Decimal]]] = None,
                 max_account_risk_per_trade: float = float(MAX_ACCOUNT_RISK_PERCENTAGE),
                 stop_loss_atr_multiplier: float = float(STOP_LOSS_ATR_MULTIPLIER),
                 atr_period: int = ATR_PERIOD, render_mode: Optional[str] = None):
        super().__init__()
        self.dataset = dataset; self.instrument_info_manager = instrument_info_manager
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
        logger.info(f"UniversalTradingEnvV4 (Detailed Step - V4.8) 初始化完成。")


    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        # ... (與 V4.3 相同) ...
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
        self.peak_portfolio_value_episode = self.initial_capital; self.max_drawdown_episode = Decimal('0.0')
        logger.debug(f"Env reset. Initial capital: {self.cash} {ACCOUNT_CURRENCY}. Start step: {self.current_step_in_dataset}")
        all_prices_map, _ = self._get_current_raw_prices_for_all_dataset_symbols()
        self._update_atr_values(all_prices_map)
        self._update_portfolio_and_equity_value(all_prices_map)
        return self._get_observation(), self._get_info()

    def _get_current_raw_prices_for_all_dataset_symbols(self) -> Tuple[Dict[str, Tuple[Decimal, Decimal]], pd.Timestamp]:
        # (與V4.3版本相同)
        dataset_sample = self.dataset[min(self.current_step_in_dataset, len(self.dataset)-1)]
        latest_raw_prices_np = dataset_sample["raw_prices"][:, -1, :].numpy().astype(np.float64)
        prices_map: Dict[str, Tuple[Decimal, Decimal]] = {}
        for i, symbol_name in enumerate(self.dataset.symbols):
            bid_price = Decimal(str(latest_raw_prices_np[i, 0]))
            ask_price = Decimal(str(latest_raw_prices_np[i, 1]))
            if bid_price > 0 and ask_price > 0 and ask_price >= bid_price:
                prices_map[symbol_name] = (bid_price, ask_price)
            else:
                prices_map[symbol_name] = (Decimal('0.0'), Decimal('0.0'))
                logger.debug(f"Symbol {symbol_name} 在當前步驟獲得無效價格: bid={latest_raw_prices_np[i, 0]}, ask={latest_raw_prices_np[i, 1]}")
        timestamp_index = self.current_step_in_dataset + self.dataset.timesteps_history - 1
        timestamp_index = min(timestamp_index, len(self.dataset.aligned_timestamps)-1)
        current_timestamp = self.dataset.aligned_timestamps[timestamp_index]
        return prices_map, current_timestamp
    
    def _get_specific_rate(self, base_curr: str, quote_curr: str, current_prices_map: Dict[str, Tuple[Decimal, Decimal]]) -> Optional[Decimal]:
        # (與V4.3版本相同)
        base_curr_upper = base_curr.upper(); quote_curr_upper = quote_curr.upper()
        if base_curr_upper == quote_curr_upper: return Decimal('1.0')
        pair1 = f"{base_curr_upper}_{quote_curr_upper}"; pair2 = f"{quote_curr_upper}_{base_curr_upper}"
        
        price_pair1 = current_prices_map.get(pair1)
        if price_pair1 and price_pair1[1] > 0: # ask price for pair1
            return price_pair1[1]
            
        price_pair2 = current_prices_map.get(pair2)
        if price_pair2 and price_pair2[0] > 0: # bid price for pair2
            return Decimal('1.0') / price_pair2[0]
        return None

    def _get_exchange_rate_to_account_currency(self, from_currency: str, current_prices_map: Dict[str, Tuple[Decimal, Decimal]]) -> Decimal:
        # (與V4.3版本相同)
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
        logger.warning(f"無法找到匯率將 {from_currency} 轉換到 {ACCOUNT_CURRENCY} (使用價格快照)。將使用後備值 0.0。")
        return Decimal('0.0')
    
    def _update_atr_values(self, current_prices_map: Dict[str, Tuple[Decimal, Decimal]]):
        # (與V4.3版本相同)
        for slot_idx in range(self.num_env_slots):
            symbol = self.slot_to_symbol_map.get(slot_idx)
            if not symbol: self.atr_values_qc[slot_idx] = Decimal('0.0'); continue
            details = self.instrument_details_map.get(symbol)
            current_price_tuple = current_prices_map.get(symbol)
            if not details or not current_price_tuple: self.atr_values_qc[slot_idx] = Decimal('0.0'); continue
            current_bid_qc, current_ask_qc = current_price_tuple
            if current_bid_qc <=0 or current_ask_qc <=0 or current_ask_qc < current_bid_qc:
                self.atr_values_qc[slot_idx] = details.pip_value_in_quote_currency_per_unit * Decimal('10'); continue
            spread = current_ask_qc - current_bid_qc; estimated_atr = spread * Decimal('5')
            min_atr_val = details.pip_value_in_quote_currency_per_unit * Decimal('10')
            self.atr_values_qc[slot_idx] = max(estimated_atr, min_atr_val)

    def _update_stop_loss_prices(self, current_prices_map: Dict[str, Tuple[Decimal, Decimal]]):
        # (與V4.3版本相同)
        stop_loss_mult = self.stop_loss_atr_multiplier
        for slot_idx in range(self.num_env_slots):
            units = self.current_positions_units[slot_idx]
            if abs(units) > Decimal('1e-9'):
                avg_entry_qc = self.avg_entry_prices_qc[slot_idx]; atr_qc = self.atr_values_qc[slot_idx]
                if atr_qc <= Decimal(0): self.stop_loss_prices_qc[slot_idx] = Decimal('0.0'); continue
                if units > 0: self.stop_loss_prices_qc[slot_idx] = avg_entry_qc - (atr_qc * stop_loss_mult)
                else: self.stop_loss_prices_qc[slot_idx] = avg_entry_qc + (atr_qc * stop_loss_mult)
            else: self.stop_loss_prices_qc[slot_idx] = Decimal('0.0')

    def _update_portfolio_and_equity_value(self, current_prices_map: Dict[str, Tuple[Decimal, Decimal]]):
        # (與V4.3版本相同)
        self.equity_ac = self.cash;
        for slot_idx in range(self.num_env_slots):
            self.unrealized_pnl_ac[slot_idx] = Decimal('0.0')
            units = self.current_positions_units[slot_idx]
            if abs(units) > Decimal('1e-9'):
                symbol = self.slot_to_symbol_map[slot_idx]; avg_entry_qc = self.avg_entry_prices_qc[slot_idx]
                if not symbol or avg_entry_qc <= Decimal('0'): continue
                details = self.instrument_details_map[symbol]; current_price_tuple = current_prices_map.get(symbol)
                if not current_price_tuple : continue
                current_price_qc = current_price_tuple[0] if units > 0 else current_price_tuple[1]
                if current_price_qc <= Decimal('0'): continue
                pnl_per_unit_qc = (current_price_qc - avg_entry_qc) if units > 0 else (avg_entry_qc - current_price_qc)
                total_pnl_qc = pnl_per_unit_qc * abs(units); pnl_in_ac = total_pnl_qc
                if details.quote_currency != ACCOUNT_CURRENCY:
                    exchange_rate_qc_to_ac = self._get_exchange_rate_to_account_currency(details.quote_currency, current_prices_map)
                    if exchange_rate_qc_to_ac > 0: pnl_in_ac = total_pnl_qc * exchange_rate_qc_to_ac
                    else: pnl_in_ac = Decimal('0.0')
                self.unrealized_pnl_ac[slot_idx] = pnl_in_ac; self.equity_ac += pnl_in_ac
        self.portfolio_value_ac = self.equity_ac
        
    # --- step 和 _calculate_reward 的簡化版 (將在下一個回復中提供完整版) ---
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # (與V4.3相同的簡化版step)
        self.episode_step_count += 1
        all_prices_map, current_timestamp = self._get_current_raw_prices_for_all_dataset_symbols()
        prev_portfolio_value_ac = self.portfolio_value_ac
        self._update_atr_values(all_prices_map); self._update_stop_loss_prices(all_prices_map)
        for slot_idx in self.current_episode_tradable_slot_indices:
            units = self.current_positions_units[slot_idx]; stop_loss_price_qc = self.stop_loss_prices_qc[slot_idx]
            symbol = self.slot_to_symbol_map.get(slot_idx)
            if symbol and abs(units) > Decimal('1e-9') and stop_loss_price_qc > Decimal('0'):
                current_bid_qc, current_ask_qc = all_prices_map.get(symbol, (Decimal('0'), Decimal('0')))
                closed_by_sl = False
                if units > 0 and current_bid_qc <= stop_loss_price_qc and current_bid_qc > 0 : closed_by_sl = True
                elif units < 0 and current_ask_qc >= stop_loss_price_qc and current_ask_qc > 0: closed_by_sl = True
                if closed_by_sl:
                    logger.info(f"止損觸發 for {symbol} at step {self.episode_step_count}.")
                    self.current_positions_units[slot_idx] = Decimal('0.0'); self.avg_entry_prices_qc[slot_idx] = Decimal('0.0'); self.margin_used_per_position_ac[slot_idx] = Decimal('0.0')
        self._update_portfolio_and_equity_value(all_prices_map)
        self.portfolio_value_history.append(float(self.portfolio_value_ac))
        reward = self._calculate_reward(prev_portfolio_value_ac, Decimal('0.0'))
        self.current_step_in_dataset += 1
        terminated, truncated = self._check_termination_truncation()
        next_observation = self._get_observation()
        info = self._get_info(); info["reward_this_step"] = reward
        return next_observation, reward, terminated, truncated, info

    def _calculate_reward(self, prev_portfolio_value_ac: Decimal, commission_this_step_ac: Decimal) -> float:
        # (與V4.3版本相同)
        log_return = Decimal('0.0')
        if prev_portfolio_value_ac > Decimal('0'): log_return = (self.portfolio_value_ac / prev_portfolio_value_ac).ln()
        reward_val = self.reward_config["portfolio_log_return_factor"] * log_return
        commission_penalty = self.reward_config["commission_penalty_factor"] * (commission_this_step_ac / self.initial_capital)
        reward_val -= commission_penalty
        self.peak_portfolio_value_episode = max(self.peak_portfolio_value_episode, self.portfolio_value_ac)
        current_dd = (self.peak_portfolio_value_episode - self.portfolio_value_ac) / (self.peak_portfolio_value_episode + Decimal('1e-9'))
        if current_dd > self.max_drawdown_episode :
            reward_val -= self.reward_config["max_drawdown_penalty_factor"] * (current_dd - self.max_drawdown_episode)
            self.max_drawdown_episode = current_dd
        elif current_dd > Decimal('0'): reward_val -= self.reward_config["max_drawdown_penalty_factor"] * current_dd * Decimal('0.1')
        return float(reward_val)

    def _check_termination_truncation(self) -> Tuple[bool, bool]:
        # (與V4.3版本相同)
        terminated = False
        oanda_closeout_level_decimal = Decimal(str(OANDA_MARGIN_CLOSEOUT_LEVEL))
        if self.portfolio_value_ac < self.initial_capital * oanda_closeout_level_decimal * Decimal('0.4'):
            logger.warning(f"Episode terminated: Portfolio value ({self.portfolio_value_ac:.2f}) too low.")
            terminated = True
        self.total_margin_used_ac = sum(self.margin_used_per_position_ac)
        if self.total_margin_used_ac > Decimal('0'):
            margin_level = self.equity_ac / self.total_margin_used_ac
            if margin_level < oanda_closeout_level_decimal:
                logger.warning(f"強制平倉觸發! Equity={self.equity_ac:.2f}, MarginUsed={self.total_margin_used_ac:.2f}, Level={margin_level:.2%}")
                terminated = True
        truncated = self.episode_step_count >= self.max_episode_steps
        if self.current_step_in_dataset >= len(self.dataset) :
            truncated = True; terminated = True if not terminated else True
        return terminated, truncated

    def _get_observation(self) -> Dict[str, np.ndarray]:
        # (與V4.3版本相同)
        dataset_sample = self.dataset[min(self.current_step_in_dataset, len(self.dataset)-1)]
        features_raw = dataset_sample["features"].numpy()
        obs_f = np.zeros((self.num_env_slots, TIMESTEPS, self.dataset.num_features_per_symbol), dtype=np.float32)
        obs_pr = np.zeros(self.num_env_slots, dtype=np.float32); obs_upl_r = np.zeros(self.num_env_slots, dtype=np.float32)
        obs_tslt_ratio = np.zeros(self.num_env_slots, dtype=np.float32)
        obs_pm = np.ones(self.num_env_slots, dtype=np.bool_)
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
                nominal_value_qc = abs(units) * price_for_value_calc_qc
                nominal_value_ac = nominal_value_qc
                if details.quote_currency != ACCOUNT_CURRENCY:
                    rate = self._get_exchange_rate_to_account_currency(details.quote_currency, current_prices_map)
                    if rate > 0: nominal_value_ac *= rate
                    else: nominal_value_ac = Decimal('0.0')
                obs_pr[slot_idx] = float( (nominal_value_ac / self.initial_capital) * units.copy_sign(Decimal('1')) )
                obs_upl_r[slot_idx] = float(self.unrealized_pnl_ac[slot_idx] / self.initial_capital)
                obs_pm[slot_idx] = False
                if self.last_trade_step_per_slot[slot_idx] == -1 or self.episode_step_count == 0 : obs_tslt_ratio[slot_idx] = 1.0
                else:
                    steps_since_last = self.episode_step_count - self.last_trade_step_per_slot[slot_idx]
                    obs_tslt_ratio[slot_idx] = min(1.0, steps_since_last / (self.max_episode_steps / 10.0 if self.max_episode_steps > 0 else 100.0) )
        margin_level_val = float(self.equity_ac / (self.total_margin_used_ac + Decimal('1e-9')))
        return {"features_from_dataset": obs_f,
                "current_positions_nominal_ratio_ac": np.clip(obs_pr, -5.0, 5.0).astype(np.float32),
                "unrealized_pnl_ratio_ac": np.clip(obs_upl_r, -1.0, 5.0).astype(np.float32),
                "margin_level": np.clip(np.array([margin_level_val]), 0.0, 100.0).astype(np.float32),
                "time_since_last_trade_ratio": obs_tslt_ratio.astype(np.float32),
                "padding_mask": obs_pm}

    def _get_info(self) -> Dict[str, Any]:
        return {"cash_ac": float(self.cash), "portfolio_value_ac": float(self.portfolio_value_ac), "equity_ac": float(self.equity_ac), "total_margin_used_ac": float(self.total_margin_used_ac), "episode_step": self.episode_step_count}
    def _init_render_figure(self): pass
    def render(self): pass
    def close(self): logger.info("關閉 TradingEnvV4。"); pass

# --- if __name__ == "__main__": 測試塊 (與V4.3版本相同) ---
if __name__ == "__main__":
    # ... (與您上一個版本 UniversalTradingEnvV4.3 __main__ 測試塊相同的代碼) ...
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
    # symbols_needed_for_details 的計算應使用正確的 ACCOUNT_CURRENCY
    account_currency_upper_main = ACCOUNT_CURRENCY.upper()
    symbols_needed_for_details = list(set(active_episode_symbols_main + [sym for sym in test_symbols_list_main if account_currency_upper_main in sym.upper().split("_") or "USD" in sym.upper().split("_") or sym == f"{account_currency_upper_main}_USD" or sym == f"USD_{account_currency_upper_main}"]))
    logger.info(f"為環境準備InstrumentDetails的Symbols列表: {symbols_needed_for_details}")
    # test_instrument_details_map_for_env 不再需要，因為env會自己從manager獲取
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