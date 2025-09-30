"""Risk management utilities for the live trading system."""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from oanda_trading_bot.training_system.data_manager.currency_manager import (
    CurrencyDependencyManager,
)
from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager

from .position_manager import PositionManager
from ..core.system_state import SystemState
from ..core.oanda_client import OandaClient


def _to_int(value: Any) -> int:
    try:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return int(float(str(value).strip()))
    except Exception:
        return 0


class RiskManager:
    """Enforces position sizing, exposure limits, and drawdown protection."""

    def __init__(
        self,
        config: Dict[str, Any],
        system_state: SystemState,
        position_manager: PositionManager,
        oanda_client: OandaClient,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.system_state = system_state
        self.position_manager = position_manager
        self.client = oanda_client

        # Base configuration snapshot
        live_cfg = config or {}
        self.config = dict(live_cfg.get('risk_management', {}))

        # Core risk parameters
        self.max_total_exposure_usd: float = float(self.config.get('max_total_exposure_usd', 0) or 0)
        self.max_total_exposure_percent: Optional[float] = (
            float(self.config['max_total_exposure_percent'])
            if self.config.get('max_total_exposure_percent') is not None
            else None
        )
        self.max_position_exposure_percent: Optional[float] = (
            float(self.config['max_position_exposure_percent'])
            if self.config.get('max_position_exposure_percent') is not None
            else None
        )
        self.max_risk_per_trade_percent: float = float(
            self.config.get('max_risk_per_trade_percent', 1.0)
        )

        # Volatility-aware sizing
        self.use_atr_sizing: bool = bool(self.config.get('use_atr_sizing', False))
        self.atr_period: int = int(self.config.get('atr_period', 14))
        self.stop_loss_atr_multiplier: float = float(self.config.get('stop_loss_atr_multiplier', 2.0))
        self.take_profit_atr_multiplier: float = float(self.config.get('take_profit_atr_multiplier', 3.0))

        # Pip based fallbacks
        self.stop_loss_pips: float = float(self.config.get('stop_loss_pips', 20))
        self.take_profit_pips: float = float(self.config.get('take_profit_pips', 40))

        # Drawdown protection
        self.hard_stop_loss_percent: Optional[float] = (
            float(self.config['hard_stop_loss_percent'])
            if self.config.get('hard_stop_loss_percent') is not None
            else None
        )
        self.trailing_stop_loss_percent: Optional[float] = (
            float(self.config['trailing_stop_loss_percent'])
            if self.config.get('trailing_stop_loss_percent') is not None
            else None
        )
        self.daily_loss_limit_percent: Optional[float] = (
            float(live_cfg.get('daily_loss_limit_pct'))
            if live_cfg.get('daily_loss_limit_pct') is not None
            else None
        )

        self.account_currency: str = (live_cfg.get('account_currency') or 'USD').upper()
        self.cur_mgr = CurrencyDependencyManager(self.account_currency, apply_oanda_markup=True)
        self.iim = InstrumentInfoManager()

        # Session runtime state
        self.session_target_capital: Optional[float] = None
        self.halt_triggered: bool = False
        self.halt_reason: Optional[str] = None

        self.logger.info(
            "RiskManager initialised: max_exposure_usd=%s, max_risk_per_trade_pct=%s",
            self.max_total_exposure_usd,
            self.max_risk_per_trade_percent,
        )

    # ------------------------------------------------------------------
    # Session lifecycle hooks
    # ------------------------------------------------------------------
    def register_target_capital(self, amount: Optional[float]) -> None:
        self.session_target_capital = float(amount) if amount is not None else None

    def on_session_start(self, equity: Optional[float]) -> None:
        self.halt_triggered = False
        self.halt_reason = None
        self.system_state.mark_session_start_equity(equity)
        if equity is not None:
            self.system_state.update_peak_equity(equity)

    # ------------------------------------------------------------------
    # Public helpers used by the API / UI
    # ------------------------------------------------------------------
    def get_risk_snapshot(self) -> Dict[str, Any]:
        exposure, per_symbol = self._compute_open_exposure()
        limit_total = self._exposure_cap_absolute()
        account = self._safe_account_summary()
        equity = account.get('equity')
        available = None
        if limit_total is not None:
            available = max(limit_total - exposure, 0.0)
        return {
            'account_currency': self.account_currency,
            'target_capital': self.session_target_capital,
            'max_total_exposure_usd': self.max_total_exposure_usd,
            'max_total_exposure_percent': self.max_total_exposure_percent,
            'max_position_exposure_percent': self.max_position_exposure_percent,
            'max_risk_per_trade_percent': self.max_risk_per_trade_percent,
            'use_atr_sizing': self.use_atr_sizing,
            'atr_period': self.atr_period,
            'stop_loss_atr_multiplier': self.stop_loss_atr_multiplier,
            'take_profit_atr_multiplier': self.take_profit_atr_multiplier,
            'stop_loss_pips': self.stop_loss_pips,
            'take_profit_pips': self.take_profit_pips,
            'hard_stop_loss_percent': self.hard_stop_loss_percent,
            'trailing_stop_loss_percent': self.trailing_stop_loss_percent,
            'daily_loss_limit_percent': self.daily_loss_limit_percent,
            'total_exposure': exposure,
            'per_symbol_exposure': per_symbol,
            'exposure_capacity': limit_total,
            'available_capacity': available,
            'halt_triggered': self.halt_triggered,
            'halt_reason': self.halt_reason,
            'session_start_equity': self.system_state.session_start_equity,
            'session_peak_equity': self.system_state.session_peak_equity,
            'account_equity': equity,
        }

    def should_halt_trading(self) -> Tuple[bool, Optional[str]]:
        if self.halt_triggered and self.halt_reason:
            return True, self.halt_reason

        account = self._safe_account_summary()
        equity = account.get('equity')
        if equity is None:
            return False, None

        self.system_state.update_peak_equity(equity)

        start_eq = self.system_state.session_start_equity
        if (
            start_eq is not None
            and self.hard_stop_loss_percent is not None
            and start_eq > 0
        ):
            drawdown = (start_eq - equity) / start_eq * 100
            if drawdown >= self.hard_stop_loss_percent:
                reason = (
                    f"Hard stop triggered: drawdown {drawdown:.2f}% exceeds "
                    f"{self.hard_stop_loss_percent}%"
                )
                return self._record_halt(reason)

        peak_eq = self.system_state.session_peak_equity
        if (
            peak_eq is not None
            and self.trailing_stop_loss_percent is not None
            and peak_eq > 0
        ):
            trailing_dd = max(0.0, (peak_eq - equity) / peak_eq * 100)
            if trailing_dd >= self.trailing_stop_loss_percent:
                reason = (
                    f"Trailing stop triggered: drawdown {trailing_dd:.2f}% exceeds "
                    f"{self.trailing_stop_loss_percent}%"
                )
                return self._record_halt(reason)

        return False, None

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------
    def update_params(self, params: Dict[str, Any]) -> None:
        if not params:
            return
        self.config.update(params)
        mapping = {
            'max_total_exposure_usd': 'max_total_exposure_usd',
            'max_total_exposure_percent': 'max_total_exposure_percent',
            'max_position_exposure_percent': 'max_position_exposure_percent',
            'max_risk_per_trade_percent': 'max_risk_per_trade_percent',
            'use_atr_sizing': 'use_atr_sizing',
            'atr_period': 'atr_period',
            'stop_loss_atr_multiplier': 'stop_loss_atr_multiplier',
            'take_profit_atr_multiplier': 'take_profit_atr_multiplier',
            'stop_loss_pips': 'stop_loss_pips',
            'take_profit_pips': 'take_profit_pips',
            'hard_stop_loss_percent': 'hard_stop_loss_percent',
            'trailing_stop_loss_percent': 'trailing_stop_loss_percent',
        }
        for key, attr in mapping.items():
            if key not in params:
                continue
            value = params[key]
            if value is None:
                setattr(self, attr, None)
            else:
                setattr(self, attr, float(value) if isinstance(value, (int, float)) else float(str(value)))

        if 'use_atr_sizing' in params:
            self.use_atr_sizing = bool(params['use_atr_sizing'])

        if 'max_total_exposure_usd' in params:
            self.max_total_exposure_usd = float(params['max_total_exposure_usd'] or 0)
        self.logger.info("Risk parameters updated: %s", params)

    # ------------------------------------------------------------------
    # Trade assessment (main entry point for OrderManager)
    # ------------------------------------------------------------------
    def assess_trade(self, instrument: str, signal: int, price: float) -> Optional[Dict[str, Any]]:
        self.logger.info("Assessing trade for %s signal=%s price=%s", instrument, signal, price)

        existing_position = self.position_manager.get_position(instrument)
        if existing_position:
            same_direction = (
                signal == 1 and existing_position.position_type == 'long'
            ) or (
                signal == -1 and existing_position.position_type == 'short'
            )
            if same_direction:
                self.logger.warning("Ignoring pyramiding attempt on %s", instrument)
                return None

        account = self._safe_account_summary()
        equity = account.get('equity') or 0.0

        details = self.iim.get_details(instrument)
        if details is None:
            self.logger.error("Missing instrument details for %s", instrument)
            return None

        price_map = self._build_price_map(instrument, self.client)
        rate_qc_to_ac = float(self.cur_mgr.convert_to_account_currency(details.quote_currency, price_map))
        if rate_qc_to_ac <= 0:
            self.logger.error("Unable to convert %s quote currency into %s", instrument, self.account_currency)
            return None

        risk_amount_ac = (self.max_risk_per_trade_percent / 100.0) * equity if equity > 0 else 0.0
        if risk_amount_ac <= 0:
            risk_amount_ac = min(10.0, abs(float(price) * rate_qc_to_ac))

        trade_size_units, stop_loss_price, take_profit_price = self._size_trade(
            instrument=instrument,
            signal=signal,
            price=price,
            rate_qc_to_ac=rate_qc_to_ac,
            risk_amount_ac=risk_amount_ac,
            details=details,
            )
        if trade_size_units == 0:
            self.logger.warning("Trade size rounded to zero for %s", instrument)
            return None

        approved_units = self._apply_exposure_limits(
            instrument=instrument,
            proposed_units=trade_size_units,
            price=price,
            rate_qc_to_ac=rate_qc_to_ac,
        )
        if approved_units == 0:
            self.logger.info("Exposure limits prevent new trade on %s", instrument)
            return None
        if approved_units != trade_size_units:
            self.logger.info("Exposure adjustment changed units from %s to %s for %s", trade_size_units, approved_units, instrument)
            trade_size_units = approved_units

        stop_loss_price = self._round_price_to_instrument(instrument, stop_loss_price) if stop_loss_price is not None else None
        take_profit_price = self._round_price_to_instrument(instrument, take_profit_price) if take_profit_price is not None else None

        return {
            'units': int(trade_size_units),
            'stop_loss': float(stop_loss_price) if stop_loss_price is not None else None,
            'take_profit': float(take_profit_price) if take_profit_price is not None else None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_halt(self, reason: str) -> Tuple[bool, str]:
        self.halt_triggered = True
        self.halt_reason = reason
        self.system_state.record_hard_stop(reason)
        self.logger.error(reason)
        return True, reason

    def _safe_account_summary(self) -> Dict[str, float]:
        try:
            summary = self.client.get_account_summary() or {}
            account = summary.get('account') or {}
            equity = float(account.get('NAV') or account.get('equity') or account.get('balance') or 0.0)
            balance = float(account.get('balance') or 0.0)
            unrealised = float(account.get('unrealizedPL') or 0.0)
            return {'equity': equity, 'balance': balance, 'unrealizedPL': unrealised}
        except Exception:
            return {'equity': 0.0, 'balance': 0.0, 'unrealizedPL': 0.0}

    def _exposure_cap_absolute(self) -> Optional[float]:
        limits: List[float] = []
        if self.max_total_exposure_usd:
            limits.append(float(self.max_total_exposure_usd))
        if self.session_target_capital and self.max_total_exposure_percent:
            limits.append(self.session_target_capital * self.max_total_exposure_percent / 100.0)
        if self.session_target_capital and not limits:
            limits.append(float(self.session_target_capital))
        return min(limits) if limits else None

    def _position_limit_for_symbol(self) -> Optional[float]:
        if self.session_target_capital and self.max_position_exposure_percent:
            return self.session_target_capital * self.max_position_exposure_percent / 100.0
        return None

    def _compute_open_exposure(self) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        per_symbol: Dict[str, float] = {}
        try:
            data = self.client.get_open_positions() or {}
        except Exception:
            return 0.0, {}

        for pos in data.get('positions', []):
            instrument = pos.get('instrument') or ''
            if not instrument:
                continue
            details = self.iim.get_details(instrument)
            if details is None:
                continue
            price = float(
                pos.get('long', {}).get('averagePrice')
                or pos.get('short', {}).get('averagePrice')
                or 0.0
            )
            if price <= 0:
                continue
            price_map = self._build_price_map(instrument, self.client)
            rate_qc_to_ac = float(
                self.cur_mgr.convert_to_account_currency(details.quote_currency, price_map)
            )
            if rate_qc_to_ac <= 0:
                continue
            long_units = abs(_to_int(pos.get('long', {}).get('units')))
            short_units = abs(_to_int(pos.get('short', {}).get('units')))
            notional_qc = (long_units + short_units) * price
            notional_ac = notional_qc * rate_qc_to_ac
            if notional_ac <= 0:
                continue
            total += notional_ac
            per_symbol[instrument] = per_symbol.get(instrument, 0.0) + notional_ac
        return total, per_symbol

    def _apply_exposure_limits(
        self,
        instrument: str,
        proposed_units: int,
        price: float,
        rate_qc_to_ac: float,
    ) -> int:
        exposure_total, per_symbol = self._compute_open_exposure()
        proposed_notional = abs(proposed_units) * price * rate_qc_to_ac

        total_cap = self._exposure_cap_absolute()
        if total_cap is not None:
            if exposure_total >= total_cap:
                self.logger.warning("Total exposure cap already reached (cap=%s)", total_cap)
                return 0
            remaining = max(total_cap - exposure_total, 0.0)
            if proposed_notional > remaining:
                scale = remaining / max(proposed_notional, 1e-9)
                proposed_units = int(proposed_units * scale)
                proposed_notional = abs(proposed_units) * price * rate_qc_to_ac

        per_symbol_cap = self._position_limit_for_symbol()
        if per_symbol_cap is not None:
            current = per_symbol.get(instrument, 0.0)
            if current >= per_symbol_cap:
                self.logger.warning("Symbol %s exposure cap reached (cap=%s)", instrument, per_symbol_cap)
                return 0
            remaining_symbol = max(per_symbol_cap - current, 0.0)
            if proposed_notional > remaining_symbol:
                scale = remaining_symbol / max(proposed_notional, 1e-9)
                proposed_units = int(proposed_units * scale)

        details = self.iim.get_details(instrument)
        if details is not None and proposed_units != 0:
            try:
                rounded = details.round_units(proposed_units)
                proposed_units = int(rounded)
            except Exception:
                proposed_units = int(proposed_units)
        return proposed_units

    def _round_price_to_instrument(self, instrument: str, price: Optional[float]) -> Optional[float]:
        if price is None:
            return None
        try:
            det = self.iim.get_details(instrument)
            pip_location = int(getattr(det, 'pip_location', 0)) if det is not None else 0
            decimals = max(0, -pip_location)
            quant = Decimal('1').scaleb(-decimals)
            return float(Decimal(str(price)).quantize(quant))
        except Exception:
            return float(price)

    def _build_price_map(self, instrument: str, oanda_client: OandaClient) -> Dict[str, Tuple[Decimal, Decimal]]:
        sym = instrument
        parts = sym.split('_')
        base, quote = parts[0], parts[1]
        mp: Dict[str, Tuple[Decimal, Decimal]] = {}

        try:
            valid_symbols = set(
                getattr(self.iim, 'all_symbols', []) or self.iim.get_all_available_symbols()
            )
        except Exception:
            valid_symbols = set()

        def _fetch(pair_name: str) -> Optional[Tuple[Decimal, Decimal]]:
            try:
                candles = oanda_client.get_bid_ask_candles_combined(pair_name, count=1) or []
                if candles:
                    last = candles[-1]
                    return (
                        Decimal(str(last.get('bid_close', 0.0))),
                        Decimal(str(last.get('ask_close', 0.0))),
                    )
            except Exception:
                return None
            return None

        def _invert(bid_ask: Tuple[Decimal, Decimal]) -> Tuple[Decimal, Decimal]:
            bid, ask = bid_ask
            try:
                inv_bid = Decimal('1') / (ask if ask != 0 else Decimal('1'))
                inv_ask = Decimal('1') / (bid if bid != 0 else Decimal('1'))
                return inv_bid, inv_ask
            except Exception:
                return Decimal('0'), Decimal('0')

        recs = oanda_client.get_bid_ask_candles_combined(sym, count=1) or []
        if recs:
            last = recs[-1]
            mp[sym] = (
                Decimal(str(last.get('bid_close', 0.0))),
                Decimal(str(last.get('ask_close', 0.0))),
            )

        candidates = [
            (quote, self.account_currency),
            (self.account_currency, quote),
            (base, self.account_currency),
            (self.account_currency, base),
        ]

        for a, b in candidates:
            pair = f"{a}_{b}"
            if pair in mp:
                continue
            if a == b:
                mp[pair] = (Decimal('1'), Decimal('1'))
                continue
            fetch_direct = None
            if not valid_symbols or pair in valid_symbols:
                fetch_direct = _fetch(pair)
            if fetch_direct:
                mp[pair] = fetch_direct
                continue
            rev = f"{b}_{a}"
            fetch_rev = None
            if not valid_symbols or rev in valid_symbols:
                fetch_rev = _fetch(rev)
            if fetch_rev:
                mp[pair] = _invert(fetch_rev)
        return mp

    def _size_trade(
        self,
        instrument: str,
        signal: int,
        price: float,
        rate_qc_to_ac: float,
        risk_amount_ac: float,
        details: Any,
    ) -> Tuple[int, Optional[float], Optional[float]]:
        trade_units = 0
        stop_loss_price = None
        take_profit_price = None

        if self.use_atr_sizing and self.stop_loss_atr_multiplier > 0:
            atr_value = self._compute_atr(instrument)
            if atr_value > 0:
                stop_distance_qc = atr_value * self.stop_loss_atr_multiplier
                risk_per_unit_ac = stop_distance_qc * rate_qc_to_ac
                if risk_per_unit_ac > 0:
                    units_float = risk_amount_ac / risk_per_unit_ac
                    units_decimal = details.round_units(units_float)
                    trade_units = int(units_decimal) if signal == 1 else -int(units_decimal)
                    stop_loss_price = price - stop_distance_qc if signal == 1 else price + stop_distance_qc
                    if self.take_profit_atr_multiplier > 0:
                        tp_distance_qc = atr_value * self.take_profit_atr_multiplier
                        take_profit_price = price + tp_distance_qc if signal == 1 else price - tp_distance_qc

        if trade_units == 0:
            pip_value_qc_per_unit = abs(float(Decimal(str(10)) ** Decimal(str(details.pip_location))))
            pip_value_ac_per_unit = pip_value_qc_per_unit * rate_qc_to_ac
            if pip_value_ac_per_unit <= 0:
                return 0, None, None
            units_float = risk_amount_ac / max(self.stop_loss_pips * pip_value_ac_per_unit, 1e-9)
            units_decimal = details.round_units(units_float)
            trade_units = int(units_decimal) if signal == 1 else -int(units_decimal)
            pip_step = (10 ** details.pip_location)
            stop_loss_price = price - self.stop_loss_pips * pip_step if signal == 1 else price + self.stop_loss_pips * pip_step
            take_profit_price = price + self.take_profit_pips * pip_step if signal == 1 else price - self.take_profit_pips * pip_step

        if trade_units == 0:
            min_units = int(getattr(details, 'minimum_trade_size', 1))
            trade_units = min_units if signal == 1 else -min_units

        return trade_units, stop_loss_price, take_profit_price

    def _compute_atr(self, instrument: str) -> float:
        try:
            candles = self.client.get_bid_ask_candles_combined(instrument, count=max(self.atr_period + 2, 20)) or []
        except Exception:
            return 0.0
        if len(candles) < self.atr_period + 1:
            return 0.0
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []
        for rec in candles:
            bid_high = float(rec.get('bid_high', 0.0))
            ask_high = float(rec.get('ask_high', 0.0))
            bid_low = float(rec.get('bid_low', 0.0))
            ask_low = float(rec.get('ask_low', 0.0))
            bid_close = float(rec.get('bid_close', 0.0))
            ask_close = float(rec.get('ask_close', 0.0))
            highs.append((bid_high + ask_high) / 2.0)
            lows.append((bid_low + ask_low) / 2.0)
            closes.append((bid_close + ask_close) / 2.0)
        true_ranges: List[float] = []
        for idx in range(1, len(highs)):
            tr = max(
                highs[idx] - lows[idx],
                abs(highs[idx] - closes[idx - 1]),
                abs(lows[idx] - closes[idx - 1]),
            )
            true_ranges.append(tr)
        if len(true_ranges) < self.atr_period:
            return 0.0
        return sum(true_ranges[-self.atr_period:]) / float(self.atr_period)


