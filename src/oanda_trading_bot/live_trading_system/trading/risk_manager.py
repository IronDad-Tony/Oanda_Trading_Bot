import logging
from typing import Dict, Any, Optional
from decimal import Decimal
from oanda_trading_bot.training_system.data_manager.currency_manager import CurrencyDependencyManager
from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager

from .position_manager import PositionManager, Position
from ..core.system_state import SystemState

class RiskManager:
    """
    Enforces risk management rules before any order is placed.
    """
    def __init__(
        self, 
        config: Dict[str, Any], 
        system_state: SystemState, 
        position_manager: PositionManager
    ):
        """
        Initializes the RiskManager.

        Args:
            config (Dict[str, Any]): The system's configuration, containing risk parameters.
            system_state (SystemState): The global state manager.
            position_manager (PositionManager): The manager for open positions.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get('risk_management', {})
        self.system_state = system_state
        self.position_manager = position_manager

        # Load risk parameters from config
        self.max_total_exposure_usd = self.config.get('max_total_exposure_usd', 1000)
        self.max_risk_per_trade_percent = self.config.get('max_risk_per_trade_percent', 1.0)
        # Fallback (pip-based) parameters
        self.stop_loss_pips = self.config.get('stop_loss_pips', 20)
        self.take_profit_pips = self.config.get('take_profit_pips', 40)
        # ATR-based sizing parameters
        self.use_atr_sizing = self.config.get('use_atr_sizing', False)
        self.atr_period = int(self.config.get('atr_period', 14))
        self.stop_loss_atr_multiplier = float(self.config.get('stop_loss_atr_multiplier', 2.0))
        self.take_profit_atr_multiplier = float(self.config.get('take_profit_atr_multiplier', 3.0))
        
        self.logger.info("RiskManager initialized with the following parameters:")
        self.logger.info(f"- Max Total Exposure (USD): {self.max_total_exposure_usd}")
        self.logger.info(f"- Max Risk Per Trade: {self.max_risk_per_trade_percent}%")
        self.logger.info(f"- Default Stop Loss (pips): {self.stop_loss_pips}")
        self.logger.info(f"- Default Take Profit (pips): {self.take_profit_pips}")
        self.logger.info(f"- Use ATR Sizing: {self.use_atr_sizing}")
        if self.use_atr_sizing:
            self.logger.info(f"- ATR Period: {self.atr_period}")
            self.logger.info(f"- SL ATR Multiplier: {self.stop_loss_atr_multiplier}")
        # Live conversion helper
        self.account_currency = (config.get('account_currency') or 'USD').upper()
        self.cur_mgr = CurrencyDependencyManager(self.account_currency, apply_oanda_markup=True)
        self.iim = InstrumentInfoManager()

    def update_params(self, params: Dict[str, Any]):
        """
        Update risk parameters at runtime from a dictionary. Only known keys are applied.
        """
        if not params:
            return
        self.config.update(params)
        # Apply supported keys to attributes
        for key in [
            'max_total_exposure_usd',
            'max_risk_per_trade_percent',
            'use_atr_sizing',
            'atr_period',
            'stop_loss_atr_multiplier',
            'take_profit_atr_multiplier',
            'stop_loss_pips',
            'take_profit_pips',
        ]:
            if key in params and params[key] is not None:
                setattr(self, key, params[key])
        # Log summary of changes
        try:
            self.logger.info(f"Risk parameters updated: { {k:v for k,v in params.items() if v is not None} }")
        except Exception:
            pass

    def _build_price_map(self, instrument: str, oanda_client) -> Dict[str, tuple]:
        sym = instrument
        parts = sym.split('_')
        base, quote = parts[0], parts[1]
        mp: Dict[str, tuple] = {}
        recs = oanda_client.get_bid_ask_candles_combined(sym, count=1) or []
        if recs:
            r = recs[-1]
            mp[sym] = (Decimal(str(r.get('bid_close', 0.0))), Decimal(str(r.get('ask_close', 0.0))))
        # add conversion pairs for quote/account and base/account as needed
        for a, b in [(quote, self.account_currency), (self.account_currency, quote), (base, self.account_currency), (self.account_currency, base)]:
            pair = f"{a}_{b}"
            if pair not in mp:
                rr = oanda_client.get_bid_ask_candles_combined(pair, count=1) or []
                if rr:
                    last = rr[-1]
                    mp[pair] = (Decimal(str(last.get('bid_close', 0.0))), Decimal(str(last.get('ask_close', 0.0))))
        return mp

    def assess_trade(self, instrument: str, signal: int, price: float) -> Optional[Dict[str, Any]]:
        """
        Assesses a potential trade against risk management rules.

        Args:
            instrument (str): The instrument to trade.
            signal (int): The trading signal (1 for buy, -1 for sell).
            price (float): The current market price.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with order details (units, stop_loss, take_profit)
                                      if the trade is approved, otherwise None.
        """
        self.logger.info(f"Assessing trade for {instrument} with signal {signal} at price {price}")

        # 1. Check for existing positions
        existing_position = self.position_manager.get_position(instrument)
        if existing_position:
            # Rule: Do not open a new position if one already exists for the same instrument.
            # More complex logic (e.g., adding to a position) can be added later.
            if (signal == 1 and existing_position.position_type == 'long') or \
               (signal == -1 and existing_position.position_type == 'short'):
                self.logger.warning(f"Signal in the same direction as existing position for {instrument}. No action taken.")
                return None
            # If the signal is opposite, it implies closing the existing position, which is handled by OrderManager.
            # We don't open a new one immediately.

        # 2. Calculate order size based on risk. Prefer ATR sizing if enabled and available; otherwise fallback to pips.
        try:
            # Equity from account summary
            from ..core.oanda_client import OandaClient
            client = OandaClient.from_env()
            acct = client.get_account_summary()
            equity = float(acct['account'].get('NAV', acct['account'].get('balance', 0.0))) if acct and 'account' in acct else 0.0
        except Exception:
            equity = 0.0

        details = self.iim.get_details(instrument)
        if details is None:
            self.logger.error(f"Instrument details not found for {instrument}.")
            return None

        # Build price map for conversions and ATR computation
        price_map = self._build_price_map(instrument, client)
        rate_qc_to_ac = float(self.cur_mgr.convert_to_account_currency(details.quote_currency, price_map))
        if rate_qc_to_ac <= 0:
            self.logger.error("Failed to compute quote->account currency rate; aborting trade.")
            return None

        risk_amount_ac = (self.max_risk_per_trade_percent / 100.0) * equity if equity > 0 else 10.0
        trade_size_units = 0
        stop_loss_price = None
        take_profit_price = None

        def _compute_atr(instrument: str, count: int = None) -> float:
            cnt = max(self.atr_period + 2, 20) if count is None else count
            recs = client.get_bid_ask_candles_combined(instrument, count=cnt) or []
            if len(recs) < self.atr_period + 1:
                return 0.0
            highs = []
            lows = []
            closes = []
            for r in recs:
                h = (float(r.get('bid_high', 0.0)) + float(r.get('ask_high', 0.0))) / 2.0
                l = (float(r.get('bid_low', 0.0)) + float(r.get('ask_low', 0.0))) / 2.0
                c = (float(r.get('bid_close', 0.0)) + float(r.get('ask_close', 0.0))) / 2.0
                highs.append(h); lows.append(l); closes.append(c)
            trs = []
            for i in range(1, len(highs)):
                tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
                trs.append(tr)
            if len(trs) < self.atr_period:
                return 0.0
            atr = sum(trs[-self.atr_period:]) / float(self.atr_period)
            return atr

        used_atr = False
        if self.use_atr_sizing and self.stop_loss_atr_multiplier > 0:
            atr_val = _compute_atr(instrument)
            if atr_val > 0:
                sl_distance_qc = atr_val * self.stop_loss_atr_multiplier
                risk_per_unit_ac = sl_distance_qc * rate_qc_to_ac
                if risk_per_unit_ac > 0:
                    units_float = risk_amount_ac / risk_per_unit_ac
                    units_decimal = details.round_units(units_float)
                    trade_size_units = int(units_decimal) if signal == 1 else -int(units_decimal)
                    # SL by ATR distance
                    if signal == 1:
                        stop_loss_price = price - sl_distance_qc
                    elif signal == -1:
                        stop_loss_price = price + sl_distance_qc
                    if self.take_profit_atr_multiplier and self.take_profit_atr_multiplier > 0:
                        tp_distance_qc = atr_val * self.take_profit_atr_multiplier
                        if signal == 1:
                            take_profit_price = price + tp_distance_qc
                        elif signal == -1:
                            take_profit_price = price - tp_distance_qc
                    used_atr = True

        if not used_atr:
            # Fallback: pip-based sizing
            pip_value_qc_per_unit = abs(float(Decimal(str(10)) ** Decimal(str(details.pip_location))))
            pip_value_ac_per_unit = pip_value_qc_per_unit * rate_qc_to_ac
            if pip_value_ac_per_unit <= 0:
                self.logger.error("Failed to compute pip value in account currency; aborting trade.")
                return None
            units_float = risk_amount_ac / (self.stop_loss_pips * pip_value_ac_per_unit)
            units_decimal = details.round_units(units_float)
            trade_size_units = int(units_decimal) if signal == 1 else -int(units_decimal)
            if signal == 1:
                stop_loss_price = price - self.stop_loss_pips * (10 ** details.pip_location)
            elif signal == -1:
                stop_loss_price = price + self.stop_loss_pips * (10 ** details.pip_location)

        # Take profitÃ¯Â¼Å¡Ã¤Â¿ÂÃ§â€¢â„¢ pip Ã¦Â¨Â¡Ã¥Â¼ÂÃ¯Â¼Ë†Ã¥ÂÂ¯Ã¦â€“Â¼Ã¦Å“ÂªÃ¤Â¾â€ Ã¦â€œÂ´Ã¥â€¦â€¦ ATR TPÃ¯Â¼â€°
        if signal == 1: # Buy
            take_profit_price = price + self.take_profit_pips * (10 ** details.pip_location)
        elif signal == -1:
            take_profit_price = price - self.take_profit_pips * (10 ** details.pip_location)
        else:
            return None

        # Cap by max_total_exposure_usd
        try:
            notional_qc = abs(trade_size_units) * price
            notional_ac = notional_qc * rate_qc_to_ac
            if self.max_total_exposure_usd and notional_ac > float(self.max_total_exposure_usd):
                scale = float(self.max_total_exposure_usd) / max(1e-9, notional_ac)
                scaled_units = abs(trade_size_units) * scale
                # Round using broker rules
                units_decimal = details.round_units(scaled_units)
                trade_size_units = int(units_decimal) if trade_size_units >= 0 else -int(units_decimal)
                self.logger.info(f"Capped trade size by exposure: new units {trade_size_units} for {instrument}.")
        except Exception:
            pass

        # Safety: ensure non-zero units for minimal live execution if sizing rounded to 0
        if trade_size_units == 0:
            try:
                min_units = int(getattr(details, 'minimum_trade_size', 1))
            except Exception:
                min_units = 1
            min_units = max(1, min_units)
            trade_size_units = min_units if signal == 1 else -min_units
            self.logger.info(f"Risk sizing rounded to zero; using minimum trade size {trade_size_units} for {instrument}.")

        # 4. Final check (placeholder for more complex checks like total exposure)
        # In a real system, you would check if this new trade exceeds max_total_exposure_usd.

        self.logger.info(f"Trade approved for {instrument}. Units: {trade_size_units}, SL: {stop_loss_price}, TP: {take_profit_price}")

        return {
            "units": trade_size_units if signal == 1 else -trade_size_units,
            "stop_loss": round(stop_loss_price, 5) if stop_loss_price is not None else None,
            "take_profit": round(take_profit_price, 5) if take_profit_price is not None else None
        }
