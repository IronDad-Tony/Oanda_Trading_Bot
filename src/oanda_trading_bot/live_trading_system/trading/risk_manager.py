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
        self.stop_loss_pips = self.config.get('stop_loss_pips', 20)
        self.take_profit_pips = self.config.get('take_profit_pips', 40)
        
        self.logger.info("RiskManager initialized with the following parameters:")
        self.logger.info(f"- Max Total Exposure (USD): {self.max_total_exposure_usd}")
        self.logger.info(f"- Max Risk Per Trade: {self.max_risk_per_trade_percent}%")
        self.logger.info(f"- Default Stop Loss (pips): {self.stop_loss_pips}")
        self.logger.info(f"- Default Take Profit (pips): {self.take_profit_pips}")
        # Live conversion helper
        self.account_currency = (config.get('account_currency') or 'USD').upper()
        self.cur_mgr = CurrencyDependencyManager(self.account_currency, apply_oanda_markup=True)
        self.iim = InstrumentInfoManager()

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

        # 2. Calculate order size based on risk and pip value in account currency
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

        # pip value per unit in quote currency
        pip_value_qc_per_unit = abs(float(Decimal(str(10)) ** Decimal(str(details.pip_location))))
        # conversion to account currency using live prices
        price_map = self._build_price_map(instrument, client)
        rate_qc_to_ac = float(self.cur_mgr.convert_to_account_currency(details.quote_currency, price_map))
        pip_value_ac_per_unit = pip_value_qc_per_unit * rate_qc_to_ac if rate_qc_to_ac > 0 else 0.0
        if pip_value_ac_per_unit <= 0:
            self.logger.error("Failed to compute pip value in account currency; aborting trade.")
            return None

        risk_amount_ac = (self.max_risk_per_trade_percent / 100.0) * equity if equity > 0 else 10.0
        units_float = risk_amount_ac / (self.stop_loss_pips * pip_value_ac_per_unit)
        # Round units per instrument rules
        units_decimal = details.round_units(units_float)
        trade_size_units = int(units_decimal) if signal == 1 else -int(units_decimal)

        # 3. Define Stop Loss and Take Profit
        if signal == 1: # Buy
            stop_loss_price = price - self.stop_loss_pips * (10 ** details.pip_location)
            take_profit_price = price + self.take_profit_pips * (10 ** details.pip_location)
        elif signal == -1: # Sell
            stop_loss_price = price + self.stop_loss_pips * (10 ** details.pip_location)
            take_profit_price = price - self.take_profit_pips * (10 ** details.pip_location)
        else: # Hold
            return None

        # 4. Final check (placeholder for more complex checks like total exposure)
        # In a real system, you would check if this new trade exceeds max_total_exposure_usd.

        self.logger.info(f"Trade approved for {instrument}. Units: {trade_size_units}, SL: {stop_loss_price}, TP: {take_profit_price}")

        return {
            "units": trade_size_units if signal == 1 else -trade_size_units,
            "stop_loss": round(stop_loss_price, 5),
            "take_profit": round(take_profit_price, 5)
        }
