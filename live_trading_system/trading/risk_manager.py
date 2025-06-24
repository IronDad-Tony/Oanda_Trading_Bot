import logging
from typing import Dict, Any, Optional

from live_trading_system.trading.position_manager import PositionManager, Position
from live_trading_system.core.system_state import SystemState

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

        # 2. Calculate order size
        # This is a simplified calculation. A real system would need account balance and currency conversion.
        # For now, we use a fixed size or a simple calculation.
        trade_size_units = self.config.get('default_trade_size_units', 100)

        # 3. Define Stop Loss and Take Profit
        pip_value = 0.0001 # Simplified for now, varies by instrument
        if signal == 1: # Buy
            stop_loss_price = price - self.stop_loss_pips * pip_value
            take_profit_price = price + self.take_profit_pips * pip_value
        elif signal == -1: # Sell
            stop_loss_price = price + self.stop_loss_pips * pip_value
            take_profit_price = price - self.take_profit_pips * pip_value
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
