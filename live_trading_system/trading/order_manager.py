import logging
from typing import Dict, Any

from live_trading_system.core.oanda_client import OandaClient
from live_trading_system.trading.position_manager import PositionManager
from live_trading_system.core.system_state import SystemState
from live_trading_system.database.database_manager import DatabaseManager
from live_trading_system.trading.risk_manager import RiskManager


class OrderManager:
    """
    Manages the full lifecycle of an order: risk assessment, execution,
    and position tracking.
    """

    def __init__(
        self,
        client: OandaClient,
        system_state: SystemState,
        position_manager: PositionManager,
        risk_manager: RiskManager,
        db_manager: DatabaseManager,  # Added db_manager
    ):
        """
        Initializes the OrderManager.

        Args:
            client: The OandaClient instance for API interaction.
            system_state: The SystemState instance.
            position_manager: The manager for tracking open positions.
            risk_manager: The manager for assessing trade risk.
            db_manager: The manager for database operations.
        """
        self.logger = logging.getLogger(__name__)
        self.client = client
        self.system_state = system_state
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.db_manager = db_manager  # Store db_manager instance
        self.logger.info("OrderManager initialized (Full Implementation).")

    def process_signal(self, signal_info: Dict[str, Any]):
        """
        Processes a trading signal, assesses risk, and places an order if approved.

        Args:
            signal_info (Dict[str, Any]): Signal details from TradingLogic.
        """
        instrument = signal_info.get("instrument")
        signal = signal_info.get("signal")
        price = signal_info.get("price")

        self.logger.info(f"Processing signal for {instrument} at price {price} with signal {signal}.")

        # 1. Assess the trade with the Risk Manager
        trade_details = self.risk_manager.assess_trade(instrument, signal, price)

        if not trade_details:
            self.logger.info(f"Trade for {instrument} was not approved by RiskManager.")
            return

        # 2. If trade is approved, create the order
        self.logger.info(f"Trade approved. Preparing to place order for {instrument}.")
        
        order_response = self.client.create_order(
            instrument=instrument,
            units=trade_details["units"],
            stop_loss=trade_details["stop_loss"],
            take_profit=trade_details["take_profit"],
        )

        # 3. If order is filled, update the Position Manager
        if order_response and "orderFillTransaction" in order_response:
            fill_details = order_response["orderFillTransaction"]
            executed_price = float(fill_details.get("price", 0))
            executed_units = int(fill_details.get("units", 0))
            
            self.logger.info(f"Order for {instrument} filled at {executed_price} for {executed_units} units.")
            
            self.position_manager.update_position(
                instrument=instrument,
                units=executed_units,
                price=executed_price,
            )
        elif order_response and "orderCancelTransaction" in order_response:
             self.logger.warning(f"Order for {instrument} was cancelled. Reason: {order_response['orderCancelTransaction'].get('reason')}")
        else:
            self.logger.error(f"Order execution failed or no fill details received for {instrument}. Response: {order_response}")


    def close_all_positions(self):
        """
        Emergency function to close all open positions.
        """
        self.logger.warning("EMERGENCY: Closing all open positions.")
        all_positions = self.position_manager.get_all_positions()
        if not all_positions:
            self.logger.info("No open positions to close.")
            return

        for instrument, position in all_positions.items():
            self.logger.info(f"Closing position for {instrument}...")
            # Determine units to close (negative for long, positive for short)
            units_to_close = -position.units if position.position_type == 'long' else position.units
            
            # Create a market order to close the position
            order_response = self.client.create_order(instrument, units_to_close)

            if order_response and "orderFillTransaction" in order_response:
                self.logger.info(f"Successfully closed position for {instrument}.")
                self.position_manager.close_position(instrument)
            else:
                self.logger.error(f"Failed to close position for {instrument}. Response: {order_response}")
