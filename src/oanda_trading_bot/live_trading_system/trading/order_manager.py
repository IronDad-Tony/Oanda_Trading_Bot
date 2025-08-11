import logging
from typing import Dict, Any

from .core.oanda_client import OandaClient
from .trading.position_manager import PositionManager
from .core.system_state import SystemState
from .database.database_manager import DatabaseManager
from .trading.risk_manager import RiskManager


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
        db_manager: DatabaseManager,
    ):
        # 添加参数验证
        if not all([client, system_state, position_manager, risk_manager, db_manager]):
            raise ValueError("OrderManager missing required dependencies")
        
        self.logger = logging.getLogger(__name__)
        self.client = client
        self.system_state = system_state
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.db_manager = db_manager
        self.logger.info("OrderManager initialized.")

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
            self.logger.warning(f"Trade for {instrument} was not approved by RiskManager.")
            return

        # 2. If trade is approved, create the order
        self.logger.info(f"Trade approved. Preparing to place order for {instrument}.")
        
        # Map risk manager output to OANDA API format
        stop_loss_on_fill = {"price": str(trade_details["stop_loss"])} if trade_details.get("stop_loss") else None
        take_profit_on_fill = {"price": str(trade_details["take_profit"])} if trade_details.get("take_profit") else None

        order_response = self.client.create_order(
            instrument=instrument,
            units=trade_details["units"],
            stop_loss_on_fill=stop_loss_on_fill,
            take_profit_on_fill=take_profit_on_fill,
        )

        # 3. If order is filled, update the Position Manager and log to DB
        if order_response and "orderFillTransaction" in order_response:
            fill_details = order_response["orderFillTransaction"]
            self.logger.debug(f"Full orderFillTransaction response: {fill_details}")

            trade_id = None
            # Correctly parse the tradeID from the 'tradeOpened' dictionary
            if "tradeOpened" in fill_details and isinstance(fill_details["tradeOpened"], dict):
                trade_id = fill_details["tradeOpened"].get("tradeID")

            if not trade_id:
                self.logger.error("CRITICAL: Could not extract tradeID from orderFillTransaction.")
                # Depending on strictness, we might want to stop or handle this error
                return 

            executed_price = float(fill_details.get("price", 0))
            executed_units = int(fill_details.get("units", 0))
            
            self.logger.info(f"Order for {instrument} filled. Trade ID: {trade_id}, Price: {executed_price}, Units: {executed_units}.")
            
            # Update internal state
            self.position_manager.update_position(
                instrument=instrument,
                units=executed_units,
                price=executed_price,
                trade_id=trade_id
            )
            
            # Log trade to database
            self.db_manager.log_trade(
                trade_id=trade_id,
                instrument=instrument,
                units=executed_units,
                entry_price=executed_price,
                signal=signal,
                status="OPEN"
            )

        elif order_response and "orderCancelTransaction" in order_response:
             self.logger.warning(f"Order for {instrument} was cancelled. Reason: {order_response['orderCancelTransaction'].get('reason')}")
        else:
            self.logger.error(f"Order execution failed or no fill details received for {instrument}. Response: {order_response}")

    def close_all_positions(self):
        """
        Closes all open positions and updates the database with the final PnL.
        """
        self.logger.warning("EMERGENCY: Closing all open positions.")
        open_positions_data = self.client.get_open_positions()
        
        if not open_positions_data or not open_positions_data.get("positions"):
            self.logger.info("No open positions found on OANDA to close.")
            return

        for position in open_positions_data["positions"]:
            instrument = position["instrument"]
            internal_pos = self.position_manager.get_position(instrument)
            if not internal_pos:
                self.logger.warning(f"Found position for {instrument} on OANDA but not in internal state. Skipping DB update for this one.")
                # Decide if you still want to close it without internal tracking
                # For safety, we will proceed to close it.

            long_units = position.get("long", {}).get("units", "0")
            short_units = position.get("short", {}).get("units", "0")

            response = None
            if long_units != "0":
                self.logger.info(f"Closing long position for {instrument} ({long_units} units)...")
                response = self.client.close_position(instrument, long_units="ALL")
            elif short_units != "0":
                self.logger.info(f"Closing short position for {instrument} ({short_units} units)...")
                response = self.client.close_position(instrument, short_units="ALL")

            if response:
                # Extract the closing transaction details
                fill_transaction = response.get('longOrderFillTransaction') or response.get('shortOrderFillTransaction')
                if fill_transaction and fill_transaction.get('tradesClosed'):
                    self.logger.info(f"Successfully submitted close order for {instrument}.")
                    
                    # Process each closed trade within the transaction
                    for closed_trade in fill_transaction.get('tradesClosed', []):
                        trade_id = closed_trade.get("tradeID")
                        price = float(closed_trade.get("price", 0))
                        pnl = float(closed_trade.get("realizedPL", 0))
                        
                        self.logger.info(f"Trade {trade_id} closed. Close Price: {price}, PnL: {pnl}")
                        self.db_manager.update_trade_on_close(trade_id, price, pnl)

                    # Remove from internal state after processing all parts of the closure
                    self.position_manager.close_position(instrument)
                else:
                    self.logger.error(f"Failed to get fill details when closing {instrument}. Response: {response}")
            else:
                self.logger.error(f"Failed to receive response when closing position for {instrument}.")
