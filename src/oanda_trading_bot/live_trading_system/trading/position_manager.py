import logging
from typing import Dict, Optional

class Position:
    """Represents a single position for an instrument."""
    def __init__(self, instrument: str, position_type: str, units: int, price: float, trade_id: str):
        self.instrument = instrument
        self.position_type = position_type  # 'long' or 'short'
        self.units = units
        self.entry_price = price
        self.trade_id = trade_id # OANDA Trade ID

    def __repr__(self):
        return f"Position(instrument={self.instrument}, trade_id={self.trade_id}, type={self.position_type}, units={self.units}, entry_price={self.entry_price})"

class PositionManager:
    """
    Manages and tracks all open positions across all instruments.
    This is an in-memory representation and should be synchronized with the broker.
    """
    def __init__(self):
        """Initializes the PositionManager."""
        self.logger = logging.getLogger(__name__)
        self.positions: Dict[str, Position] = {}
        self.logger.info("PositionManager initialized.")

    def update_position(self, instrument: str, units: int, price: float, trade_id: str):
        """
        Updates or creates a position based on a new trade.

        Args:
            instrument (str): The instrument being traded.
            units (int): The number of units traded. Positive for long, negative for short.
            price (float): The price at which the trade was executed.
            trade_id (str): The OANDA trade ID for the position.
        """
        if instrument in self.positions:
            # Logic for updating an existing position (e.g., pyramiding) can be complex.
            # For now, we log a warning as the current strategy is one position per instrument.
            self.logger.warning(f"Received update for an already existing position on {instrument}. Current strategy does not support pyramiding. Ignoring update.")
            return

        if units > 0:
            self.positions[instrument] = Position(instrument, 'long', units, price, trade_id)
            self.logger.info(f"New long position opened: {self.positions[instrument]}")
        elif units < 0:
            self.positions[instrument] = Position(instrument, 'short', abs(units), price, trade_id)
            self.logger.info(f"New short position opened: {self.positions[instrument]}")

    def close_position(self, instrument: str):
        """
        Removes a position from the in-memory store when it is closed.

        Args:
            instrument (str): The instrument whose position is to be closed.
        """
        if instrument in self.positions:
            closed_position = self.positions.pop(instrument)
            self.logger.info(f"Position removed from internal tracking: {closed_position}")
        else:
            self.logger.warning(f"Attempted to close a non-existent internal position for {instrument}.")

    def get_position(self, instrument: str) -> Optional[Position]:
        """
        Retrieves the current position for a given instrument.

        Args:
            instrument (str): The instrument to check.

        Returns:
            Optional[Position]: The Position object if it exists, otherwise None.
        """
        return self.positions.get(instrument)

    def get_all_positions(self) -> Dict[str, Position]:
        """
        Retrieves all open positions from the internal store.

        Returns:
            Dict[str, Position]: A dictionary of all open positions.
        """
        return self.positions
