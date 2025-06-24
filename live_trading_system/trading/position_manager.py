import logging
from typing import Dict, Optional

from live_trading_system.database.database_manager import DatabaseManager
from live_trading_system.core.system_state import SystemState

class Position:
    """Represents a single position for an instrument."""
    def __init__(self, instrument: str, position_type: str, units: int, price: float):
        self.instrument = instrument
        self.position_type = position_type  # 'long' or 'short'
        self.units = units
        self.entry_price = price

    def __repr__(self):
        return f"Position(instrument={self.instrument}, type={self.position_type}, units={self.units}, entry_price={self.entry_price})"

class PositionManager:
    """
    Manages and tracks all open positions across all instruments.
    """
    def __init__(self):
        """Initializes the PositionManager."""
        self.logger = logging.getLogger(__name__)
        self.positions: Dict[str, Position] = {}
        self.logger.info("PositionManager initialized.")

    def update_position(self, instrument: str, units: int, price: float):
        """
        Updates or creates a position based on a new trade.

        Args:
            instrument (str): The instrument being traded.
            units (int): The number of units traded. Positive for long, negative for short.
            price (float): The price at which the trade was executed.
        """
        if instrument not in self.positions:
            if units > 0:
                self.positions[instrument] = Position(instrument, 'long', units, price)
                self.logger.info(f"New long position opened: {self.positions[instrument]}")
            elif units < 0:
                self.positions[instrument] = Position(instrument, 'short', abs(units), price)
                self.logger.info(f"New short position opened: {self.positions[instrument]}")
        else:
            # Logic for updating an existing position (e.g., averaging down) can be complex.
            # For now, we assume one position per instrument. We will close and reopen.
            # A more sophisticated implementation would handle adding to/reducing positions.
            self.logger.warning(f"Update on existing position for {instrument} not yet implemented. Treating as new.")
            # This part will be refined later. For now, we just log it.
            # In a real scenario, you might close the old and open a new one or adjust the existing.

    def close_position(self, instrument: str):
        """
        Removes a position when it is closed.

        Args:
            instrument (str): The instrument whose position is to be closed.
        """
        if instrument in self.positions:
            closed_position = self.positions.pop(instrument)
            self.logger.info(f"Position closed: {closed_position}")
        else:
            self.logger.warning(f"Attempted to close a non-existent position for {instrument}.")

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
        Retrieves all open positions.

        Returns:
            Dict[str, Position]: A dictionary of all open positions.
        """
        return self.positions
