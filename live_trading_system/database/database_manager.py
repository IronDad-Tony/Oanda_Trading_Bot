import sqlite3
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

class DatabaseManager:
    """
    Manages all database operations, including storing trades, logs, and performance metrics.
    Uses SQLite for simplicity and portability.
    """
    def __init__(self, db_path: str):
        """
        Initializes the DatabaseManager and creates necessary tables.

        Args:
            db_path (str): The file path for the SQLite database.
        """
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            # Return rows as dictionaries
            self.conn.row_factory = sqlite3.Row
            self.logger.info(f"Successfully connected to database at {self.db_path}")
            self._create_tables()
        except sqlite3.Error as e:
            self.logger.critical(f"Database connection failed: {e}", exc_info=True)

    def _create_tables(self):
        """
        Creates the database tables if they do not already exist.
        The 'trades' table is designed to track the full lifecycle of a trade.
        """
        if not self.conn:
            return
        try:
            cursor = self.conn.cursor()
            # Table for tracking individual trades from open to close
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    oanda_trade_id TEXT UNIQUE NOT NULL,
                    instrument TEXT NOT NULL,
                    units INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    signal TEXT,
                    status TEXT NOT NULL, -- 'OPEN', 'CLOSED'
                    pnl REAL,
                    open_timestamp TEXT NOT NULL,
                    close_timestamp TEXT,
                    close_price REAL
                )
            """)
            self.conn.commit()
            self.logger.info("Database 'trades' table verified/created successfully.")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create tables: {e}", exc_info=True)

    def log_trade(self, trade_id: str, instrument: str, units: int, entry_price: float, signal: Any, status: str):
        """
        Logs a new trade when it is opened.

        Args:
            trade_id (str): The OANDA trade ID.
            instrument (str): The instrument being traded.
            units (int): The number of units (positive for long, negative for short).
            entry_price (float): The price at which the trade was executed.
            signal (Any): The signal that triggered the trade.
            status (str): The initial status of the trade (e.g., 'OPEN').
        """
        if not self.conn:
            return
        sql = """INSERT INTO trades (oanda_trade_id, instrument, units, entry_price, signal, status, open_timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?)"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, (
                trade_id,
                instrument,
                units,
                entry_price,
                str(signal), # Ensure signal is stored as text
                status,
                datetime.utcnow().isoformat()
            ))
            self.conn.commit()
            self.logger.info(f"Logged new trade {trade_id} for {instrument} to database.")
        except sqlite3.IntegrityError:
            self.logger.warning(f"Trade with OANDA ID {trade_id} already exists in the database.")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to log trade {trade_id}: {e}", exc_info=True)

    def update_trade_on_close(self, oanda_trade_id: str, close_price: float, pnl: float):
        """
        Updates a trade record when it is closed.

        Args:
            oanda_trade_id (str): The OANDA ID of the trade to update.
            close_price (float): The price at which the trade was closed.
            pnl (float): The realized profit or loss for the trade.
        """
        if not self.conn:
            return
        sql = """UPDATE trades
                 SET status = 'CLOSED', close_price = ?, pnl = ?, close_timestamp = ?
                 WHERE oanda_trade_id = ?"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, (
                close_price,
                pnl,
                datetime.utcnow().isoformat(),
                oanda_trade_id
            ))
            self.conn.commit()
            if cursor.rowcount > 0:
                self.logger.info(f"Updated closed trade {oanda_trade_id} in database with PnL: {pnl}.")
            else:
                self.logger.warning(f"Attempted to update non-existent trade with OANDA ID {oanda_trade_id}.")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to update closed trade {oanda_trade_id}: {e}", exc_info=True)

    def get_trade_by_oanda_id(self, oanda_trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single trade by its OANDA trade ID.
        """
        if not self.conn:
            return None
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE oanda_trade_id = ?", (oanda_trade_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            self.logger.error(f"Failed to fetch trade by OANDA ID {oanda_trade_id}: {e}", exc_info=True)
            return None

    def get_last_trade_for_instrument(self, instrument: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the most recently opened trade for a given instrument.
        """
        if not self.conn:
            return None
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE instrument = ? ORDER BY open_timestamp DESC LIMIT 1", (instrument,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            self.logger.error(f"Failed to fetch last trade for {instrument}: {e}", exc_info=True)
            return None

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed.")
