
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
            self.logger.info(f"Successfully connected to database at {self.db_path}")
            self._create_tables()
        except sqlite3.Error as e:
            self.logger.critical(f"Database connection failed: {e}", exc_info=True)

    def _create_tables(self):
        """
        Creates the database tables if they do not already exist.
        """
        if not self.conn:
            return
        try:
            cursor = self.conn.cursor()
            # Table for all trade activities (open, close)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    instrument TEXT NOT NULL,
                    action TEXT NOT NULL, -- e.g., 'OPEN', 'CLOSE'
                    units INTEGER NOT NULL,
                    price REAL NOT NULL,
                    details TEXT -- JSON string for extra details like SL/TP
                )
            """)
            # Table for system events and logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL, -- e.g., INFO, WARNING, ERROR
                    message TEXT NOT NULL
                )
            """)
            self.conn.commit()
            self.logger.info("Database tables verified/created successfully.")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create tables: {e}", exc_info=True)

    def save_trade(self, trade_info: Dict[str, Any]):
        """
        Saves a trade record to the database.

        Args:
            trade_info (Dict[str, Any]): A dictionary containing trade details.
                                         e.g., {'instrument', 'action', 'units', 'price', 'details'}
        """
        if not self.conn:
            return
        sql = """INSERT INTO trades (timestamp, instrument, action, units, price, details)
                 VALUES (?, ?, ?, ?, ?, ?)"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, (
                datetime.utcnow().isoformat(),
                trade_info.get('instrument'),
                trade_info.get('action'),
                trade_info.get('units'),
                trade_info.get('price'),
                json.dumps(trade_info.get('details', {}))
            ))
            self.conn.commit()
            self.logger.debug(f"Saved trade to database: {trade_info}")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to save trade: {e}", exc_info=True)

    def save_log(self, level: str, message: str):
        """
        Saves a system log message to the database.

        Args:
            level (str): The log level (e.g., 'INFO', 'ERROR').
            message (str): The log message.
        """
        if not self.conn:
            return
        sql = """INSERT INTO system_logs (timestamp, level, message)
                 VALUES (?, ?, ?)"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, (datetime.utcnow().isoformat(), level, message))
            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Failed to save log to database: {e}", exc_info=True)

    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent trade history.

        Args:
            limit (int): The maximum number of records to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of trade records.
        """
        if not self.conn:
            return []
        sql = "SELECT timestamp, instrument, action, units, price, details FROM trades ORDER BY id DESC LIMIT ?"
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
            trades = []
            for row in rows:
                trades.append({
                    "timestamp": row[0],
                    "instrument": row[1],
                    "action": row[2],
                    "units": row[3],
                    "price": row[4],
                    "details": json.loads(row[5])
                })
            return trades
        except sqlite3.Error as e:
            self.logger.error(f"Failed to retrieve trade history: {e}", exc_info=True)
            return []

    def close(self):
        """
        Closes the database connection.
        """
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed.")
