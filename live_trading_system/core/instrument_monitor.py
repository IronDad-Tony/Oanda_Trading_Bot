import logging
import time
import threading
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from live_trading_system.core.oanda_client import OandaClient
from live_trading_system.core.system_state import SystemState

class InstrumentMonitor:
    """
    Monitors a list of instruments in the background to dynamically select the
    best one for trading based on a volatility metric (e.g., ATR).
    """
    def __init__(self, client: OandaClient, system_state: SystemState, config: Dict[str, Any]):
        """
        Initializes the InstrumentMonitor.

        Args:
            client (OandaClient): The Oanda API client.
            system_state (SystemState): The global system state manager.
            config (Dict[str, Any]): The main system configuration.
        """
        self.logger = logging.getLogger(__name__)
        self.client = client
        self.system_state = system_state
        self.config = config.get('instrument_monitor', {})
        
        self.monitor_list: List[str] = self.config.get('instruments', [])
        self.interval: int = self.config.get('interval_seconds', 300)
        self.volatility_period: int = self.config.get('volatility_period', 20)
        self.granularity: str = self.config.get('granularity', "M5")
        
        self.monitor_thread: Optional[threading.Thread] = None
        self.logger.info(f"InstrumentMonitor initialized. Monitoring: {self.monitor_list}")

    def _calculate_volatility(self, candles: List[Dict[str, Any]]) -> float:
        """
        Calculates Average True Range (ATR) as the volatility metric.

        Args:
            candles (List[Dict[str, Any]]): A list of candle data from Oanda.

        Returns:
            float: The calculated ATR value.
        """
        if len(candles) < self.volatility_period:
            self.logger.warning(f"Not enough candles to calculate ATR (need {self.volatility_period}, got {len(candles)}).")
            return 0.0
        
        try:
            df = pd.DataFrame([{
                'h': float(c['mid']['h']),
                'l': float(c['mid']['l']),
                'c': float(c['mid']['c'])
            } for c in candles])

            df['tr1'] = df['h'] - df['l']
            df['tr2'] = abs(df['h'] - df['c'].shift(1))
            df['tr3'] = abs(df['l'] - df['c'].shift(1))
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            atr = df['tr'].rolling(window=self.volatility_period, min_periods=self.volatility_period).mean().iloc[-1]
            return atr if pd.notna(atr) else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}", exc_info=True)
            return 0.0

    def select_best_instrument(self) -> Optional[str]:
        """
        Fetches data for all monitored instruments and selects the one with the highest volatility.

        Returns:
            Optional[str]: The instrument with the highest volatility, or None if none could be assessed.
        """
        volatilities = {}
        for instrument in self.monitor_list:
            # Fetch enough candles for calculation
            candles = self.client.get_candles(instrument, count=self.volatility_period + 1, granularity=self.granularity)
            if candles:
                volatility = self._calculate_volatility(candles)
                volatilities[instrument] = volatility
                self.logger.debug(f"Instrument: {instrument}, Volatility (ATR): {volatility:.5f}")
            else:
                self.logger.warning(f"Could not fetch candles for {instrument} during selection process.")

        if not volatilities:
            self.logger.error("Could not calculate volatility for any instrument in the monitor list.")
            return None

        # Select instrument with the highest volatility
        best_instrument = max(volatilities, key=volatilities.get)
        self.logger.info(f"Best instrument selected: {best_instrument} with volatility {volatilities[best_instrument]:.5f}")
        return best_instrument

    def _monitor_loop(self):
        """
        The main loop for the monitoring thread. It periodically selects the best instrument
        and updates the system state.
        """
        self.logger.info("Instrument monitor thread started.")
        while self.system_state.is_running():
            self.logger.info("Running instrument monitor cycle...")
            best_instrument = self.select_best_instrument()
            
            if best_instrument and best_instrument != self.system_state.get_current_instrument():
                self.logger.info(f"Switching active instrument from '{self.system_state.get_current_instrument()}' to '{best_instrument}'")
                self.system_state.set_current_instrument(best_instrument)
            
            # Wait for the next interval
            time.sleep(self.interval)
        self.logger.info("Instrument monitor thread has stopped.")

    def start(self):
        """
        Starts the monitoring process in a background thread if it's not already running.
        """
        if not self.monitor_list:
            self.logger.warning("No instruments to monitor. InstrumentMonitor will not start.")
            return

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.logger.warning("Monitor thread is already running.")
            return

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
