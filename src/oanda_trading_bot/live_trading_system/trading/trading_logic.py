import logging
import time
import collections
from typing import Dict, Any, List

from .core.oanda_client import OandaClient
from .data.data_preprocessor import LivePreprocessor
from .model.prediction_service import PredictionService
from .core.system_state import SystemState
from .trading.order_manager import OrderManager

class TradingLogic:
    """
    The core decision-making engine of the trading system.

    This class orchestrates the process of fetching data, getting predictions,
    and sending signals to the OrderManager. It handles the initial data
    window filling ("cold start") and subsequent real-time updates.
    """
    def __init__(
        self, 
        config: Dict[str, Any],
        client: OandaClient, 
        system_state: SystemState, 
        preprocessor: LivePreprocessor,
        prediction_service: PredictionService,
        order_manager: OrderManager
    ):
        """
        Initializes the TradingLogic component.

        Args:
            config (Dict[str, Any]): The system\'s configuration.
            client (OandaClient): The client for interacting with the Oanda API.
            system_state (SystemState): The global state manager.
            preprocessor (LivePreprocessor): The data preprocessing service.
            prediction_service (PredictionService): The model prediction service.
            order_manager (OrderManager): The order execution manager.
        """
        self.logger = logging.getLogger("LiveTradingSystem")
        self.config = config
        self.client = client
        self.system_state = system_state
        self.preprocessor = preprocessor
        self.prediction_service = prediction_service
        self.order_manager = order_manager
        
        self.granularity = self.config.get("trading_granularity", "S5")
        self.lookback_window = self.config.get("model_lookback_window", 100)

        # Data buffers are now created on-the-fly when an instrument is processed.
        self.data_buffers: Dict[str, collections.deque] = {}
        self.logger.info("TradingLogic initialized to handle dynamic instrument selection.")

    def _initialize_all_data_buffers(self):
        """Pre-fills the data buffers for all instruments on startup."""
        # This method is no longer called from __init__ to support dynamic instruments.
        # It is kept in case a "pre-warm" feature is needed in the future.
        self.logger.info("Initializing data buffers for all instruments...")
        # The logic to get instruments would need to be redefined, e.g., from config.
        instruments = [] # e.g., self.config.get('tradeable_instruments', [])
        for instrument in instruments:
            self._fill_buffer(instrument)

    def _fill_buffer(self, instrument: str):
        """
        Fills the data buffer for a given instrument with historical data.
        """
        try:
            self.logger.info(f"Attempting to pre-fill data buffer for {instrument} with {self.lookback_window} candles...")
            candles = self.client.get_candles(instrument, self.lookback_window, self.granularity)
            if candles and len(candles) >= self.lookback_window:
                self.data_buffers[instrument].extend(candles)
                self.logger.info(f"Successfully pre-filled data buffer for {instrument}. Buffer size: {len(self.data_buffers[instrument])}")
            else:
                self.logger.warning(f"Could not pre-fill buffer for {instrument}. "
                                  f"Received {len(candles) if candles else 0} candles, but expected {self.lookback_window}.")
        except Exception as e:
            self.logger.error(f"Error initializing data buffer for {instrument}: {e}", exc_info=True)

    def execute_trade_cycle(self):
        """
        執行所有當前選擇標的的單次交易循環。
        會遍歷 SystemState 中所有已選擇的 instrument，對每個 instrument 執行一次 cycle。
        """
        instruments = self.system_state.get_selected_instruments()
        if not instruments:
            self.logger.debug("目前未選擇任何交易標的，跳過本次交易循環。")
            return

        for instrument in instruments:
            # 若首次遇到該 instrument，動態建立 buffer 並初始化
            if instrument not in self.data_buffers:
                self.logger.info(f"Data buffer for new instrument '{instrument}' not found. Creating and initializing.")
                self.data_buffers[instrument] = collections.deque(maxlen=self.lookback_window)
                self._fill_buffer(instrument)

            self.logger.info(f"Running logic cycle for {instrument}.")

            try:
                # 檢查 buffer 是否已準備好
                if len(self.data_buffers[instrument]) < self.lookback_window:
                    self.logger.warning(f"Data buffer for {instrument} is not ready. Attempting to re-initialize.")
                    self._fill_buffer(instrument)
                    if len(self.data_buffers[instrument]) < self.lookback_window:
                        self.logger.warning(f"Skipping cycle for {instrument}, buffer still not ready.")
                        continue

                # 取得最新 K 線，保持 buffer 新鮮
                latest_candles = self.client.get_candles(instrument, 2, self.granularity)
                if not latest_candles:
                    self.logger.warning(f"Could not fetch latest candle for {instrument}. Skipping.")
                    continue

                last_buffered_time = self.data_buffers[instrument][-1]['time']
                new_candle = latest_candles[-1]
                if new_candle['time'] > last_buffered_time:
                    self.data_buffers[instrument].append(new_candle)
                    self.logger.debug(f"Appended new candle for {instrument} from {new_candle['time']}.")
                else:
                    self.logger.debug(f"No new candle data for {instrument}. Last known time: {last_buffered_time}")
                    continue

                # 特徵工程
                self.logger.debug(f"Preprocessing data for {instrument}.")
                features = self.preprocessor.transform(list(self.data_buffers[instrument]), instrument)
                if features is None or features.size == 0:
                    self.logger.error(f"Data preprocessing failed for {instrument}.")
                    continue

                # 模型推論
                self.logger.debug(f"Getting prediction for {instrument} from model.")
                prediction_map = self.prediction_service.predict({instrument: features})
                if not prediction_map or instrument not in prediction_map:
                    self.logger.error(f"Failed to get a prediction for {instrument}.")
                    continue

                signal = prediction_map[instrument]
                self.logger.info(f"Generated signal for {instrument}: {signal}")

                # 處理交易訊號
                last_candle = self.data_buffers[instrument][-1]
                signal_info = {
                    "instrument": instrument,
                    "signal": signal,
                    "price": float(last_candle['mid']['c']),
                    "timestamp": last_candle['time']
                }
                self.order_manager.process_signal(signal_info)

            except Exception as e:
                self.logger.critical(f"Critical error in logic cycle for {instrument}: {e}", exc_info=True)

    def run(self):
        """
        The main loop for the trading logic.
        """
        self.logger.info("Trading logic run loop started.")
        while self.system_state.is_running:
            self.execute_trade_cycle()
            time.sleep(self.config.get('trading_logic', {}).get('cycle_time_seconds', 60))
        self.logger.info("Trading logic run loop stopped.")
