import os
import sys
import json
import logging
import threading
import time
import asyncio # New import
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from .core.logger_setup import setup_logger
from .core.oanda_client import OandaClient
from .core.system_state import SystemState
from .core.instrument_monitor import InstrumentMonitor
from .data.data_preprocessor import LivePreprocessor
from .model.prediction_service import PredictionService
from .trading.trading_logic import TradingLogic
from .trading.order_manager import OrderManager
from .trading.position_manager import PositionManager
from .trading.risk_manager import RiskManager
from .database.database_manager import DatabaseManager
from .core.transaction_stream import TransactionStreamWorker
from oanda_trading_bot.common.account_profile import (
    AccountProfileError,
    get_account_profile,
)
# New import for the broadcaster
from .web_ui.broadcaster import initialize_broadcaster, get_broadcaster

def load_config() -> Optional[Dict[str, Any]]:
    """Loads the live_config.json file."""
    # This path will need to be updated to find the new configs directory.
    # For now, we leave it and will address configuration management globally later.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    config_path = os.path.join(project_root, 'configs', 'live', 'live_config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.critical(f"Configuration file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        logging.critical(f"Error decoding JSON from the configuration file.")
        return None

def initialize_system(
    mock_client: Optional[OandaClient] = None,
    mock_prediction_service: Optional[PredictionService] = None,
    mock_db_manager: Optional[DatabaseManager] = None
) -> Optional[Dict[str, Any]]:
    """
    Initializes all components of the trading system.
    Allows injecting mock components for testing purposes.
    """
    # Load environment variables from .env file at the project root
    # The project root is the parent directory of the 'oanda_trading_bot' directory
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    dotenv_path = os.path.join(project_root_path, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    
    setup_logger()
    logging.info("--- Starting System Initialization ---")
    
    config = load_config()
    if not config:
        return None
    logging.info("1. Configuration loaded.")

    try:
        # --- Core Components ---
        logging.info("2. Loading environment variables...")
        oanda_api_key = os.getenv("OANDA_API_KEY")
        oanda_account_id = os.getenv("OANDA_ACCOUNT_ID")

        if not oanda_api_key or not oanda_account_id:
            logging.critical("CRITICAL: OANDA_API_KEY or OANDA_ACCOUNT_ID not found in .env file.")
            logging.critical(f"Attempted to load .env from: {dotenv_path}")
            return None
        logging.info("2a. OANDA credentials found.")

        # Resolve account profile (currency + preferred environment)
        try:
            logging.info("3. Resolving OANDA account profile...")
            account_profile = get_account_profile()
            logging.info("3a. Account profile resolved: %s", account_profile)
        except AccountProfileError as exc:
            logging.critical("Unable to resolve account profile: %s", exc)
            return None

        # Initialize OandaClient using resolved environment unless overridden by mocks
        logging.info("4. Initializing OandaClient...")
        if mock_client:
            client = mock_client
        else:
            client = OandaClient(
                api_key=oanda_api_key,
                account_id=oanda_account_id,
                environment=account_profile.environment,
            )
        logging.info("4a. OandaClient initialized.")
        
        # Initialize the WebSocket broadcaster
        logging.info("5. Initializing WebSocket broadcaster...")
        initialize_broadcaster(client)
        logging.info("5a. WebSocket broadcaster initialized.")

        logging.info("6. Initializing SystemState...")
        system_state = SystemState()
        logging.info("6a. SystemState initialized.")
        
        # Construct absolute path for the database
        logging.info("7. Initializing DatabaseManager...")
        db_path_from_config = config.get('database', {}).get('path', 'data/live_trading/trading_history.db')
        db_full_path = os.path.join(project_root_path, db_path_from_config)
        db_manager = mock_db_manager if mock_db_manager else DatabaseManager(db_path=db_full_path)
        logging.info("7a. DatabaseManager initialized.")

        logging.info("8. Initializing InstrumentMonitor...")
        instrument_monitor = InstrumentMonitor(client, system_state, config)
        logging.info("8a. InstrumentMonitor initialized.")

        # Align account currency with broker account to size risk in native ccy
        config['account_currency'] = account_profile.currency
        logging.info("Account currency resolved to %s via OANDA profile.", account_profile.currency)

        # --- Data and Model Components ---
        logging.info("9. Initializing LivePreprocessor...")
        scalers_path = config.get('preprocessor', {}).get('scalers_path', 'data/training/scalers.json')
        model_config_path = os.path.join(project_root_path, 'configs', 'training', 'enhanced_model_config.json')
        preprocessor = LivePreprocessor(scalers_path, config, model_config_path)
        logging.info("9a. LivePreprocessor initialized.")

        logging.info("10. Initializing PredictionService...")
        prediction_service = mock_prediction_service if mock_prediction_service else PredictionService(config)
        logging.info("10a. PredictionService initialized.")

        # Auto-load model onto GPU when available (falls back to CPU)
        try:
            logging.info("11. Attempting to load model...")
            model_cfg = config.get('model', {}) if isinstance(config, dict) else {}
            model_rel = model_cfg.get('path')
            if model_rel:
                model_abs = os.path.join(project_root_path, model_rel)
                if os.path.exists(model_abs):
                    # device=None triggers auto 'cuda' if available inside PredictionService
                    prediction_service.load_model(model_abs, device=None)
                    logging.info(f"11a. Loaded model for live trading: {model_abs}")
                else:
                    logging.warning(f"Configured model path not found: {model_abs}")
            else:
                logging.info("11b. No model path configured in live_config.json, skipping model load.")
        except Exception as e:
            logging.error(f"Failed to auto-load model: {e}", exc_info=True)

        # --- Trading Components ---
        logging.info("12. Initializing PositionManager...")
        position_manager = PositionManager()
        logging.info("12a. PositionManager initialized.")

        logging.info("13. Initializing RiskManager...")
        risk_manager = RiskManager(config, system_state, position_manager, client)
        logging.info("13a. RiskManager initialized.")

        logging.info("14. Initializing OrderManager...")
        order_manager = OrderManager(client, system_state, position_manager, risk_manager, db_manager, config)
        logging.info("14a. OrderManager initialized.")
        
        # --- Main Logic Engine ---
        logging.info("15. Initializing TradingLogic...")
        trading_logic = TradingLogic(
            config, client, system_state, preprocessor, prediction_service, order_manager
        )
        logging.info("15a. TradingLogic initialized.")

        # Start transactions stream if enabled
        exec_cfg = config.get('execution', {}) if isinstance(config, dict) else {}
        use_stream = bool(exec_cfg.get('use_transactions_stream', False))
        tx_stream = None
        if use_stream:
            try:
                logging.info("16. Starting TransactionStreamWorker...")
                tx_stream = TransactionStreamWorker(
                    api_client=client.client,
                    account_id=oanda_account_id,
                    last_transaction_id_getter=lambda: client.last_transaction_id,
                    on_transaction=order_manager.on_transaction_event,
                )
                tx_stream.start()
                logging.info("16a. Transactions stream started.")
            except Exception as e:
                logging.error(f"Failed to start transactions stream: {e}", exc_info=True)

        logging.info("--- All system components initialized successfully. ---")

        return {
            "config": config,
            "client": client,
            "system_state": system_state,
            "db_manager": db_manager,
            "instrument_monitor": instrument_monitor,
            "preprocessor": preprocessor,
            "prediction_service": prediction_service,
            "position_manager": position_manager,
            "risk_manager": risk_manager,
            "order_manager": order_manager,
            "trading_logic": trading_logic,
            "transactions_stream": tx_stream,
        }
    except Exception as e:
        logging.critical(f"A critical error occurred during system initialization: {e}", exc_info=True)
        return None

async def async_trading_loop(components: Dict[str, Any]):
    """
    The new asynchronous main trading loop that also broadcasts updates.
    """
    system_state = components['system_state']
    trading_logic = components['trading_logic']
    broadcaster = get_broadcaster()
    logger = logging.getLogger("LiveTradingSystem")

    logger.info("Asynchronous trading loop started.")
    while getattr(system_state, 'is_running', False):
        try:
            # Execute the synchronous trading cycle logic
            trading_logic.execute_trade_cycle()
            
            # Broadcast updates to the UI
            if broadcaster:
                logger.debug("Broadcasting updates to UI...")
                await broadcaster.broadcast_account_update()
                await broadcaster.broadcast_positions_update()
            
            # The sleep duration can be configured
            await asyncio.sleep(components['config'].get('trading_loop_interval_seconds', 60))
        except Exception as e:
            logger.error(f"An error occurred in the async trading loop: {e}", exc_info=True)
            # Avoid rapid-fire loops on persistent errors
            await asyncio.sleep(60)
    logger.info("Asynchronous trading loop has been stopped.")

def trading_loop(components: Dict[str, Any]):
    """
    (DEPRECATED) The main trading loop.
    This function is designed to be run in a background thread from the UI.
    """
    system_state = components['system_state']
    trading_logic = components['trading_logic']
    logger = logging.getLogger("LiveTradingSystem")

    logger.warning("Using DEPRECATED synchronous trading_loop.")
    logger.info("Trading loop started.")
    while getattr(system_state, 'is_running', False):
        try:
            trading_logic.execute_trade_cycle()
            # The sleep duration can be configured
            time.sleep(components['config'].get('trading_loop_interval_seconds', 60))
        except Exception as e:
            logger.error(f"An error occurred in the trading loop: {e}", exc_info=True)
            # Avoid rapid-fire loops on persistent errors
            time.sleep(60)
    logger.info("Trading loop has been stopped.")


"""
NOTE:
Legacy CLI that launched the old Streamlit UI has been removed.
Keep using `initialize_system()` and `trading_loop()` from this module
and launch the new UI via the updated start_live_trading_ui.bat.
"""
