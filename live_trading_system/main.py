import os
import sys
import json
import logging
import threading
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Add project root to sys.path to allow direct imports
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from live_trading_system.core.logger_setup import setup_logger
from live_trading_system.core.oanda_client import OandaClient
from live_trading_system.core.system_state import SystemState
from live_trading_system.core.instrument_monitor import InstrumentMonitor
from live_trading_system.data.data_preprocessor import LivePreprocessor
from live_trading_system.model.prediction_service import PredictionService
from live_trading_system.trading.trading_logic import TradingLogic
from live_trading_system.trading.order_manager import OrderManager
from live_trading_system.trading.position_manager import PositionManager
from live_trading_system.trading.risk_manager import RiskManager
from live_trading_system.database.database_manager import DatabaseManager
from dotenv import load_dotenv

def load_config() -> Optional[Dict[str, Any]]:
    """Loads the live_config.json file."""
    config_path = os.path.join(project_root, 'live_config.json')
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
    # The project root is the parent directory of the 'live_trading_system' directory
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dotenv_path = os.path.join(project_root_path, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    
    setup_logger()
    config = load_config()
    if not config:
        return None

    try:
        # --- Core Components ---
        oanda_api_key = os.getenv("OANDA_API_KEY")
        oanda_account_id = os.getenv("OANDA_ACCOUNT_ID")
        oanda_environment = os.getenv("OANDA_ENVIRONMENT", "practice")

        if not oanda_api_key or not oanda_account_id:
            logging.critical("CRITICAL: OANDA_API_KEY or OANDA_ACCOUNT_ID not found in .env file.")
            logging.critical(f"Attempted to load .env from: {dotenv_path}")
            return None

        # Correctly initialize OandaClient with individual arguments
        client = mock_client if mock_client else OandaClient(
            api_key=oanda_api_key,
            account_id=oanda_account_id,
            environment=oanda_environment
        )
        
        system_state = SystemState()
        
        # Construct absolute path for the database
        db_path_from_config = config.get('database', {}).get('path', 'database/trading_history.db')
        db_full_path = os.path.join(project_root_path, db_path_from_config)
        db_manager = mock_db_manager if mock_db_manager else DatabaseManager(db_path=db_full_path)

        instrument_monitor = InstrumentMonitor(client, system_state, config)

        # --- Data and Model Components ---
        scalers_path = config.get('preprocessor', {}).get('scalers_path', 'path/to/default/scalers.json')
        model_config_path = os.path.join(project_root_path, 'configs', 'enhanced_model_config.json')
        preprocessor = LivePreprocessor(scalers_path, config, model_config_path)
        prediction_service = mock_prediction_service if mock_prediction_service else PredictionService(config)

        # --- Trading Components ---
        position_manager = PositionManager()
        risk_manager = RiskManager(config, system_state, position_manager)
        order_manager = OrderManager(client, system_state, position_manager, risk_manager, db_manager)
        
        # --- Main Logic Engine ---
        trading_logic = TradingLogic(
            config, client, system_state, preprocessor, prediction_service, order_manager
        )

        logging.info("All system components initialized successfully.")

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
        }
    except Exception as e:
        logging.critical(f"A critical error occurred during system initialization: {e}", exc_info=True)
        return None

def trading_loop(components: Dict[str, Any]):
    """
    The main trading loop.
    This function is designed to be run in a background thread from the UI.
    """
    system_state = components['system_state']
    trading_logic = components['trading_logic']
    logger = logging.getLogger("LiveTradingSystem")

    logger.info("Trading loop started.")
    while system_state.is_running():
        try:
            trading_logic.execute_trade_cycle()
            # The sleep duration can be configured
            time.sleep(components['config'].get('trading_loop_interval_seconds', 60))
        except Exception as e:
            logger.error(f"An error occurred in the trading loop: {e}", exc_info=True)
            # Avoid rapid-fire loops on persistent errors
            time.sleep(60)
    logger.info("Trading loop has been stopped.")


def main():
    """
    The main entry point for the application.
    It launches the Streamlit UI as a separate process.
    """
    setup_logger() 
    
    ui_app_path = os.path.join(os.path.dirname(__file__), 'ui', 'app.py')
    logging.info(f"Launching Streamlit UI from: {ui_app_path}")

    import subprocess
    
    command = [sys.executable, "-m", "streamlit", "run", ui_app_path]
    
    try:
        logging.info(f"Executing: {' '.join(command)}")
        subprocess.run(command, check=True)
    except FileNotFoundError:
        logging.critical("Could not find `streamlit`. Please ensure it is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        logging.critical(f"Streamlit application failed to launch or exited with an error: {e}")
    except Exception as e:
        logging.critical(f"An unexpected error occurred while trying to launch Streamlit: {e}")

if __name__ == "__main__":
    main()
