import pytest
import os
import json
import time
import logging
from dotenv import load_dotenv

from live_trading_system.core.oanda_client import OandaClient
from live_trading_system.database.database_manager import DatabaseManager
from live_trading_system.trading.order_manager import OrderManager
from live_trading_system.trading.position_manager import PositionManager
from live_trading_system.core.system_state import SystemState
from live_trading_system.trading.risk_manager import RiskManager

# Configure basic logging for tests
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Test Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

CONFIG_PATH = os.path.join(project_root, 'live_trading_system', 'live_config.json')
DATABASE_PATH = os.path.join(project_root, 'live_trading_system', 'database', 'e2e_test_trading_log.sqlite')
LOG_FILE_PATH = os.path.join(project_root, 'live_trading_system', 'logs', 'real_api_e2e_test.log')

# Setup logger to write to file
log_directory = os.path.dirname(LOG_FILE_PATH)
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Remove old log file if it exists
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)

# Get root logger and set level
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) 

# Create file handler and set level to debug
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to the logger
if not any(isinstance(h, logging.FileHandler) and h.baseFilename == file_handler.baseFilename for h in logger.handlers):
    logger.addHandler(file_handler)


# --- Helper Functions ---

def load_test_config():
    """Loads the live trading configuration for the test."""
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def cleanup_test_artifacts():
    """Removes test-generated files."""
    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)
    if os.path.exists(LOG_FILE_PATH):
        # Optional: remove log file if needed
        # os.remove(LOG_FILE_PATH)
        pass

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def system_environment():
    """
    Prepares the environment for the E2E test.
    - Cleans up previous test artifacts.
    - Loads configuration and environment variables.
    - Initializes all essential components.
    - Yields the initialized components to the test function.
    - Cleans up any open positions and artifacts after the test.
    """
    logging.info("\n--- Setting up E2E test environment ---")
    cleanup_test_artifacts()

    config = load_test_config()
    oanda_account_id = os.getenv("OANDA_ACCOUNT_ID")
    oanda_api_key = os.getenv("OANDA_API_KEY")
    oanda_environment = os.getenv("OANDA_ENVIRONMENT", "practice")
    
    assert oanda_account_id, "OANDA_ACCOUNT_ID not set in .env file"
    assert oanda_api_key, "OANDA_API_KEY not set in .env file"

    # Initialize components
    oanda_client = OandaClient(oanda_api_key, oanda_account_id, oanda_environment)
    db_manager = DatabaseManager(db_path=DATABASE_PATH)
    system_state = SystemState()
    position_manager = PositionManager()
    risk_manager = RiskManager(config, system_state, position_manager)
    order_manager = OrderManager(oanda_client, system_state, position_manager, risk_manager, db_manager)

    # Use a common, liquid instrument for testing
    instrument = "EUR_USD"
    system_state.set_selected_instruments([instrument])

    # Initial cleanup of any lingering positions from previous failed runs
    logging.info("Performing initial cleanup of any open positions...")
    order_manager.close_all_positions()
    time.sleep(3) # Allow time for cleanup

    yield {
        "oanda_client": oanda_client,
        "db_manager": db_manager,
        "order_manager": order_manager,
        "position_manager": position_manager,
        "system_state": system_state,
        "instrument": instrument
    }

    # --- Teardown ---
    logging.info("\n--- Tearing down test environment ---")
    try:
        logging.info("Closing all positions as part of teardown...")
        order_manager.close_all_positions()
        time.sleep(3) # Allow time for closing orders to process
    except Exception as e:
        logging.error(f"Error during teardown cleanup: {e}")
    finally:
        db_manager.close()
        cleanup_test_artifacts()
        logging.info("--- Teardown complete ---")


# --- End-to-End Test ---

@pytest.mark.e2e
def test_real_api_order_placement_and_logging(system_environment):
    """
    Tests the full end-to-end flow with the real OANDA practice API.
    1. Processes a BUY signal to place a market order.
    2. Verifies that the position is opened on OANDA and tracked internally.
    3. Verifies that the trade is logged in the database.
    4. Closes all positions.
    5. Verifies that the position is closed on OANDA and internally.
    6. Verifies that the closing trade is logged and PnL is recorded.
    """
    # --- 1. Setup ---
    order_manager = system_environment["order_manager"]
    position_manager = system_environment["position_manager"]
    db_manager = system_environment["db_manager"]
    oanda_client = system_environment["oanda_client"]
    instrument = system_environment["instrument"]

    # --- 2. Process a BUY Signal ---
    logging.info(f"\n--- Processing BUY signal for {instrument} ---")
    
    # Fetch the latest price to make the test realistic
    latest_candles = oanda_client.get_candles(instrument, granularity="S5", count=1)
    assert latest_candles and len(latest_candles) > 0, "Could not fetch latest candle to determine price."
    current_price = float(latest_candles[0]['mid']['c'])
    logging.info(f"Using current market price for test: {current_price}")

    # A dummy price is fine, as RiskManager will calculate SL/TP, but the order is MARKET.
    signal_info = {"instrument": instrument, "signal": 1, "price": current_price}
    order_manager.process_signal(signal_info)
    
    logging.info("Waiting for order to fill...")
    time.sleep(5) # Allow time for the order to be filled and reflected

    # --- 3. Verify Position Opened ---
    logging.info("--- Verifying position ---")
    
    # Verify with OANDA directly
    open_positions_data = oanda_client.get_open_positions()
    assert open_positions_data is not None and "positions" in open_positions_data
    
    found_position_on_oanda = any(
        p["instrument"] == instrument and p.get("long", {}).get("units", "0") != "0"
        for p in open_positions_data["positions"]
    )
    assert found_position_on_oanda, f"Position for {instrument} not found on OANDA."
    logging.info(f"Confirmed: Position for {instrument} is open on OANDA.")

    # Verify with internal PositionManager
    internal_position = position_manager.get_position(instrument)
    assert internal_position is not None
    assert internal_position.position_type == 'long'
    logging.info(f"Confirmed: PositionManager is tracking the long position for {instrument}.")

    # --- 4. Verify Database Logging (Open) ---
    logging.info("--- Verifying database log (open) ---")
    # Fetch the last trade logged for the instrument
    trade_log = db_manager.get_last_trade_for_instrument(instrument)
    
    assert trade_log is not None, "Trade was not logged in the database."
    assert trade_log['oanda_trade_id'] == internal_position.trade_id
    assert trade_log['instrument'] == instrument
    assert trade_log['units'] > 0
    assert trade_log['status'] == 'OPEN'
    assert trade_log['pnl'] is None
    logging.info(f"Confirmed: Opening trade (OANDA ID: {trade_log['oanda_trade_id']}) is correctly logged in DB.")

    # --- 5. Close the Position ---
    logging.info(f"\n--- Closing position for {instrument} ---")
    order_manager.close_all_positions()
    logging.info("Waiting for close order to process...")
    time.sleep(5)

    # --- 6. Verify Position Closed ---
    logging.info("--- Verifying position is closed ---")
    
    # Verify with OANDA
    positions_after_close = oanda_client.get_open_positions()
    assert positions_after_close is not None and "positions" in positions_after_close
    position_exists_after_close = any(p["instrument"] == instrument for p in positions_after_close["positions"])
    assert not position_exists_after_close, f"Position for {instrument} was not closed on OANDA."
    logging.info(f"Confirmed: Position for {instrument} is closed on OANDA.")

    # Verify with internal PositionManager
    assert position_manager.get_position(instrument) is None, "Position not removed from internal tracking."
    logging.info("Confirmed: PositionManager has removed the position.")

    # --- 7. Verify Database Logging (Close) ---
    logging.info("--- Verifying database log (close) ---")
    final_trade_record = db_manager.get_trade_by_oanda_id(trade_log['oanda_trade_id'])
    assert final_trade_record is not None, "Could not retrieve final trade record from DB."
    assert final_trade_record['status'] == 'CLOSED', "Trade status was not updated to CLOSED."
    assert final_trade_record['pnl'] is not None, "PnL was not recorded for the closed trade."
    assert final_trade_record['close_price'] is not None, "Close price was not recorded."
    logging.info(f"Confirmed: Database log updated to CLOSED with PnL: {final_trade_record['pnl']}.")
