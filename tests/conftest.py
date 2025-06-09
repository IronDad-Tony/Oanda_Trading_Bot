'''Configuration for pytest'''
import sys
import os
import pytest
import logging

# Add the project root to the Python path to allow imports from src
# __file__ is tests/conftest.py, so os.path.dirname(__file__) is tests/
# os.path.join(os.path.dirname(__file__), '..') goes up one level to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure logger is robustly available
try:
    from src.common.logger_setup import logger as common_logger # Use the project's logger
    logger = common_logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
    logger = logging.getLogger(__name__)
    logger.error("Failed to import common_logger in conftest.py, using basicConfig.")

# Now that project root is in path, we can import project modules
try:
    from src.environment.trading_env import UniversalTradingEnvV4
    from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset, cleanup_mmap_temp_files as global_cleanup_mmap_files # Import global cleanup
    from src.data_manager.instrument_info_manager import InstrumentInfoManager
    from src.common.config import (
        DEFAULT_TRAIN_START_ISO,
        DEFAULT_TRAIN_END_ISO,
        DEFAULT_SYMBOLS,
        MAX_SYMBOLS_ALLOWED,
        TIMESTEPS,
        ACCOUNT_CURRENCY,
        INITIAL_CAPITAL
    )
except ImportError as e:
    logger.error(f"Critical module import failed in conftest.py: {e}")
    # Fallback logger if common_logger fails or modules are not found
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
    logger = logging.getLogger(__name__)
    logger.error(f"Critical module import failed in conftest.py: {e}")
    # If core classes are missing, fixtures might not work.
    # Pytest will likely fail to collect tests or fixtures if these imports fail.

# --- Test Configuration (moved from test_trading_environment.py) ---
TEST_SYMBOLS_CONFTEST = DEFAULT_SYMBOLS[:min(MAX_SYMBOLS_ALLOWED, 2)] # Reduced for faster tests
TEST_START_ISO_CONFTEST = DEFAULT_TRAIN_START_ISO
TEST_END_ISO_CONFTEST = DEFAULT_TRAIN_END_ISO
INITIAL_TEST_CAPITAL_CONFTEST = float(INITIAL_CAPITAL)
MAX_TEST_EPISODE_STEPS_CONFTEST = 50  # Run a short episode for testing

@pytest.fixture(scope="session") # session scope for env, it's expensive to create
def env():
    logger.info("Setting up test environment fixture (env)...")
    local_dataset_instance = None # Keep a reference for cleanup
    try:
        logger.info(f"Initializing InstrumentInfoManager for fixture...")
        instrument_info_mgr = InstrumentInfoManager()
        for sym in TEST_SYMBOLS_CONFTEST:
            if not instrument_info_mgr.get_details(sym):
                logger.warning(f"Could not retrieve details for instrument {sym} in fixture.")

        logger.info(f"Initializing UniversalMemoryMappedDataset for fixture (Symbols: {TEST_SYMBOLS_CONFTEST}, Start: {TEST_START_ISO_CONFTEST}, End: {TEST_END_ISO_CONFTEST})...")
        local_dataset_instance = UniversalMemoryMappedDataset( # Assign to local_dataset_instance
            symbols=TEST_SYMBOLS_CONFTEST,
            start_time_iso=TEST_START_ISO_CONFTEST,
            end_time_iso=TEST_END_ISO_CONFTEST,
            timesteps_history=TIMESTEPS,
        )
        if len(local_dataset_instance) == 0:
            logger.error("Test dataset for fixture is empty.")
            if local_dataset_instance: # Check if it was created
                 local_dataset_instance.close() # Use close method
            pytest.fail("Test dataset is empty, cannot proceed with environment fixture.")
            return None

        logger.info(f"Initializing UniversalTradingEnvV4 for fixture (Active Symbols: {TEST_SYMBOLS_CONFTEST})...")
        environment = UniversalTradingEnvV4(
            dataset=local_dataset_instance, # Pass the created dataset
            instrument_info_manager=instrument_info_mgr,
            active_symbols_for_episode=TEST_SYMBOLS_CONFTEST,
            initial_capital=INITIAL_TEST_CAPITAL_CONFTEST,
            max_episode_steps=MAX_TEST_EPISODE_STEPS_CONFTEST,
        )
        logger.info("Test environment fixture (env) setup successful.")
        yield environment # Use yield for fixtures with setup and teardown

        # Teardown:
        logger.info("Tearing down test environment fixture (env)...")
        if hasattr(environment, 'dataset') and environment.dataset is not None:
            logger.info("Closing environment dataset...")
            environment.dataset.close() # Use the close method of the instance
        elif local_dataset_instance is not None: # If env creation failed but dataset was made
             logger.info("Closing standalone dataset from fixture...")
             local_dataset_instance.close() # Use the close method
        # global_cleanup_mmap_files() # Optionally call global cleanup if desired after all session tests
        logger.info("Test environment fixture (env) teardown complete.")

    except Exception as e:
        logger.error(f"Test environment fixture (env) setup/teardown failed: {e}", exc_info=True)
        if local_dataset_instance is not None: # Attempt to close dataset even if env setup fails
            logger.warning("Attempting to close dataset after fixture setup failure...")
            local_dataset_instance.close()
        pytest.fail(f"Failed to setup/teardown 'env' fixture: {e}")
        return None # Should not reach here

@pytest.fixture(scope="session")
def num_steps():
    return MAX_TEST_EPISODE_STEPS_CONFTEST

# It might be good to call the global cleanup once at the end of the test session
# def pytest_sessionfinish(session, exitstatus):
#     logger.info("Pytest session finished. Performing global mmap cleanup.")
#     global_cleanup_mmap_files()
