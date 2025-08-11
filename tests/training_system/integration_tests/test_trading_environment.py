# filepath: c:\\Users\\tonyh\\Oanda_Trading_Bot\\tests\\integration_tests\\test_trading_environment.py
# filepath: c:\\Users\\tonyh\\Oanda_Trading_Bot\\test_trading_environment.py
import os
import sys
import logging
import numpy as np

# Path adjustments are now handled by conftest.py
# try:
#     from src.environment.trading_env import UniversalTradingEnvV4
#     from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset
#     from src.data_manager.instrument_info_manager import InstrumentInfoManager
#     from src.common.config import (
#         DEFAULT_TRAIN_START_ISO, 
#         DEFAULT_TRAIN_END_ISO, 
#         DEFAULT_SYMBOLS,
#         MAX_SYMBOLS_ALLOWED,
#         TIMESTEPS,
#         ACCOUNT_CURRENCY,
#         INITIAL_CAPITAL
#     )
#     from src.common.logger_setup import logger as common_logger
#     logger = common_logger
# except ImportError as e:
#     print(f"Critical module import failed: {e}")
#     print("Please ensure PYTHONPATH is set correctly and all necessary dependencies are installed.")
#     print("The script may not run as expected.")
#     # Fallback logger if common_logger fails
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
#     logger = logging.getLogger(__name__)
#     if "UniversalTradingEnvV4" not in globals(): # If class itself is not imported
#         sys.exit("Cannot import UniversalTradingEnvV4, tests cannot continue.")

# Use the logger configured by conftest.py or the project's common logger
logger = logging.getLogger(__name__) # Pytest will capture logs

# Test configurations are now in conftest.py
# TEST_SYMBOLS = DEFAULT_SYMBOLS[:min(MAX_SYMBOLS_ALLOWED, 3)] 
# TEST_START_ISO = DEFAULT_TRAIN_START_ISO 
# TEST_END_ISO = DEFAULT_TRAIN_END_ISO 
# INITIAL_TEST_CAPITAL = float(INITIAL_CAPITAL)
# MAX_TEST_EPISODE_STEPS = 50

# setup_test_environment function is now part of conftest.py as a fixture

def test_env_initialization(env):
    logger.info("--- Starting Test: Environment Initialization ---")
    assert env is not None, "Environment (from fixture) is None, initialization test failed."
    assert env.observation_space is not None, "Observation space not defined"
    assert env.action_space is not None, "Action space not defined"
    logger.info(f"Observation Space: {env.observation_space}")
    logger.info(f"Action Space: {env.action_space}")
    logger.info(f"Initial Capital: {env.initial_capital} {env.currency_manager.account_currency}")
    logger.info(f"Number of Active Tradable Symbols: {env.num_tradable_symbols_this_episode}")
    logger.info("--- Test Passed: Environment Initialization ---")
    # return True # Replaced with asserts

def test_env_reset(env):
    logger.info("--- Starting Test: Environment Reset (reset) ---")
    assert env is not None, "Environment (from fixture) is None"
    try:
        obs, info = env.reset()
        assert obs is not None, "Observation is None after reset"
        assert isinstance(obs, dict), "Observation should be a dictionary"
        assert "features_from_dataset" in obs, "Missing \'features_from_dataset\' in observation"
        logger.info(f"Initial observation after reset (partial): {{k: v.shape if isinstance(v, np.ndarray) else v for k,v in obs.items()}}")
        logger.info(f"Initial info after reset: {info}")
        # Check observation space compatibility
        for key, value in obs.items():
            assert key in env.observation_space.spaces, f"Observation key {key} not in observation space"
            assert env.observation_space.spaces[key].contains(value), f"Observation value for {key} not compatible with its space definition"
        logger.info("--- Test Passed: Environment Reset (reset) ---")
        # return True # Replaced with asserts
    except Exception as e:
        logger.error(f"Environment reset test failed: {e}", exc_info=True)
        assert False, f"Environment reset failed: {e}" # Fail the test explicitly

def test_env_step(env):
    logger.info("--- Starting Test: Environment Single Step (step) ---")
    assert env is not None, "Environment (from fixture) is None"
    try:
        _ = env.reset() # Ensure env is reset
        action = env.action_space.sample() # Take a random action
        logger.info(f"Executing random action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        assert obs is not None, "Observation is None after step"
        assert isinstance(obs, dict), "Observation should be a dictionary"
        assert isinstance(reward, float), f"Reward should be float, got {type(reward)}"
        assert isinstance(done, bool), f"Done flag should be bool, got {type(done)}"
        assert isinstance(info, dict), "Info should be a dictionary"
        logger.info(f"Observation after step (partial): {{k: v.shape if isinstance(v, np.ndarray) else v for k,v in obs.items()}}")
        logger.info(f"Reward: {reward}, Done: {done}")
        logger.info(f"Info: {info}")
        # Check observation space compatibility again after step
        for key, value in obs.items():
            assert key in env.observation_space.spaces, f"Observation key {key} not in observation space (step)"
            assert env.observation_space.spaces[key].contains(value), f"Observation value for {key} not compatible with its space definition (step)"

        logger.info("--- Test Passed: Environment Single Step (step) ---")
        # return True # Replaced with asserts
    except Exception as e:
        logger.error(f"Environment single step test failed: {e}", exc_info=True)
        assert False, f"Environment single step failed: {e}" # Fail the test explicitly

def test_env_episode_run(env, num_steps): # num_steps is now a fixture
    logger.info(f"--- Starting Test: Environment Full Episode Run ({num_steps} steps) ---")
    assert env is not None, "Environment (from fixture) is None"
    try:
        obs, info = env.reset()
        total_reward = 0.0
        actual_steps = 0
        for step_num in range(num_steps):
            actual_steps +=1
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            # logger.debug(f"Episode {step_num+1}/{num_steps} - Action: {action}, Reward: {reward:.4f}, Done: {done}")
            if done:
                logger.info(f"Episode finished early after {step_num+1} steps.")
                break
        logger.info(f"Full episode run completed. Total steps executed: {actual_steps}, Total reward: {total_reward:.4f}")
        logger.info(f"Final portfolio value (approximate): {info.get('portfolio_value_ac', 'N/A')}")
        logger.info("--- Test Passed: Environment Full Episode Run ---")
        # return True # Replaced with asserts
    except Exception as e:
        logger.error(f"Environment full episode run test failed: {e}", exc_info=True)
        assert False, f"Environment full episode run failed: {e}" # Fail the test explicitly

# The main() function is no longer needed for pytest execution.
# It can be removed or commented out if you don't run this file directly.
# def main():
#     logger.info("========= Starting Trading Environment Integration Test Script ==========")
    
#     # Attempt to create a log file to record the test process
#     # Logging is now handled by pytest and conftest.py
#     # ... (old logging setup removed) ...

#     # env fixture replaces manual setup here
#     # env = setup_test_environment(...)

#     # results = {}
#     # if env is not None:
#     #     results[\"initialization\"] = test_env_initialization(env)
#     #     results[\"reset\"] = test_env_reset(env)
#     #     results[\"step\"] = test_env_step(env)
#     #     results[\"episode_run\"] = test_env_episode_run(env, MAX_TEST_EPISODE_STEPS) # Use constant from conftest
#     # else:
#     #     logger.error(\"Environment setup failed, cannot run tests.\")
#     #     results[\"setup\"] = False
    
#     # logger.info(\"========= Trading Environment Integration Test Summary =========\")
#     # for test_name, success in results.items():
#     #     logger.info(f\"Test \'{test_name}\': {\'PASSED\' if success else \'FAILED\'}\")
#     # logger.info(\"===============================================================\")

#     # if file_handler:
#     #     logger.removeHandler(file_handler)
#     #     file_handler.close()

# # if __name__ == \"__main__\":
# #    main()
