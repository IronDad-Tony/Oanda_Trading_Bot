#!/usr/bin/env python3
"""
Rewritten training script for the Universal SAC Trading Model.

This script handles:
1. Configuration management.
2. Environment setup (UniversalTradingEnvV4 with UniversalMemoryMappedDataset).
3. Agent initialization (QuantumEnhancedSAC).
4. Training the agent.
5. Model saving.
6. Basic logging.
"""
import sys
import os
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import logging
from datetime import datetime

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset
from src.data_manager.currency_manager import ensure_currency_data_for_trading
from src.common.config import ACCOUNT_CURRENCY
from src.data_manager.instrument_info_manager import InstrumentInfoManager
from src.environment.trading_env import UniversalTradingEnvV4
from src.agent.sac_agent_wrapper import QuantumEnhancedSAC
from src.common.config import (
    DEFAULT_TRAIN_START_ISO, 
    DEFAULT_TRAIN_END_ISO, 
    DEFAULT_SYMBOLS,
    MAX_SYMBOLS_ALLOWED
)

# Set PYTORCH_CUDA_ALLOC_CONF to potentially mitigate OOM errors
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# --- Configuration ---
CONFIG = {
    "symbols": DEFAULT_SYMBOLS[:MAX_SYMBOLS_ALLOWED],  # Ensure we don't exceed max allowed
    "train_start_iso": DEFAULT_TRAIN_START_ISO,
    "train_end_iso": DEFAULT_TRAIN_END_ISO,
    "total_timesteps": 8640,  # Approximately 12 hours of 5-second data
    "model_save_freq": 2000, # Adjusted save frequency
    "model_save_path": "trained_models/sac_universal_trader",
    "log_level": logging.INFO,
    "sac_params": {
        "batch_size": 64,  # Default batch size, will be overridden in the test loop
        # Add any specific SAC params for QuantumEnhancedSAC if needed
        # e.g., "learning_rate": 0.0003, "buffer_size": 1_000_000, ...
        # These would typically be passed to QuantumEnhancedSAC constructor
        # or it uses its own defaults.
    },
    "env_params": {
        # Add any specific environment params if needed
        # e.g., "initial_balance": 10000, "transaction_fee": 0.001
    }
}

# --- Logger Setup ---
def setup_logging(log_level: int = logging.INFO, log_file: str = None):
    """Configures basic logging."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()] + ([logging.FileHandler(log_file)] if log_file else [])
    )
    # Suppress overly verbose logs from libraries if necessary
    logging.getLogger("stable_baselines3").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    # Add more as needed

logger = logging.getLogger(__name__)

# --- Environment Setup ---
def create_training_env(symbols: list, start_time_iso: str, end_time_iso: str, env_params: dict):
    """Creates and wraps the trading environment."""
    logger.info(f"Creating UniversalMemoryMappedDataset for symbols: {symbols}")
    logger.info(f"Training period: {start_time_iso} to {end_time_iso}")
    
    # Ensure currency conversion pairs are present in DB for accurate training conversion
    try:
        ok, all_syms = ensure_currency_data_for_trading(
            trading_symbols=symbols,
            account_currency=ACCOUNT_CURRENCY,
            start_time_iso=start_time_iso,
            end_time_iso=end_time_iso,
            granularity=CONFIG.get("trading_granularity", "S5")
        )
        if ok:
            logger.info(f"Currency conversion pairs ensured. Total symbols fetched: {len(all_syms)}")
    except Exception as e:
        logger.warning(f"ensure_currency_data_for_trading failed: {e}")

    dataset = UniversalMemoryMappedDataset(
        symbols=symbols,
        start_time_iso=start_time_iso,
        end_time_iso=end_time_iso,
        # Add other dataset params if necessary
    )
    
    logger.info("Creating InstrumentInfoManager.")
    info_mgr = InstrumentInfoManager() # Assuming it loads info for the given symbols or all
    
    logger.info("Creating UniversalTradingEnvV4.")
    env = UniversalTradingEnvV4(
        dataset=dataset,
        instrument_info_manager=info_mgr,
        active_symbols_for_episode=symbols, # Ensure this matches dataset symbols for training
        **env_params # Pass additional env parameters
    )
    
    # Wrap in DummyVecEnv for SB3 compatibility
    wrapped_env = DummyVecEnv([lambda: env])
    logger.info("Environment created and wrapped successfully.")
    return wrapped_env

# --- Agent Training ---
def train_agent(env, total_timesteps: int, model_save_path: str, model_save_freq: int, sac_params: dict):
    """Initializes and trains the SAC agent."""
    logger.info("Initializing QuantumEnhancedSAC agent.")

    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache before training.")
        torch.cuda.empty_cache()
    
    # Ensure the save path directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Timestamp for unique model filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename_prefix = f"{os.path.basename(model_save_path)}_{timestamp}"
    
    # Checkpoint callback for saving the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=model_save_freq,
        save_path=os.path.join(os.path.dirname(model_save_path), "checkpoints"), # Save checkpoints in a subfolder
        name_prefix=model_filename_prefix,
        save_replay_buffer=True,
        save_vecnormalize=True, # If using VecNormalize wrapper
    )
    
    # QuantumEnhancedSAC might take env directly or specific parts of sac_params
    # Assuming it's an SB3-compatible wrapper
    agent_wrapper = QuantumEnhancedSAC(env, **sac_params) # Pass SAC specific parameters
    
    logger.info(f"Starting training for {total_timesteps} timesteps.")
    logger.info(f"Models will be saved to: {os.path.dirname(model_save_path)}")
    logger.info(f"Checkpoint prefix: {model_filename_prefix}")

    try:
        agent_wrapper.agent.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            log_interval=10 # Log training progress every 10 episodes/updates
        )
        final_model_path = f"{model_save_path}_final_{timestamp}.zip"
        agent_wrapper.agent.save(final_model_path)
        logger.info(f"Training complete. Final model saved to: {final_model_path}")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        raise
    finally:
        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache after training.")
            torch.cuda.empty_cache()
    
    return agent_wrapper.agent # Return the trained SB3 agent

# --- Main Execution ---
if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Generate a timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"training_pipeline_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    setup_logging(log_level=CONFIG["log_level"], log_file=log_file_path) # MODIFIED: Added log_file argument
    logger.info(f"Logging to file: {log_file_path}") # ADDED: Log the path for confirmation
    logger.info("--- Starting Universal Trader Training Script ---")
    
    # Device check (optional, QuantumEnhancedSAC might handle this)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    batch_sizes_to_test = [32, 64, 128, 256] # Example batch sizes
    successful_batch_sizes = []
    failed_batch_sizes = {}
    
    for bs in batch_sizes_to_test:
        logger.info(f"--- Testing with Batch Size: {bs} ---")
        CONFIG["sac_params"]["batch_size"] = bs
        
        try:
            # Create environment
            # Ensure a fresh environment for each run if necessary, or reuse if state is managed
            # For simplicity, recreating; consider if dataset loading is too slow for this loop
            train_env = create_training_env(
                symbols=CONFIG["symbols"],
                start_time_iso=CONFIG["train_start_iso"],
                end_time_iso=CONFIG["train_end_iso"],
                env_params=CONFIG["env_params"]
            )

            # Train agent
            trained_agent = train_agent(
                env=train_env,
                total_timesteps=CONFIG["total_timesteps"], # Consider reducing for quick tests
                model_save_path=CONFIG["model_save_path"],
                model_save_freq=CONFIG["model_save_freq"],
                sac_params=CONFIG["sac_params"]
            )
            successful_batch_sizes.append(bs)
            logger.info(f"--- Successfully trained with Batch Size: {bs} ---")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"--- CUDA OOM Error with Batch Size: {bs} ---")
                failed_batch_sizes[bs] = "CUDA OOM"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache() # Clear cache before next attempt
            else:
                logger.error(f"--- Error with Batch Size: {bs}: {e} ---", exc_info=True)
                failed_batch_sizes[bs] = str(e)
        except Exception as e:
            logger.error(f"--- Unexpected Error with Batch Size: {bs}: {e} ---", exc_info=True)
            failed_batch_sizes[bs] = str(e)
        finally:
            if 'train_env' in locals() and train_env is not None:
                train_env.close() # Clean up the environment
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    logger.info("--- Batch Size Test Summary ---")
    logger.info(f"Successfully trained with batch sizes: {successful_batch_sizes}")
    if failed_batch_sizes:
        logger.warning(f"Failed to train with batch sizes: {failed_batch_sizes}")
    logger.info("--- Universal Trader Training Script Finished ---")

    # Optional: Add evaluation step here
    # logger.info("Starting evaluation...")
    # evaluate_agent(trained_agent, eval_env_config) 
    # (Requires separate eval_env setup and evaluation function)
