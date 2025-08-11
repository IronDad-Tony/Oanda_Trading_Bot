import sys
import os
import logging

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Since this script is at the root, we need to add the 'src' directory to the Python path
# to allow the imports to work, just as they would in a real execution environment.
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

logger.info(f"Added '{src_path}' to Python path.")

# --- Verification ---
import_errors = []

def verify_import(module_name, class_name=None, alias=None):
    """A helper function to try importing modules and classes and log the result."""
    try:
        if class_name:
            if alias:
                exec(f"from {module_name} import {class_name} as {alias}")
            else:
                exec(f"from {module_name} import {class_name}")
            logger.info(f"✅ Successfully imported '{class_name}' from '{module_name}'")
        else:
            exec(f"import {module_name}")
            logger.info(f"✅ Successfully imported module '{module_name}'")
    except (ImportError, ModuleNotFoundError) as e:
        error_message = f"❌ FAILED to import from '{module_name}': {e}"
        logger.error(error_message)
        import_errors.append(error_message)

def main():
    logger.info("🚀 Starting Import Verification Script...")
    logger.info("This script will attempt to import all key modules to verify the new project structure.")
    logger.info("=====================================================================================")

    # --- Training System Imports ---
    logger.info("\n--- Verifying Training System Modules ---")
    verify_import("oanda_trading_bot.training_system.app")
    verify_import("oanda_trading_bot.training_system.trainer.universal_trainer", "UniversalTrainer")
    verify_import("oanda_trading_bot.training_system.agent.sac_agent_wrapper", "QuantumEnhancedSAC")
    verify_import("oanda_trading_bot.training_system.environment.trading_env", "UniversalTradingEnvV4")
    verify_import("oanda_trading_bot.training_system.models.enhanced_transformer", "EnhancedTransformer")
    verify_import("oanda_trading_bot.training_system.data_manager.mmap_dataset", "UniversalMemoryMappedDataset")

    # --- Live Trading System Imports ---
    logger.info("\n--- Verifying Live Trading System Modules ---")
    verify_import("oanda_trading_bot.live_trading_system.main", "initialize_system")
    verify_import("oanda_trading_bot.live_trading_system.app")
    verify_import("oanda_trading_bot.live_trading_system.core.oanda_client", "OandaClient")
    verify_import("oanda_trading_bot.live_trading_system.trading.position_manager", "PositionManager")
    verify_import("oanda_trading_bot.live_trading_system.database.database_manager", "DatabaseManager")

    # --- Common Module Imports ---
    logger.info("\n--- Verifying Common Modules ---")
    verify_import("oanda_trading_bot.common.instrument_info_manager", "InstrumentInfoManager")

    # --- Final Report ---
    logger.info("=====================================================================================")
    if not import_errors:
        logger.info("🎉 SUCCESS! All key modules were imported successfully.")
        logger.info("The project structure and import paths appear to be correct.")
    else:
        logger.error(f"🔥 FAILED! Found {len(import_errors)} import error(s). Please review the logs above.")
        for error in import_errors:
            print(error) # Print to stdout as well for high visibility
        sys.exit(1) # Exit with an error code if there were failures

if __name__ == "__main__":
    main()
