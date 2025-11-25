import asyncio
import sys
import os
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.oanda_trading_bot.live_trading_system.main import initialize_system, async_trading_loop

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    """
    Initializes and runs the headless live trading system.
    """
    logging.info("Initializing live trading system...")
    
    components = initialize_system()
    
    if components is None:
        logging.critical("System initialization failed. Exiting.")
        return
        
    system_state = components.get("system_state")
    if not system_state:
        logging.critical("SystemState component not found. Exiting.")
        return

    # Set the system to running
    system_state.is_running = True
    
    # Manually set some instruments to trade for testing purposes
    # In a real scenario, this would be controlled by the UI or config
    system_state.select_instrument("EUR_USD")
    system_state.select_instrument("GBP_USD")
    logging.info(f"Manually selected instruments: {system_state.get_selected_instruments()}")

    logging.info("Starting the asynchronous trading loop...")
    try:
        await async_trading_loop(components)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down.")
    finally:
        system_state.is_running = False
        logging.info("Trading loop stopped. System shutdown.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Headless trading system shut down.")
