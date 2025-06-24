import streamlit as st
import threading
import time
import sys
import os

# Add project root to sys.path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from live_trading_system.main import initialize_system, trading_loop
from live_trading_system.ui import dashboard

def trading_thread_target(components):
    """Target function for the trading logic thread."""
    trading_loop(components)

def start_trading_system():
    """Initializes and starts the trading system in a background thread."""
    if 'components' not in st.session_state:
        st.session_state.components = initialize_system()

    if st.session_state.components:
        components = st.session_state.components
        system_state = components['system_state']

        if not system_state.is_running():
            system_state.set_running(True)
            thread = threading.Thread(
                target=trading_thread_target,
                args=(components,),
                daemon=True
            )
            thread.start()
            st.session_state.trading_thread = thread
            st.success("Trading system started.")
            try:
                st.experimental_rerun()
            except st.errors.RerunException:
                pass
        else:
            st.warning("Trading system is already running.")
    else:
        st.error("System components not initialized. Cannot start trading logic.")

def stop_trading_system():
    """Stops the trading logic."""
    if 'components' in st.session_state:
        system_state = st.session_state.components['system_state']
        if system_state.is_running():
            system_state.set_running(False)
            if 'trading_thread' in st.session_state:
                st.session_state.trading_thread.join(timeout=10)
            st.success("Trading system stopped.")
            try:
                st.experimental_rerun()
            except st.errors.RerunException:
                pass
        else:
            st.warning("Trading system is not running.")

def run_app():
    """
    The main function to build and run the Streamlit UI.
    """
    st.set_page_config(page_title="Oanda Trading System", layout="wide")
    st.title("Oanda Real-Time Trading Dashboard")

    if 'components' not in st.session_state:
        with st.spinner('Initializing trading system components...'):
            st.session_state.components = initialize_system()

        if st.session_state.components is None:
            st.error("Failed to initialize trading system. Check logs for details.")
            return

    components = st.session_state.components
    system_state = components['system_state']
    position_manager = components['position_manager']
    db_manager = components['db_manager']
    instrument_monitor = components['instrument_monitor']

    dashboard.create_control_panel(system_state, start_trading_system, stop_trading_system)

    col1, col2 = st.columns(2)
    with col1:
        dashboard.display_system_status(system_state)
    with col2:
        dashboard.display_instrument_status(instrument_monitor)

    st.divider()
    dashboard.display_open_positions(position_manager)
    st.divider()
    dashboard.display_trade_history(db_manager)
    st.divider()

    if system_state.is_running():
        time.sleep(5)
        try:
            st.experimental_rerun()
        except st.errors.RerunException:
            pass

def start_streamlit_app(components):
    """
    This function is designed to be called from main.py to launch the UI.
    """
    if 'components' not in st.session_state:
        st.session_state.components = components
    
    run_app()

if __name__ == '__main__':
    run_app()
