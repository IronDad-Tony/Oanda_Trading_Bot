
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from ..core.system_state import SystemState
from ..trading.position_manager import PositionManager
from ..database.database_manager import DatabaseManager

def display_system_status(state: SystemState):
    """Displays the current status of the trading system."""
    st.header("System Status")
    status = "Running" if state.is_running() else "Stopped"
    instrument = state.get_current_instrument() or "Not Selected"
    model = state.get_current_model() or "Not Loaded"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("System Control", status)
    col2.metric("Active Instrument", instrument)
    col3.metric("Active Model", model)

def display_open_positions(position_manager: PositionManager):
    """Displays a table of all open positions."""
    st.header("Open Positions")
    positions = position_manager.get_all_positions()
    if not positions:
        st.info("No open positions.")
        return

    data = []
    for inst, pos in positions.items():
        data.append({
            "Instrument": pos.instrument,
            "Type": pos.position_type.upper(),
            "Units": pos.units,
            "Entry Price": f"{pos.entry_price:.5f}"
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

def display_trade_history(db_manager: DatabaseManager):
    """Displays a table of the most recent trade history from the database."""
    st.header("Recent Trade History")
    trade_history = db_manager.get_trade_history(limit=50)
    if not trade_history:
        st.info("No trade history found.")
        return

    df = pd.DataFrame(trade_history)
    # Reorder for better readability
    df = df[['timestamp', 'instrument', 'action', 'units', 'price', 'details']]
    st.dataframe(df, use_container_width=True)

def display_performance_metrics():
    """Displays key performance indicators (KPIs). Placeholder for now."""
    st.header("Performance Metrics")
    st.warning("Performance metrics calculation will be implemented in a future phase.")
    # Example of what could be here:
    # col1, col2, col3 = st.columns(3)
    # col1.metric("Total P/L", "$1,250.50")
    # col2.metric("Win Rate", "62%")
    # col3.metric("Sharpe Ratio", "1.8")

def create_control_panel(system_state: SystemState, main_logic_thread_func):
    """Creates control buttons for the system."""
    st.sidebar.header("Control Panel")
    
    # --- System Control ---
    if not system_state.is_running():
        if st.sidebar.button("Start System"):
            system_state.set_running(True)
            main_logic_thread_func()
            st.sidebar.success("System started!")
            st.experimental_rerun()
    else:
        if st.sidebar.button("Stop System"):
            system_state.set_running(False)
            st.sidebar.warning("System stopping...")
            st.experimental_rerun()

    # --- Instrument and Model Selection ---
    st.sidebar.subheader("Configuration")
    # In a real app, this list would come from a config file or an API
    available_instruments = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
    selected_instrument = st.sidebar.selectbox(
        "Select Instrument", 
        options=available_instruments, 
        index=available_instruments.index(system_state.get_current_instrument() or "EUR_USD")
    )
    if selected_instrument != system_state.get_current_instrument():
        system_state.set_current_instrument(selected_instrument)
        st.experimental_rerun()

    # This would also be dynamic in a full implementation
    available_models = ["model_v1.pth", "model_v2_lstm.pth"]
    selected_model = st.sidebar.selectbox(
        "Select Model", 
        options=available_models,
        index=available_models.index(system_state.get_current_model() or "model_v1.pth")
    )
    if selected_model != system_state.get_current_model():
        system_state.set_current_model(selected_model)
        st.experimental_rerun()
