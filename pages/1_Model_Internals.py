import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from src.common.shared_data_manager import get_shared_data_manager

st.set_page_config(layout="wide")

st.title("🤖 Model Internals Visualization")

# Initialize session state
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = 0

# Placeholder for the data display
data_container = st.empty()

def get_diagnostics_data():
    """
    Retrieves all available diagnostics data from the shared data manager and returns the latest record.
    """
    try:
        shared_data_manager = get_shared_data_manager()
        # Fetch all available diagnostic records and clear the queue
        all_diagnostics = shared_data_manager.get_and_clear_latest_diagnostics()
        
        if all_diagnostics:
            # Return the most recent record for display
            latest_record = all_diagnostics[-1]
            # Log how many items we just processed
            st.session_state.processed_count = len(all_diagnostics)
            return latest_record.get('data')
            
        st.session_state.processed_count = 0
        return None
    except (FileNotFoundError, ConnectionRefusedError):
        st.warning("Shared data manager is not available. Is the training process running?")
        return None
    except Exception as e:
        st.error(f"An error occurred while fetching diagnostics data: {e}")
        return None

def display_transformer_activations(diagnostics_data):
    """
    Displays transformer layer activations.
    """
    st.subheader("Transformer Activations")
    if not diagnostics_data or "transformer_activations" not in diagnostics_data:
        st.info("No transformer activation data available.")
        return

    activations = diagnostics_data["transformer_activations"]
    
    # Assuming activations is a list of tensors/arrays for each layer
    for i, layer_activation in enumerate(activations):
        if isinstance(layer_activation, dict) and "mean" in layer_activation:
            st.write(f"**Layer {i+1} Activations (Mean):**")
            fig = px.imshow([layer_activation["mean"]], text_auto=True, aspect="auto", labels=dict(x="Neuron", y="Batch Item", color="Activation"))
            fig.update_layout(title=f"Layer {i+1} Mean Activations Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"Data for layer {i+1} is not in the expected format.")


def display_quantum_strategy_pool(diagnostics_data):
    """
    Displays quantum strategy pool weights and activations.
    """
    st.subheader("Quantum Strategy Pool")
    if not diagnostics_data or "strategy_pool" not in diagnostics_data:
        st.info("No quantum strategy pool data available.")
        return

    pool_data = diagnostics_data["strategy_pool"]
    weights = pool_data.get("weights")
    activations = pool_data.get("activations")

    if weights is not None:
        st.write("**Strategy Weights:**")
        fig = px.bar(x=[f"Strategy {i}" for i in range(len(weights))], y=weights, labels={'x': 'Strategy', 'y': 'Weight'}, title="Strategy Pool Weights")
        st.plotly_chart(fig, use_container_width=True)

    if activations is not None:
        st.write("**Strategy Activations:**")
        fig = px.imshow([activations], text_auto=True, aspect="auto", labels=dict(x="Strategy", y="Batch Item", color="Activation"), title="Strategy Pool Activations")
        st.plotly_chart(fig, use_container_width=True)


def update_dashboard():
    """
    Main function to update the dashboard with the latest data.
    """
    diagnostics_data = get_diagnostics_data()

    with data_container.container():
        if diagnostics_data:
            st.success(f"Dashboard updated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            col1, col2 = st.columns(2)

            with col1:
                display_transformer_activations(diagnostics_data)

            with col2:
                display_quantum_strategy_pool(diagnostics_data)
        else:
            st.info("Waiting for data from the training process...")

# Main loop to auto-refresh
while True:
    update_dashboard()
    time.sleep(5) # Refresh every 5 seconds
