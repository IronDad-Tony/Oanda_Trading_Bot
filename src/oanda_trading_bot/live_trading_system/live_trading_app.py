"""
OANDA AI Live Trading System - Professional Grade Streamlit UI
專業級實時交易系統 UI - 完整版（使用真實模型推論）

Features:
- Real-time account data from OANDA API
- Full integration with TradingLogic and PredictionService
- Transformer model inference for trading signals
- Symbol selection with categories
- Risk parameters configuration
- Live positions monitoring
- Trade execution with confirmation
- Real-time PnL tracking
"""
import os
import sys
from pathlib import Path

# Setup path
script_path = Path(__file__).resolve()
src_dir = script_path.parents[2]
project_root = script_path.parents[3]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('STREAMLIT_SERVER_FILE_WATCHER_TYPE', 'poll')

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
import time
import threading
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import OANDA basic components
OANDA_AVAILABLE = False
FULL_SYSTEM_AVAILABLE = False

try:
    from oanda_trading_bot.common.account_profile import get_account_profile
    from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
    from oanda_trading_bot.live_trading_system.core.oanda_client import OandaClient
    OANDA_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import OANDA components: {e}")

# Import full trading system (deferred to avoid circular imports)
def get_initialize_system():
    """Lazy import of initialize_system to avoid circular imports"""
    try:
        from oanda_trading_bot.live_trading_system.main import initialize_system
        return initialize_system
    except ImportError as e:
        logger.error(f"Failed to import initialize_system: {e}")
        return None


# Page config
st.set_page_config(
    page_title="OANDA AI Live Trading",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00d4aa;
    }
    .positive { color: #00d4aa !important; }
    .negative { color: #ff6b6b !important; }
    .neutral { color: #9e9e9e !important; }
    .status-connected { 
        background-color: #00d4aa; 
        color: white; 
        padding: 0.25rem 0.5rem; 
        border-radius: 5px;
    }
    .status-disconnected { 
        background-color: #ff6b6b; 
        color: white; 
        padding: 0.25rem 0.5rem; 
        border-radius: 5px;
    }
    .trade-log {
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        background-color: #1a1a2e;
        padding: 0.5rem;
        border-radius: 5px;
        max-height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


def load_model_config() -> Dict[str, Any]:
    """Load model configuration from live_config.json"""
    try:
        config_path = Path(__file__).parents[3] / "configs" / "live" / "live_config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load live_config.json: {e}")
    return {}


def initialize_trading_system() -> Optional[Dict[str, Any]]:
    """Initialize the complete trading system with all components"""
    initialize_system = get_initialize_system()
    if initialize_system is None:
        logger.error("Full trading system not available - initialize_system function not found")
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Trading system module not available"
        st.session_state.trade_logs.append(log_entry)
        return None
    
    try:
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Initializing complete trading system..."
        st.session_state.trade_logs.append(log_entry)
        
        # Use the main.py initialize_system function
        components = initialize_system()
        
        if components:
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Trading system initialized successfully!"
            st.session_state.trade_logs.append(log_entry)
            
            # Check if model is loaded
            prediction_service = components.get('prediction_service')
            if prediction_service and prediction_service.model is not None:
                st.session_state.model_loaded = True
                log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Model loaded and ready for inference"
                st.session_state.trade_logs.append(log_entry)
            else:
                log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ No model loaded - check model path in config"
                st.session_state.trade_logs.append(log_entry)
            
            return components
        else:
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Failed to initialize trading system"
            st.session_state.trade_logs.append(log_entry)
            return None
    except Exception as e:
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ System init error: {str(e)}"
        st.session_state.trade_logs.append(log_entry)
        logger.error(f"Failed to initialize trading system: {e}", exc_info=True)
        return None


def init_session_state():
    """Initialize session state variables"""
    # Load model config to get the correct lookback window
    model_config = load_model_config()
    model_lookback = model_config.get('model_lookback_window', 512)  # Default from Transformer model
    
    defaults = {
        'oanda_client': None,
        'account_info': {},
        'positions': [],
        'selected_symbols': [],
        'trading_active': False,
        'trade_logs': [],
        'last_refresh': None,
        'risk_per_trade': 0.02,
        'max_position_size': 0.1,
        'stop_loss_pips': 20,
        'take_profit_pips': 40,
        'capital_to_use': 10000.0,
        'pending_trades': [],
        'completed_trades': [],
        # Warmup and Auto-trading states
        'warmup_status': 'not_started',  # not_started, in_progress, completed, failed
        'warmup_progress': {},  # {symbol: buffer_count}
        'auto_trading_active': False,
        'model_loaded': False,
        'lookback_window': model_lookback,  # From model config (e.g., 512 for Transformer)
        'model_config': model_config,  # Store full config
        'data_buffers': {},  # {symbol: deque of candles}
        # Full trading system components
        'trading_system': None,  # Will hold all initialized components
        'system_initialized': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_oanda_client() -> Optional[OandaClient]:
    """Get or create OANDA client"""
    if not OANDA_AVAILABLE:
        return None
    
    if st.session_state.oanda_client is None:
        try:
            profile = get_account_profile()
            api_key = os.getenv('OANDA_API_KEY')
            client = OandaClient(api_key, profile.account_id, profile.environment)
            st.session_state.oanda_client = client
            logger.info("OANDA client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to create OANDA client: {e}")
            return None
    
    return st.session_state.oanda_client


def refresh_account_data():
    """Refresh all account data from OANDA API"""
    client = get_oanda_client()
    if client is None:
        return False
    
    try:
        # Get account summary
        response = client.get_account_summary()
        if response and 'account' in response:
            account = response['account']
            st.session_state.account_info = {
                'balance': float(account.get('balance', 0)),
                'nav': float(account.get('NAV', 0)),
                'unrealized_pl': float(account.get('unrealizedPL', 0)),
                'realized_pl': float(account.get('pl', 0)),
                'margin_used': float(account.get('marginUsed', 0)),
                'margin_available': float(account.get('marginAvailable', 0)),
                'currency': account.get('currency', 'USD'),
                'open_trades': int(account.get('openTradeCount', 0)),
                'open_positions': int(account.get('openPositionCount', 0)),
            }
        
        # Get positions
        positions_response = client.get_open_positions()
        if positions_response and 'positions' in positions_response:
            st.session_state.positions = positions_response['positions']
        else:
            st.session_state.positions = []
        
        st.session_state.last_refresh = datetime.now()
        return True
    except Exception as e:
        logger.error(f"Failed to refresh account data: {e}")
        return False


def get_categorized_symbols() -> Dict[str, List[tuple]]:
    """Get symbols categorized by type"""
    try:
        iim = InstrumentInfoManager(force_refresh=False)
        all_symbols = iim.get_all_available_symbols()
        
        categories = {
            'Major Pairs': [],
            'Minor Pairs': [],
            'Precious Metals': [],
            'Indices': [],
            'Energy': [],
            'Commodities': [],
            'Crypto': [],
        }
        
        major_pairs = {'EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD', 'USD_CHF', 'USD_CAD', 'NZD_USD'}
        metal_keywords = ['XAU', 'XAG', 'GOLD', 'SILVER']
        index_keywords = ['SPX', 'NAS', 'US30', 'UK100', 'DE30', 'JP225', 'AU200', 'HK50']
        energy_keywords = ['OIL', 'WTICO', 'BRENT', 'NATGAS', 'BCO']
        crypto_keywords = ['BTC', 'ETH', 'LTC']
        
        for sym in all_symbols:
            details = iim.get_details(sym)
            if details is None:
                continue
            
            display = details.display_name if hasattr(details, 'display_name') else sym
            
            if sym in major_pairs:
                categories['Major Pairs'].append((sym, display))
            elif any(m in sym for m in metal_keywords):
                categories['Precious Metals'].append((sym, display))
            elif any(i in sym for i in index_keywords):
                categories['Indices'].append((sym, display))
            elif any(e in sym for e in energy_keywords):
                categories['Energy'].append((sym, display))
            elif any(c in sym for c in crypto_keywords):
                categories['Crypto'].append((sym, display))
            elif '_' in sym and details.instrument_type == 'CURRENCY':
                categories['Minor Pairs'].append((sym, display))
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    except Exception as e:
        logger.error(f"Failed to get symbols: {e}")
        return {}


def start_warmup(symbols: List[str]) -> bool:
    """
    Start the warmup process - download historical data to fill the observation window
    """
    client = get_oanda_client()
    if client is None:
        return False
    
    st.session_state.warmup_status = 'in_progress'
    st.session_state.warmup_progress = {sym: 0 for sym in symbols}
    lookback = st.session_state.lookback_window
    
    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Starting warmup for {len(symbols)} symbols (lookback: {lookback})"
    st.session_state.trade_logs.append(log_entry)
    
    try:
        from collections import deque
        
        for symbol in symbols:
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] 📥 Fetching historical data for {symbol}..."
            st.session_state.trade_logs.append(log_entry)
            
            # Fetch historical candles
            candles = client.get_bid_ask_candles_combined(symbol, count=lookback, granularity="S5")
            
            if candles and len(candles) >= lookback:
                st.session_state.data_buffers[symbol] = deque(candles, maxlen=lookback)
                st.session_state.warmup_progress[symbol] = len(candles)
                log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {symbol}: {len(candles)} candles loaded"
                st.session_state.trade_logs.append(log_entry)
            else:
                actual = len(candles) if candles else 0
                st.session_state.data_buffers[symbol] = deque(candles if candles else [], maxlen=lookback)
                st.session_state.warmup_progress[symbol] = actual
                log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ {symbol}: Only {actual}/{lookback} candles available"
                st.session_state.trade_logs.append(log_entry)
            
            time.sleep(0.2)  # Rate limiting
        
        # Check if all symbols have enough data
        all_ready = all(
            len(st.session_state.data_buffers.get(sym, [])) >= lookback 
            for sym in symbols
        )
        
        if all_ready:
            st.session_state.warmup_status = 'completed'
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Warmup completed! All buffers filled."
        else:
            st.session_state.warmup_status = 'completed'  # Still completed, but with warnings
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Warmup completed with partial data."
        
        st.session_state.trade_logs.append(log_entry)
        return True
        
    except Exception as e:
        st.session_state.warmup_status = 'failed'
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Warmup failed: {str(e)}"
        st.session_state.trade_logs.append(log_entry)
        logger.error(f"Warmup failed: {e}")
        return False


def get_warmup_status_display() -> str:
    """Get a display string for warmup status"""
    status = st.session_state.warmup_status
    if status == 'not_started':
        return "⏸️ Not Started"
    elif status == 'in_progress':
        return "🔄 In Progress..."
    elif status == 'completed':
        return "✅ Ready"
    elif status == 'failed':
        return "❌ Failed"
    return "❓ Unknown"


def execute_single_trade_cycle(symbols: List[str]) -> Dict[str, Any]:
    """
    Execute a single trade cycle using the FULL TradingLogic system.
    This uses the actual model for inference.
    """
    results = {'trades': [], 'signals': [], 'errors': []}
    
    # Check if trading system is initialized
    if not st.session_state.system_initialized or st.session_state.trading_system is None:
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Trading system not initialized"
        st.session_state.trade_logs.append(log_entry)
        return {'success': False, 'error': 'Trading system not initialized'}
    
    try:
        components = st.session_state.trading_system
        trading_logic = components.get('trading_logic')
        system_state = components.get('system_state')
        
        if trading_logic is None:
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ TradingLogic not available"
            st.session_state.trade_logs.append(log_entry)
            return {'success': False, 'error': 'TradingLogic not available'}
        
        # Set selected instruments in system state
        system_state.set_selected_instruments(symbols)
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Executing trade cycle for {len(symbols)} symbols..."
        st.session_state.trade_logs.append(log_entry)
        
        # Execute the trade cycle using TradingLogic
        # This will:
        # 1. Fetch latest candles
        # 2. Update data buffers
        # 3. Prepare features
        # 4. Get model predictions
        # 5. Execute trades via OrderManager
        trading_logic.execute_trade_cycle()
        
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Trade cycle completed"
        st.session_state.trade_logs.append(log_entry)
        
        results['success'] = True
        return results
        
    except Exception as e:
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Trade cycle error: {str(e)}"
        st.session_state.trade_logs.append(log_entry)
        logger.error(f"Error in trade cycle: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def start_auto_trading(symbols: List[str]) -> bool:
    """
    Start automatic trading using the FULL trading system with model inference.
    """
    if st.session_state.warmup_status != 'completed':
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Cannot start trading - warmup not completed!"
        st.session_state.trade_logs.append(log_entry)
        return False
    
    # Initialize full trading system if not already done
    if not st.session_state.system_initialized:
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Initializing full trading system..."
        st.session_state.trade_logs.append(log_entry)
        
        components = initialize_trading_system()
        if components:
            st.session_state.trading_system = components
            st.session_state.system_initialized = True
            
            # Initialize the data buffers in TradingLogic
            trading_logic = components.get('trading_logic')
            if trading_logic:
                log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] 🔥 Warming up TradingLogic buffers..."
                st.session_state.trade_logs.append(log_entry)
                
                # Use TradingLogic's warmup_buffers method
                success = trading_logic.warmup_buffers(symbols, max_wait_seconds=120, sleep_seconds=2)
                if success:
                    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ TradingLogic buffers ready"
                    st.session_state.trade_logs.append(log_entry)
                else:
                    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ TradingLogic warmup timeout (proceeding anyway)"
                    st.session_state.trade_logs.append(log_entry)
        else:
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Failed to initialize trading system"
            st.session_state.trade_logs.append(log_entry)
            return False
    
    st.session_state.auto_trading_active = True
    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Auto-trading started for {len(symbols)} symbols"
    st.session_state.trade_logs.append(log_entry)
    
    # Execute one trade cycle immediately
    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Executing first trade cycle..."
    st.session_state.trade_logs.append(log_entry)
    
    results = execute_single_trade_cycle(symbols)
    
    if results.get('success'):
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ First trade cycle completed"
        st.session_state.trade_logs.append(log_entry)
    
    return True


def stop_auto_trading():
    """Stop automatic trading"""
    st.session_state.auto_trading_active = False
    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ⏹️ Auto-trading stopped"
    st.session_state.trade_logs.append(log_entry)



def place_market_order(symbol: str, units: int, side: str = 'buy') -> Dict[str, Any]:
    """Place a market order"""
    client = get_oanda_client()
    if client is None:
        return {'success': False, 'error': 'Client not available'}
    
    try:
        actual_units = units if side.lower() == 'buy' else -units
        
        # Log the order attempt
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Placing {side.upper()} order: {symbol} x {abs(units)} units"
        st.session_state.trade_logs.append(log_entry)
        
        # Place the order using create_order method
        response = client.create_order(symbol, actual_units)
        
        if response and 'orderFillTransaction' in response:
            fill = response['orderFillTransaction']
            result = {
                'success': True,
                'order_id': fill.get('id'),
                'instrument': fill.get('instrument'),
                'units': float(fill.get('units', 0)),
                'price': float(fill.get('price', 0)),
                'pl': float(fill.get('pl', 0)),
                'time': fill.get('time'),
            }
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Order filled: {symbol} @ {result['price']} ({result['units']} units)"
            st.session_state.trade_logs.append(log_entry)
            st.session_state.completed_trades.append(result)
            return result
        elif response and 'orderCancelTransaction' in response:
            # Order was rejected
            cancel = response['orderCancelTransaction']
            reason = cancel.get('reason', 'Unknown')
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Order rejected: {reason}"
            st.session_state.trade_logs.append(log_entry)
            return {'success': False, 'error': reason}
        else:
            error_msg = 'No fill or cancel transaction in response'
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Order failed: {error_msg}"
            st.session_state.trade_logs.append(log_entry)
            return {'success': False, 'error': error_msg}
            
    except Exception as e:
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Order error: {str(e)}"
        st.session_state.trade_logs.append(log_entry)
        return {'success': False, 'error': str(e)}


def close_position(symbol: str) -> Dict[str, Any]:
    """Close a position"""
    client = get_oanda_client()
    if client is None:
        return {'success': False, 'error': 'Client not available'}
    
    try:
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Closing position: {symbol}"
        st.session_state.trade_logs.append(log_entry)
        
        response = client.close_position(symbol)
        
        if response:
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Position closed: {symbol}"
            st.session_state.trade_logs.append(log_entry)
            return {'success': True, 'response': response}
        else:
            return {'success': False, 'error': 'No response from API'}
            
    except Exception as e:
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Close error: {str(e)}"
        st.session_state.trade_logs.append(log_entry)
        return {'success': False, 'error': str(e)}


def close_all_positions():
    """Close all open positions"""
    client = get_oanda_client()
    if client is None:
        return
    
    refresh_account_data()
    positions = st.session_state.positions
    
    for pos in positions:
        symbol = pos.get('instrument')
        if symbol:
            # Check if there are units in long or short
            long_units = float(pos.get('long', {}).get('units', 0))
            short_units = float(pos.get('short', {}).get('units', 0))
            
            try:
                if long_units > 0:
                    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Closing LONG {symbol}..."
                    st.session_state.trade_logs.append(log_entry)
                    result = client.close_position(symbol, long_units='ALL')
                    if result:
                        st.session_state.trade_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Closed LONG {symbol}")
                
                if short_units < 0:
                    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Closing SHORT {symbol}..."
                    st.session_state.trade_logs.append(log_entry)
                    result = client.close_position(symbol, short_units='ALL')
                    if result:
                        st.session_state.trade_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Closed SHORT {symbol}")
                        
            except Exception as e:
                st.session_state.trade_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error closing {symbol}: {e}")
            
            time.sleep(0.3)  # Small delay between closes


def display_account_summary():
    """Display account summary section"""
    st.subheader("📊 Account Summary")
    
    info = st.session_state.account_info
    if not info:
        st.warning("⚠️ Account data not available. Click 'Refresh Data' to load.")
        return
    
    currency = info.get('currency', 'AUD')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"💰 Balance ({currency})",
            value=f"${info.get('balance', 0):,.2f}",
        )
    
    with col2:
        st.metric(
            label=f"📈 NAV ({currency})",
            value=f"${info.get('nav', 0):,.2f}",
        )
    
    with col3:
        unrealized = info.get('unrealized_pl', 0)
        st.metric(
            label="📊 Unrealized P/L",
            value=f"${unrealized:,.2f}",
            delta=f"{unrealized:+,.2f}" if unrealized != 0 else None,
        )
    
    with col4:
        realized = info.get('realized_pl', 0)
        st.metric(
            label="✅ Realized P/L",
            value=f"${realized:,.2f}",
            delta=f"{realized:+,.2f}" if realized != 0 else None,
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💳 Margin Used",
            value=f"${info.get('margin_used', 0):,.2f}",
        )
    
    with col2:
        st.metric(
            label="💵 Margin Available",
            value=f"${info.get('margin_available', 0):,.2f}",
        )
    
    with col3:
        st.metric(
            label="📍 Open Positions",
            value=info.get('open_positions', 0),
        )
    
    with col4:
        st.metric(
            label="📋 Open Trades",
            value=info.get('open_trades', 0),
        )


def display_positions():
    """Display current positions"""
    st.subheader("📍 Open Positions")
    
    positions = st.session_state.positions
    
    if not positions:
        st.info("No open positions")
        return
    
    # Build positions dataframe
    pos_data = []
    for pos in positions:
        instrument = pos.get('instrument', 'Unknown')
        
        for side in ['long', 'short']:
            side_data = pos.get(side, {})
            units = float(side_data.get('units', 0))
            
            if units != 0:
                pos_data.append({
                    'Symbol': instrument,
                    'Side': side.upper(),
                    'Units': int(units),
                    'Avg Price': float(side_data.get('averagePrice', 0)),
                    'Unrealized P/L': float(side_data.get('unrealizedPL', 0)),
                })
    
    if pos_data:
        df = pd.DataFrame(pos_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        if st.button("🔴 Close All Positions", type="secondary"):
            close_all_positions()
            refresh_account_data()
            st.rerun()


def display_trade_logs():
    """Display trade execution logs"""
    st.subheader("📋 Trade Execution Log")
    
    logs = st.session_state.trade_logs
    
    if not logs:
        st.info("No trade activity yet")
        return
    
    # Show last 50 logs
    for log in reversed(logs[-50:]):
        if '✅' in log:
            st.success(log)
        elif '❌' in log:
            st.error(log)
        else:
            st.info(log)


def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">💹 OANDA AI Live Trading System</h1>', unsafe_allow_html=True)
    
    # Connection status
    client = get_oanda_client()
    if client:
        st.markdown('<span class="status-connected">🟢 CONNECTED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-disconnected">🔴 DISCONNECTED</span>', unsafe_allow_html=True)
        st.error("Failed to connect to OANDA API. Please check your credentials.")
        return
    
    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 Refresh Data", type="primary"):
            with st.spinner("Refreshing..."):
                refresh_account_data()
            st.success("Data refreshed!")
    with col2:
        if st.session_state.last_refresh:
            st.caption(f"Last: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    # Auto-refresh on first load
    if st.session_state.last_refresh is None:
        refresh_account_data()
    
    # Main content
    st.divider()
    
    # Account Summary
    display_account_summary()
    
    st.divider()
    
    # Sidebar - Trading Configuration
    with st.sidebar:
        st.header("⚙️ Trading Configuration")
        
        # Capital settings
        st.subheader("💰 Capital Settings")
        max_capital = st.session_state.account_info.get('margin_available', 100000)
        
        capital = st.number_input(
            "Capital to Use",
            min_value=100.0,
            max_value=max_capital,
            value=min(st.session_state.capital_to_use, max_capital),
            step=1000.0,
            help="Maximum capital to allocate for trading"
        )
        st.session_state.capital_to_use = capital
        
        # Risk Parameters
        st.subheader("⚠️ Risk Parameters")
        
        risk_per_trade = st.slider(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.risk_per_trade * 100,
            step=0.5,
            help="Maximum percentage of capital to risk per trade"
        ) / 100
        st.session_state.risk_per_trade = risk_per_trade
        
        max_pos = st.slider(
            "Max Position Size (%)",
            min_value=5.0,
            max_value=25.0,
            value=st.session_state.max_position_size * 100,
            step=5.0,
            help="Maximum position size as percentage of capital"
        ) / 100
        st.session_state.max_position_size = max_pos
        
        sl_pips = st.number_input(
            "Stop Loss (pips)",
            min_value=5,
            max_value=100,
            value=st.session_state.stop_loss_pips,
            help="Default stop loss in pips"
        )
        st.session_state.stop_loss_pips = sl_pips
        
        tp_pips = st.number_input(
            "Take Profit (pips)",
            min_value=10,
            max_value=200,
            value=st.session_state.take_profit_pips,
            help="Default take profit in pips"
        )
        st.session_state.take_profit_pips = tp_pips
        
        st.divider()
        
        # Model Configuration Display
        st.subheader("🧠 Model Configuration")
        lookback = st.session_state.lookback_window
        granularity = st.session_state.model_config.get('trading_granularity', 'S5')
        st.info(f"**Lookback Window:** {lookback} candles")
        st.info(f"**Granularity:** {granularity}")
        
        st.divider()
        
        # Symbol Selection - Using multiselect for better UX
        st.subheader("📊 Symbol Selection")
        
        categories = get_categorized_symbols()
        selected = []
        
        for category, symbols in categories.items():
            # Sort symbols and prepare options
            sorted_symbols = sorted(symbols, key=lambda x: x[0])
            options = [f"{sym}" for sym, display in sorted_symbols]
            
            # Use multiselect for each category - dynamic height based on content
            category_selected = st.multiselect(
                f"{category} ({len(symbols)} available)",
                options=options,
                default=[],
                key=f"cat_{category.replace(' ', '_')}",
                help=f"Select trading symbols from {category}"
            )
            selected.extend(category_selected)
        
        st.session_state.selected_symbols = selected
        
        # Show selected count with color coding
        if len(selected) == 0:
            st.warning("⚠️ No symbols selected")
        elif len(selected) <= 10:
            st.success(f"✅ Selected: {len(selected)} symbols")
        else:
            st.info(f"📊 Selected: {len(selected)} symbols")
    
    # Main trading area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_positions()
    
    with col2:
        st.subheader("🤖 AI Trading Control")
        
        selected = st.session_state.selected_symbols
        
        if selected:
            st.write(f"**Selected:** {', '.join(selected[:3])}{'...' if len(selected) > 3 else ''} ({len(selected)} total)")
            
            # Warmup Status Display
            st.markdown("---")
            warmup_status = get_warmup_status_display()
            st.markdown(f"**Warmup Status:** {warmup_status}")
            
            # Show warmup progress if in progress
            if st.session_state.warmup_status == 'in_progress':
                progress = st.session_state.warmup_progress
                total = len(progress)
                completed = sum(1 for v in progress.values() if v >= st.session_state.lookback_window)
                st.progress(completed / total if total > 0 else 0)
            
            # Warmup Button
            if st.session_state.warmup_status in ['not_started', 'failed']:
                if st.button("🔥 Start Warmup", type="primary", use_container_width=True):
                    with st.spinner("Downloading historical data..."):
                        start_warmup(selected)
                    st.rerun()
            
            # Auto-Trading Controls
            st.markdown("---")
            trading_status = "🟢 ACTIVE" if st.session_state.auto_trading_active else "⚪ INACTIVE"
            st.markdown(f"**Trading Status:** {trading_status}")
            
            if st.session_state.warmup_status == 'completed':
                if not st.session_state.auto_trading_active:
                    if st.button("🚀 Start Auto-Trading", type="primary", use_container_width=True):
                        start_auto_trading(selected)
                        st.rerun()
                else:
                    # When active, show execute cycle button
                    if st.button("🔄 Execute Trade Cycle", type="primary", use_container_width=True):
                        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Manual trade cycle triggered..."
                        st.session_state.trade_logs.append(log_entry)
                        results = execute_single_trade_cycle(selected)
                        refresh_account_data()
                        st.rerun()
                    
                    if st.button("⏹️ Stop Auto-Trading", type="secondary", use_container_width=True):
                        stop_auto_trading()
                        st.rerun()
            else:
                st.warning("⚠️ Complete warmup first")
            
            # Manual Trading Section
            st.markdown("---")
            st.subheader("🎯 Manual Trading")
            
            trade_size = st.number_input(
                "Units per Trade",
                min_value=1,
                max_value=100000,
                value=1000,
                step=100,
            )
            
            col_buy, col_sell = st.columns(2)
            
            with col_buy:
                if st.button("📈 BUY ALL", type="primary", use_container_width=True):
                    for sym in selected:
                        result = place_market_order(sym, trade_size, 'buy')
                        time.sleep(0.5)  # Wait for confirmation
                    refresh_account_data()
                    st.rerun()
            
            with col_sell:
                if st.button("📉 SELL ALL", type="secondary", use_container_width=True):
                    for sym in selected:
                        result = place_market_order(sym, trade_size, 'sell')
                        time.sleep(0.5)
                    refresh_account_data()
                    st.rerun()
            
            if st.button("🔴 CLOSE ALL POSITIONS", type="secondary", use_container_width=True):
                close_all_positions()
                refresh_account_data()
                st.rerun()
        else:
            st.warning("Select symbols from the sidebar to start trading")
    
    st.divider()
    
    # Trade logs
    display_trade_logs()
    
    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Account: {st.session_state.account_info.get('currency', 'N/A')}")


if __name__ == "__main__":
    main()
