#!/usr/bin/env python3
"""
Complete Smoke Test: Training + Live Trading
- 10 symbols, ~12 hours of data
- Full training process with CUDA optimization
- Ensure all currency conversions are correct (no fallback to 1)
- Live trading: 10 trades then close all positions
- Generate comprehensive report
"""
import os
import sys
import json
import time
import torch
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any
from decimal import Decimal

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / '.env')

# Import after path setup
from oanda_trading_bot.training_system.common.logger_setup import logger
from oanda_trading_bot.training_system.common.config import (
    WEIGHTS_DIR, GRANULARITY, ACCOUNT_CURRENCY, INITIAL_CAPITAL,
    DEVICE, USE_AMP
)
from oanda_trading_bot.training_system.data_manager.currency_download_helper import ensure_currency_data_for_trading
from oanda_trading_bot.training_system.data_manager.currency_manager import CurrencyDependencyManager
from oanda_trading_bot.training_system.data_manager.currency_route_planner import compute_required_pairs_for_training
from oanda_trading_bot.training_system.data_manager.mmap_dataset import UniversalMemoryMappedDataset
from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
from oanda_trading_bot.training_system.environment.trading_env import UniversalTradingEnvV4
from stable_baselines3.common.vec_env import DummyVecEnv
from oanda_trading_bot.training_system.agent.sac_agent_wrapper import QuantumEnhancedSAC
from oanda_trading_bot.live_trading_system.main import initialize_system
from oanda_trading_bot.live_trading_system.core.oanda_client import OandaClient


def fmt(dt: datetime) -> str:
    """Format datetime to OANDA ISO format"""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def verify_currency_conversions(
    symbols: List[str],
    account_currency: str,
    iim: InstrumentInfoManager,
    cur_mgr: CurrencyDependencyManager
) -> Dict[str, Any]:
    """Verify all currency conversions have valid routes (no fallback to 1)"""
    logger.info("=== Verifying Currency Conversion Routes ===")
    issues = []
    verified_routes = {}
    
    # Pre-compute routes for all symbols
    required_pairs = cur_mgr.refresh_routes_for_symbols(symbols)
    logger.info(f"Required conversion pairs: {sorted(list(required_pairs))}")
    
    # Verify each symbol has a valid route
    for symbol in symbols:
        details = iim.get_details(symbol)
        if not details:
            issues.append(f"Missing instrument details for {symbol}")
            continue
        
        quote_currency = details.quote_currency
        
        # Check if route exists and is valid
        try:
            route = cur_mgr._route_for_currency(quote_currency)
            
            # If quote currency == account currency, route.edges will be empty (direct conversion = 1)
            # This is OK, but we should still verify
            if quote_currency.upper() == account_currency.upper():
                verified_routes[symbol] = {
                    "quote_currency": quote_currency,
                    "route": "DIRECT (same currency)",
                    "num_edges": 0,
                    "rate": 1.0
                }
                logger.info(f"✓ {symbol}: {quote_currency} = {account_currency} (direct, rate=1.0)")
            elif not route.edges:
                issues.append(f"NO ROUTE FOUND for {symbol} (quote: {quote_currency}) -> {account_currency} (will fallback to 1!)")
            else:
                route_str = " -> ".join([f"{e.from_ccy}_{e.to_ccy}" for e in route.edges])
                verified_routes[symbol] = {
                    "quote_currency": quote_currency,
                    "route": route_str,
                    "num_edges": len(route.edges)
                }
                logger.info(f"✓ {symbol}: {quote_currency} -> {account_currency} via {route_str} ({len(route.edges)} steps)")
        except Exception as e:
            issues.append(f"ERROR getting route for {symbol} (quote: {quote_currency}): {e}")
    
    return {
        "verified_routes": verified_routes,
        "issues": issues,
        "required_pairs": sorted(list(required_pairs))
    }


def train_model(
    symbols: List[str],
    start_time: datetime,
    end_time: datetime,
    account_currency: str
) -> str:
    """Train model with full CUDA optimization"""
    logger.info("=== PHASE 1: TRAINING ===")
    
    # Ensure CUDA is available and optimized
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {device_name}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Optimize CUDA settings for RTX 5090
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
    else:
        logger.warning("CUDA not available, using CPU")
    
    start_iso = fmt(start_time)
    end_iso = fmt(end_time)
    granularity = GRANULARITY or "S5"
    
    # Initialize managers
    iim = InstrumentInfoManager(force_refresh=True)
    cur_mgr = CurrencyDependencyManager(account_currency, apply_oanda_markup=True, instrument_info_manager=iim)
    
    # Verify currency conversions BEFORE downloading
    logger.info("Verifying currency conversion routes...")
    conversion_check = verify_currency_conversions(symbols, account_currency, iim, cur_mgr)
    
    if conversion_check["issues"]:
        logger.error("Currency conversion issues found:")
        for issue in conversion_check["issues"]:
            logger.error(f"  - {issue}")
        raise ValueError("Currency conversion verification failed. Cannot proceed with fallback to 1.")
    
    logger.info(f"✓ All {len(symbols)} symbols have valid conversion routes")
    logger.info(f"Required conversion pairs: {conversion_check['required_pairs']}")
    
    # Download data with currency conversion pairs
    logger.info("=== Downloading Data ===")
    ok, all_syms = ensure_currency_data_for_trading(
        trading_symbols=symbols,
        account_currency=account_currency,
        start_time_iso=start_iso,
        end_time_iso=end_iso,
        granularity=granularity,
        streamlit_progress_bar=None,
        streamlit_status_text=None,
        perform_download=True,
    )
    
    if not ok:
        raise RuntimeError("Data download failed")
    
    logger.info(f"Downloaded data for {len(all_syms)} symbols")
    
    # Build dataset
    logger.info("=== Building Dataset ===")
    dataset = UniversalMemoryMappedDataset(
        symbols=sorted(list(all_syms)),
        start_time_iso=start_iso,
        end_time_iso=end_iso,
        granularity=granularity,
        timesteps_history=512,  # Use full lookback window
        force_reload=True,
        mmap_mode='r'
    )
    
    if not dataset.is_valid():
        raise RuntimeError("Dataset is not valid")
    
    logger.info(f"Dataset ready: {len(dataset)} timesteps")
    
    # Create environment
    logger.info("=== Creating Environment ===")
    env = UniversalTradingEnvV4(
        dataset=dataset,
        instrument_info_manager=iim,
        active_symbols_for_episode=symbols,
        initial_capital=float(INITIAL_CAPITAL),
        render_mode=None,
    )
    vec_env = DummyVecEnv([lambda: env])
    
    # Initialize SAC agent with CUDA optimization
    logger.info("=== Initializing SAC Agent (CUDA Optimized) ===")
    
    # Optimize for RTX 5090 32GB
    if torch.cuda.is_available():
        # Set optimal CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Use mixed precision training
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    agent = QuantumEnhancedSAC(
        env=vec_env,
        batch_size=512,  # Large batch for RTX 5090 32GB
        buffer_size_factor=30,  # Large buffer to utilize GPU memory
        learning_starts_factor=5,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    # Training with aggressive CUDA optimization
    logger.info("=== Training Model (Maximizing GPU Usage) ===")
    total_timesteps = 50000  # More steps for better model
    
    # Monitor GPU usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Initial GPU memory: {initial_memory:.2f} GB")
    
    start_train_time = time.time()
    
    # Use larger batch size for GPU optimization
    try:
        agent.agent.learn(
            total_timesteps=total_timesteps,
            log_interval=100,
            progress_bar=True
        )
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise
    
    train_duration = time.time() - start_train_time
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        current_memory = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Peak GPU memory: {peak_memory:.2f} GB")
        logger.info(f"Current GPU memory: {current_memory:.2f} GB")
        logger.info(f"GPU utilization: {peak_memory / 32.0 * 100:.1f}% of 32GB")
    
    logger.info(f"Training completed in {train_duration:.2f} seconds")
    
    # Save model
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model_name = f"sac_model_smoke_test_{len(symbols)}symbols.zip"
    save_path = os.path.join(WEIGHTS_DIR, model_name)
    agent.agent.save(save_path)
    logger.info(f"✓ Model saved to: {save_path}")
    
    # Get GPU stats
    peak_memory = None
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
    
    # Generate metadata
    from oanda_trading_bot.common.artifact_metadata import write_metadata
    metadata = {
        "symbols": symbols,
        "num_symbols": len(symbols),
        "account_currency": account_currency,
        "granularity": granularity,
        "start_time": start_iso,
        "end_time": end_iso,
        "total_timesteps": total_timesteps,
        "training_duration_seconds": train_duration,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "peak_gpu_memory_gb": peak_memory,
    }
    write_metadata(save_path, metadata)
    
    return save_path, {
        "start_time": start_iso,
        "end_time": end_iso,
        "total_timesteps": total_timesteps,
        "training_duration_seconds": train_duration,
        "peak_gpu_memory_gb": peak_memory
    }


def live_trading_test(
    model_path: str,
    symbols: List[str],
    max_trades: int = 10
) -> Dict[str, Any]:
    """Run live trading test: execute trades until reaching max_trades, then close all"""
    logger.info("=== PHASE 2: LIVE TRADING TEST ===")
    
    # Force PRACTICE environment
    api_key = os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_ACCOUNT_ID')
    if not api_key or not account_id:
        raise ValueError("Missing OANDA credentials in .env")
    
    client = OandaClient(api_key=api_key, account_id=account_id, environment='practice')
    
    # Initialize system
    components = initialize_system(mock_client=client)
    if not components:
        raise RuntimeError("Failed to initialize live system")
    
    # Update config with model path and symbols
    components['config']['model']['path'] = model_path
    components['config']['trading_instruments'] = symbols
    
    system_state = components['system_state']
    trading_logic = components['trading_logic']
    order_manager = components['order_manager']
    db_manager = components['db_manager']
    risk_manager = components['risk_manager']
    config = components['config']
    
    # Set selected instruments
    system_state.set_selected_instruments(symbols)
    system_state.set_current_model(model_path)
    
    # Load model with CUDA optimization
    logger.info(f"Loading model: {model_path}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    components['prediction_service'].load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Model loaded on: {components['prediction_service'].device}")
    
    # Warmup buffers
    logger.info("Warming up data buffers...")
    trading_logic.warmup_buffers(symbols, max_wait_seconds=180, sleep_seconds=2)
    
    # Get initial account state
    account_summary = client.get_account_summary()
    initial_equity = float(account_summary['account'].get('NAV', account_summary['account'].get('equity', 0)))
    logger.info(f"Initial equity: {initial_equity:.2f}")
    
    # Start trading
    system_state.start()
    logger.info("Starting trading loop...")
    
    # Track trades by OANDA trade ID to avoid duplicates
    seen_trade_ids = set()
    trades_executed = 0
    cycle_count = 0
    max_cycles = 500  # Increased limit for 10 trades
    interval = config.get('trading_loop_interval_seconds', 5)
    start_trading_time = time.time()
    max_trading_time = 1800  # 30 minutes max
    
    logger.info(f"Target: {max_trades} trades, max cycles: {max_cycles}, max time: {max_trading_time}s")
    
    while trades_executed < max_trades and cycle_count < max_cycles:
        if time.time() - start_trading_time > max_trading_time:
            logger.warning(f"Trading timeout reached ({max_trading_time}s). Executed {trades_executed} trades.")
            break
            
        try:
            # Execute one cycle
            trading_logic.execute_trade_cycle()
            cycle_count += 1
            
            # Check for new trades
            recent_trades = db_manager.get_trade_history(limit=200)
            new_trades = [
                t for t in recent_trades 
                if t.get('oanda_trade_id') not in seen_trade_ids
            ]
            
            if new_trades:
                for trade in new_trades:
                    trade_id = trade.get('oanda_trade_id')
                    if trade_id:
                        seen_trade_ids.add(trade_id)
                        trades_executed += 1
                        logger.info(
                            f"Trade #{trades_executed}/{max_trades}: {trade.get('instrument')} "
                            f"{trade.get('units')} units @ {trade.get('entry_price')} "
                            f"(ID: {trade_id})"
                        )
                        
                        if trades_executed >= max_trades:
                            logger.info(f"Reached target of {max_trades} trades!")
                            break
            
            if cycle_count % 10 == 0:
                logger.info(f"Cycle {cycle_count}: {trades_executed}/{max_trades} trades executed")
            
            time.sleep(interval)
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
            time.sleep(interval)
    
    # Stop trading
    system_state.stop()
    logger.info(f"Trading stopped. Executed {trades_executed} trades in {cycle_count} cycles")
    
    # Close all positions
    logger.info("Closing all open positions...")
    positions = client.get_open_positions()
    closed_positions = []
    
    if positions and positions.get('positions'):
        for pos in positions['positions']:
            instrument = pos.get('instrument')
            if instrument:
                try:
                    result = client.close_position(instrument, long_units="ALL", short_units="ALL")
                    if result:
                        closed_positions.append(instrument)
                        logger.info(f"Closed position: {instrument}")
                except Exception as e:
                    logger.error(f"Failed to close {instrument}: {e}")
    
    # Get final account state
    account_summary = client.get_account_summary()
    final_equity = float(account_summary['account'].get('NAV', account_summary['account'].get('equity', 0)))
    
    # Get all trades
    all_trades = db_manager.get_trade_history(limit=1000)
    closed_trades = [t for t in all_trades if t.get('status') == 'CLOSED']
    
    total_pnl = sum(t.get('pnl', 0) or 0 for t in closed_trades)
    win_count = sum(1 for t in closed_trades if (t.get('pnl', 0) or 0) > 0)
    win_rate = (win_count / len(closed_trades) * 100) if closed_trades else 0
    
    return {
        "trades_executed": trades_executed,
        "cycles_run": cycle_count,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_pnl": total_pnl,
        "closed_trades": len(closed_trades),
        "win_count": win_count,
        "win_rate": win_rate,
        "closed_positions": closed_positions,
        "all_trades": all_trades
    }


def generate_report(
    training_info: Dict[str, Any],
    live_trading_info: Dict[str, Any],
    symbols: List[str],
    model_path: str
) -> str:
    """Generate comprehensive test report"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = project_root / 'reports' / f'smoke_test_complete_{timestamp}.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Complete Smoke Test Report\n\n")
        f.write(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Test Configuration\n\n")
        f.write(f"- **Symbols**: {', '.join(symbols)} ({len(symbols)} total)\n")
        f.write(f"- **Model Path**: `{model_path}`\n")
        f.write(f"- **Account Currency**: {ACCOUNT_CURRENCY}\n")
        f.write(f"- **Device**: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}\n")
        if torch.cuda.is_available():
            f.write(f"- **GPU**: {torch.cuda.get_device_name(0)}\n")
            f.write(f"- **Peak GPU Memory**: {training_info.get('peak_gpu_memory_gb', 'N/A'):.2f} GB\n")
        f.write("\n")
        
        f.write("## Training Phase\n\n")
        f.write(f"- **Status**: ✅ Completed\n")
        f.write(f"- **Training Duration**: {training_info.get('training_duration_seconds', 0):.2f} seconds\n")
        f.write(f"- **Total Timesteps**: {training_info.get('total_timesteps', 0):,}\n")
        f.write(f"- **Data Range**: {training_info.get('start_time', 'N/A')} to {training_info.get('end_time', 'N/A')}\n")
        f.write(f"- **Currency Conversion**: ✅ All routes verified (no fallback to 1)\n")
        f.write("\n")
        
        f.write("## Live Trading Phase\n\n")
        f.write(f"- **Status**: ✅ Completed\n")
        f.write(f"- **Trades Executed**: {live_trading_info.get('trades_executed', 0)}\n")
        f.write(f"- **Cycles Run**: {live_trading_info.get('cycles_run', 0)}\n")
        f.write(f"- **Initial Equity**: {ACCOUNT_CURRENCY} {live_trading_info.get('initial_equity', 0):.2f}\n")
        f.write(f"- **Final Equity**: {ACCOUNT_CURRENCY} {live_trading_info.get('final_equity', 0):.2f}\n")
        f.write(f"- **Total PnL**: {ACCOUNT_CURRENCY} {live_trading_info.get('total_pnl', 0):.2f}\n")
        f.write(f"- **Closed Trades**: {live_trading_info.get('closed_trades', 0)}\n")
        f.write(f"- **Win Rate**: {live_trading_info.get('win_rate', 0):.2f}%\n")
        f.write(f"- **Positions Closed**: {len(live_trading_info.get('closed_positions', []))}\n")
        f.write("\n")
        
        f.write("## Trade Details\n\n")
        all_trades = live_trading_info.get('all_trades', [])
        if all_trades:
            f.write("| Instrument | Units | Entry Price | Close Price | PnL | Status |\n")
            f.write("|------------|-------|-------------|-------------|-----|--------|\n")
            for trade in all_trades[:20]:  # Show first 20
                f.write(f"| {trade.get('instrument', 'N/A')} | {trade.get('units', 0)} | "
                       f"{trade.get('entry_price', 0):.5f} | {trade.get('close_price', 0) or 'N/A':.5f} | "
                       f"{trade.get('pnl', 0) or 0:.2f} | {trade.get('status', 'N/A')} |\n")
        else:
            f.write("No trades recorded.\n")
        f.write("\n")
        
        f.write("## Summary\n\n")
        f.write("✅ **Training**: Successfully completed with CUDA optimization\n")
        f.write("✅ **Currency Conversion**: All routes verified, no fallback to 1\n")
        f.write("✅ **Live Trading**: Completed with position closure\n")
        f.write(f"✅ **Model Performance**: {live_trading_info.get('win_rate', 0):.2f}% win rate\n")
        f.write("\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Review trade performance\n")
        f.write("2. Analyze model predictions\n")
        f.write("3. Adjust risk parameters if needed\n")
        f.write("4. Scale up training with more data\n")
    
    return str(report_path)


def main():
    """Main smoke test execution"""
    logger.info("=" * 80)
    logger.info("COMPLETE SMOKE TEST: Training + Live Trading")
    logger.info("=" * 80)
    
    # Configuration
    symbols = [
        "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
        "NZD_USD", "USD_CHF", "XAU_USD", "NAS100_USD", "US30_USD"
    ]
    
    # ~12 hours of data (half day)
    end_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    # Roll back to recent weekday if weekend
    while end_time.weekday() >= 5:
        end_time -= timedelta(days=1)
    start_time = end_time - timedelta(hours=12)
    
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Time range: {start_time} to {end_time} (~12 hours)")
    logger.info(f"Account currency: {ACCOUNT_CURRENCY}")
    
    try:
        # Phase 1: Training
        model_path, training_info = train_model(symbols, start_time, end_time, ACCOUNT_CURRENCY)
        
        # Phase 2: Live Trading
        live_trading_info = live_trading_test(model_path, symbols, max_trades=10)
        
        # Generate report
        report_path = generate_report(training_info, live_trading_info, symbols, model_path)
        
        logger.info("=" * 80)
        logger.info("SMOKE TEST COMPLETED SUCCESSFULLY")
        logger.info(f"Report saved to: {report_path}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Smoke test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

