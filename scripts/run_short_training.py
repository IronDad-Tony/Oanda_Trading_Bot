#!/usr/bin/env python3
"""
Headless short training runner (bypasses UI).

Downloads a few hours of S5 data for 3 FX symbols that require currency conversion,
builds the universal dataset/environment, and runs a brief SAC training to validate the full pipeline.
"""
import os
import sys
from datetime import datetime, timedelta, timezone


def find_recent_weekday_window(formatter, hours: int = 3):
    """Return (start_iso, end_iso, start_dt, end_dt) for a recent non-weekend UTC window of given hours."""
    now = datetime.now(timezone.utc)
    d = now
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    end = d.replace(hour=12, minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=hours)
    return formatter(start), formatter(end), start, end


def main():
    # Ensure project and src paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_dir = os.path.join(project_root, 'src')
    for p in [project_root, src_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Imports after sys.path adjustments
    from oanda_trading_bot.training_system.common.logger_setup import logger
    from oanda_trading_bot.training_system.common.config import (
        MAX_SYMBOLS_ALLOWED, WEIGHTS_DIR, GRANULARITY
    )
    from oanda_trading_bot.training_system.data_manager.oanda_downloader import (
        manage_data_download_for_symbols,
        format_datetime_for_oanda,
    )
    from oanda_trading_bot.training_system.data_manager.mmap_dataset import UniversalMemoryMappedDataset
    from oanda_trading_bot.training_system.data_manager.database_manager import query_historical_data
    from oanda_trading_bot.training_system.environment.trading_env import UniversalTradingEnvV4
    from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
    from stable_baselines3.common.vec_env import DummyVecEnv
    from oanda_trading_bot.training_system.agent.sac_agent_wrapper import QuantumEnhancedSAC

    # Choose 3 symbols with conversion needed (account currency in config is typically AUD)
    symbols = ["EUR_USD", "GBP_USD", "USD_JPY"]
    start_iso, end_iso, start_dt, end_dt = find_recent_weekday_window(format_datetime_for_oanda, hours=3)
    granularity = "S5" if GRANULARITY is None else GRANULARITY

    # Try up to 7 previous weekdays to ensure data
    attempts = 0
    max_attempts = 7
    while attempts < max_attempts:
        logger.info("=== Headless Short Training: Data Download ===")
        logger.info(f"Symbols: {symbols}; Window: {start_iso} to {end_iso}; Granularity: {granularity}")
        manage_data_download_for_symbols(symbols, start_iso, end_iso, granularity=granularity,
                                         streamlit_progress_bar=None, streamlit_status_text=None)
        # Quick DB presence check
        has_data = False
        try:
            for s in symbols:
                df = query_historical_data(s, granularity, start_iso, end_iso, limit=5)
                if df is not None and not df.empty:
                    has_data = True
                    break
        except Exception:
            pass
        if has_data:
            break
        # Step back to previous weekday and retry
        start_dt -= timedelta(days=1)
        end_dt -= timedelta(days=1)
        while start_dt.weekday() >= 5:
            start_dt -= timedelta(days=1)
            end_dt -= timedelta(days=1)
        start_iso = format_datetime_for_oanda(start_dt)
        end_iso = format_datetime_for_oanda(end_dt)
        attempts += 1
    if attempts >= max_attempts:
        raise RuntimeError("No data found in the last 7 weekdays for the selected symbols.")

    logger.info("=== Building Dataset ===")
    dataset = UniversalMemoryMappedDataset(
        symbols=symbols,
        start_time_iso=start_iso,
        end_time_iso=end_iso,
        granularity=granularity,
        timesteps_history=128,
        force_reload=True,
        mmap_mode='r'
    )

    logger.info("=== Creating Environment ===")
    iim = InstrumentInfoManager(force_refresh=False)
    env = UniversalTradingEnvV4(
        dataset=dataset,
        instrument_info_manager=iim,
        active_symbols_for_episode=symbols,
        initial_capital=100000.0,
        render_mode=None
    )

    vec_env = DummyVecEnv([lambda: env])

    logger.info("=== Initializing SAC Agent (QuantumEnhancedSAC with ESS) ===")
    agent = QuantumEnhancedSAC(
        env=vec_env,
        batch_size=64,
        buffer_size_factor=10,
        learning_starts_factor=2,
        verbose=1
    )

    logger.info("=== Training (short run) ===")
    total_timesteps = 1500
    agent.agent.learn(total_timesteps=total_timesteps, log_interval=10)

    logger.info("=== Saving Model ===")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model_name = f"sac_universal_MS{MAX_SYMBOLS_ALLOWED}_{granularity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join(WEIGHTS_DIR, f"{model_name}.zip")
    agent.agent.save(save_path)
    logger.info(f"Saved model to: {save_path}")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
