#!/usr/bin/env python3
"""
Two-symbol end-to-end smoke test to verify padding/masking behavior and run a short SAC training loop.

Steps:
- Download a few hours of S5 data for two symbols
- Build UniversalMemoryMappedDataset and UniversalTradingEnvV4 with MAX_SYMBOLS_ALLOWED slots
- Manually step once with actions applied only to dummy slots to confirm env ignores them
- Initialize SAC with TransformerFeatureExtractor (time-aware) and ESS advisor layer
- Train briefly and print sanity info
"""
import os
import sys
from datetime import datetime, timedelta, timezone
import numpy as np


def find_recent_weekday_window(formatter, hours: int = 3):
    now = datetime.now(timezone.utc)
    d = now
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    end = d.replace(hour=12, minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=hours)
    return formatter(start), formatter(end), start, end


def main():
    # Ensure project paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_dir = os.path.join(project_root, 'src')
    for p in [project_root, src_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Imports
    from oanda_trading_bot.training_system.common.logger_setup import logger
    from oanda_trading_bot.training_system.common.config import (
        MAX_SYMBOLS_ALLOWED, WEIGHTS_DIR, GRANULARITY
    )
    from oanda_trading_bot.training_system.data_manager.oanda_downloader import (
        manage_data_download_for_symbols, format_datetime_for_oanda
    )
    from oanda_trading_bot.training_system.data_manager.mmap_dataset import UniversalMemoryMappedDataset
    from oanda_trading_bot.training_system.data_manager.database_manager import query_historical_data
    from oanda_trading_bot.training_system.environment.trading_env import UniversalTradingEnvV4
    from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
    from stable_baselines3.common.vec_env import DummyVecEnv
    from oanda_trading_bot.training_system.agent.sac_agent_wrapper import QuantumEnhancedSAC

    # Two symbols under MAX_SYMBOLS_ALLOWED
    symbols = ["EUR_USD", "USD_JPY"]
    start_iso, end_iso, start_dt, end_dt = find_recent_weekday_window(format_datetime_for_oanda, hours=3)
    granularity = "S5" if GRANULARITY is None else GRANULARITY

    # Download data (with fallback to recent weekdays)
    attempts = 0
    max_attempts = 7
    while attempts < max_attempts:
        logger.info("=== Smoke Test: Data Download ===")
        logger.info(f"Symbols: {symbols}; Window: {start_iso} to {end_iso}; Granularity: {granularity}")
        manage_data_download_for_symbols(symbols, start_iso, end_iso, granularity=granularity,
                                         streamlit_progress_bar=None, streamlit_status_text=None)
        # verify presence
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
        # fallback to previous weekday
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

    # Manual mask/step check: send non-zero actions only to dummy slots
    obs, info = env.reset()
    padding_mask = obs.get('padding_mask')  # 1=active, 0=dummy
    if padding_mask is None:
        raise RuntimeError("padding_mask is missing from observation.")
    action = np.zeros((MAX_SYMBOLS_ALLOWED,), dtype=np.float32)
    # set +1 for dummy slots only
    action[padding_mask == 0] = 1.0
    # step once
    _, _, _, _, step_info = env.step(action)
    # after step, verify no positions exist (since active slots received 0, and dummy slots are ignored)
    any_position = False
    for slot_idx in range(env.num_env_slots):
        units = env.current_positions_units[slot_idx]
        if abs(units) > 0:
            any_position = True
            break
    print(f"Mask check — active slots: {int(padding_mask.sum())}, dummy slots: {int((padding_mask==0).sum())}")
    print(f"Mask check — positions after dummy-only action: {any_position} (expected False)")

    vec_env = DummyVecEnv([lambda: env])

    logger.info("=== Initializing SAC Agent (QuantumEnhancedSAC with ESS) ===")
    agent = QuantumEnhancedSAC(
        env=vec_env,
        batch_size=64,
        buffer_size_factor=10,
        learning_starts_factor=2,
        verbose=1
    )

    logger.info("=== Training (very short run) ===")
    total_timesteps = 500
    agent.agent.learn(total_timesteps=total_timesteps, log_interval=10)

    # Predict once and show action vs mask
    obs = vec_env.reset()
    actions, _ = agent.agent.predict(obs, deterministic=True)
    print(f"Predict — action shape: {actions.shape}")
    if isinstance(obs, dict) and 'padding_mask' in obs:
        pm = obs['padding_mask']
    else:
        # Try to get underlying env obs to fetch mask
        o_single, _ = env.reset()
        pm = o_single.get('padding_mask')
    print(f"Predict — padding_mask (first env): {pm[0] if hasattr(pm, '__getitem__') else pm}")
    print(f"Predict — action (first env): {actions[0]}")

    # Save model
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model_name = f"sac_universal_MS{MAX_SYMBOLS_ALLOWED}_{granularity}_2sym_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join(WEIGHTS_DIR, f"{model_name}.zip")
    agent.agent.save(save_path)
    logger.info(f"Saved model to: {save_path}")

    logger.info("=== Smoke Test Completed ===")


if __name__ == "__main__":
    main()

