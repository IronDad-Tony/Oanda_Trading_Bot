#!/usr/bin/env python3
"""
GPU short training on ~1 day S5 data for 5 symbols, with summary stats.

Ensures historical data (incl. required FX conversion pairs), builds dataset/env,
trains a small SAC agent for a short run on GPU, saves the model, and prints a
concise summary of trading statistics (PnL, commissions, drawdown, margin).

Run: python scripts/run_small_training_1day_5symbols_with_summary.py
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from collections import defaultdict


def fmt(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def compute_summary(env):
    # Pull logs directly from the environment
    trade_log = list(getattr(env, 'trade_log', []))
    pv_hist = list(getattr(env, 'portfolio_value_history', []))
    initial_capital = float(getattr(env, 'initial_capital', 0.0))
    final_equity = float(getattr(env, 'equity_ac', 0.0))
    final_pv = float(getattr(env, 'portfolio_value_ac', 0.0))
    max_dd = float(getattr(env, 'max_drawdown_episode', 0.0))
    margin_used = float(getattr(env, 'total_margin_used_ac', 0.0))

    total_trades = len(trade_log)
    realized_pnl = sum(t.get('realized_pnl_ac', 0.0) for t in trade_log)
    total_commission = sum(t.get('commission_ac', 0.0) for t in trade_log)
    wins = sum(1 for t in trade_log if t.get('realized_pnl_ac', 0.0) > 0)
    losses = sum(1 for t in trade_log if t.get('realized_pnl_ac', 0.0) < 0)
    win_rate = (wins / total_trades) * 100.0 if total_trades else 0.0

    ret_pct = ((final_pv - initial_capital) / initial_capital) * 100.0 if initial_capital > 0 else 0.0

    # Symbol-level PnL
    pnl_by_symbol = defaultdict(float)
    for t in trade_log:
        pnl_by_symbol[t.get('symbol', 'UNKNOWN')] += t.get('realized_pnl_ac', 0.0)
    top3 = sorted(pnl_by_symbol.items(), key=lambda x: x[1], reverse=True)[:3]
    worst3 = sorted(pnl_by_symbol.items(), key=lambda x: x[1])[:3]

    summary = {
        'initial_capital': initial_capital,
        'final_equity_ac': final_equity,
        'final_portfolio_value_ac': final_pv,
        'return_percent': ret_pct,
        'max_drawdown_percent': float(max_dd) * 100.0,  # max_dd stored as ratio
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate_percent': win_rate,
        'realized_pnl_ac': realized_pnl,
        'total_commission_ac': total_commission,
        'final_margin_used_ac': margin_used,
        'top3_symbols_by_pnl': top3,
        'bottom3_symbols_by_pnl': worst3,
    }
    return summary


def main():
    # Ensure project and src paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_dir = os.path.join(project_root, 'src')
    for p in [project_root, src_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Deferred imports after path fix
    from oanda_trading_bot.training_system.common.logger_setup import logger
    from oanda_trading_bot.training_system.common.config import (
        WEIGHTS_DIR, GRANULARITY, ACCOUNT_CURRENCY,
    )
    from oanda_trading_bot.training_system.data_manager.currency_download_helper import ensure_currency_data_for_trading
    from oanda_trading_bot.training_system.data_manager.mmap_dataset import UniversalMemoryMappedDataset
    from oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
    from oanda_trading_bot.training_system.environment.trading_env import UniversalTradingEnvV4
    from stable_baselines3.common.vec_env import DummyVecEnv
    from oanda_trading_bot.training_system.agent.sac_agent_wrapper import QuantumEnhancedSAC

    # Symbols and time window (~1 day)
    symbols = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "XAU_USD"]
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    while end.weekday() >= 5:
        end -= timedelta(days=1)
    start = end - timedelta(days=1)
    start_iso, end_iso = fmt(start), fmt(end)
    granularity = "S5" if not GRANULARITY else GRANULARITY

    logger.info("=== Ensuring data for training (1-day window, 5 symbols) ===")
    ok, all_syms = ensure_currency_data_for_trading(
        trading_symbols=symbols,
        account_currency=ACCOUNT_CURRENCY,
        start_time_iso=start_iso,
        end_time_iso=end_iso,
        granularity=granularity,
        streamlit_progress_bar=None,
        streamlit_status_text=None,
        perform_download=True,
    )
    if not ok:
        logger.warning("ensure_currency_data_for_trading did not confirm success; proceeding if DB has data.")

    logger.info("=== Building Dataset ===")
    dataset = UniversalMemoryMappedDataset(
        symbols=sorted(list(all_syms)) if all_syms else symbols,
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
        render_mode=None,
    )
    vec_env = DummyVecEnv([lambda: env])

    logger.info("=== Initializing SAC Agent ===")
    agent = QuantumEnhancedSAC(
        env=vec_env,
        batch_size=64,
        buffer_size_factor=10,
        learning_starts_factor=2,
        verbose=1,
    )

    logger.info("=== Training (short run on GPU) ===")
    total_timesteps = 2000
    agent.agent.learn(total_timesteps=total_timesteps, log_interval=10)

    logger.info("=== Saving Model to weights/sac_model_symbols5.zip ===")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(WEIGHTS_DIR, "sac_model_symbols5.zip")
    agent.agent.save(save_path)
    logger.info(f"Saved model to: {save_path}")

    # --- Summary ---
    summary = compute_summary(env)
    logger.info("=== Summary (1-day small run) ===")
    for k, v in summary.items():
        logger.info(f"{k}: {v}")


if __name__ == "__main__":
    main()

