#!/usr/bin/env python3
"""
Daily sanity training: downloads 1 day of data, runs a short SAC training,
and verifies gradients/weights update for Transformer and Quantum Strategy Layer.
Writes a concise JSON summary and a simple PNG chart of weight-norm deltas.
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

import json


def iso_day_window(day_dt: datetime) -> (str, str):
    start = day_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = day_dt.replace(hour=23, minute=59, second=59, microsecond=0)
    return (
        start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def get_weekday(dt: datetime) -> datetime:
    d = dt
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def snapshot_params(module) -> Dict[str, Any]:
    return {k: v.detach().clone().cpu() for k, v in module.named_parameters()} if module is not None else {}


def norm_delta(before: Dict[str, Any], after: Dict[str, Any]) -> float:
    import torch
    total = 0.0
    keys = set(before.keys()) & set(after.keys())
    for k in keys:
        try:
            total += torch.norm(after[k] - before[k]).item()
        except Exception:
            pass
    return float(total)


def main():
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"
    for p in [project_root, src_dir]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    # Imports after path setup
    from oanda_trading_bot.training_system.common.logger_setup import logger
    from oanda_trading_bot.training_system.common.config import (
        ACCOUNT_CURRENCY,
        WEIGHTS_DIR,
        MAX_SYMBOLS_ALLOWED,
    )
    from oanda_trading_bot.training_system.data_manager.currency_download_helper import (
        ensure_currency_data_for_trading,
    )
    from oanda_trading_bot.training_system.data_manager.mmap_dataset import (
        UniversalMemoryMappedDataset,
    )
    from oanda_trading_bot.training_system.environment.trading_env import (
        UniversalTradingEnvV4,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv
    from oanda_trading_bot.training_system.agent.sac_agent_wrapper import (
        QuantumEnhancedSAC,
    )

    # Choose a small set
    symbols = ["EUR_USD", "GBP_USD", "USD_JPY"]

    # One recent weekday day window (UTC)
    day = get_weekday(datetime.now(timezone.utc) - timedelta(days=1))
    start_iso, end_iso = iso_day_window(day)
    granularity = "S5"

    logger.info(
        f"Sanity training day={day.date()}, symbols={symbols}, granularity={granularity}"
    )

    ok, all_syms = ensure_currency_data_for_trading(
        symbols, ACCOUNT_CURRENCY, start_iso, end_iso, granularity,
        streamlit_progress_bar=None, streamlit_status_text=None, perform_download=True,
    )
    if not ok:
        logger.warning("Data ensure failed; proceeding with primary symbols only.")
        all_syms = set(symbols)

    dataset = UniversalMemoryMappedDataset(
        symbols=sorted(list(all_syms)),
        start_time_iso=start_iso,
        end_time_iso=end_iso,
        granularity=granularity,
        timesteps_history=128,
        force_reload=False,
        mmap_mode="r",
    )

    env = UniversalTradingEnvV4(
        dataset=dataset,
        instrument_info_manager=None,
        active_symbols_for_episode=symbols,
        initial_capital=100000.0,
        render_mode=None,
    )
    vec_env = DummyVecEnv([lambda: env])

    agent = QuantumEnhancedSAC(env=vec_env, batch_size=64, buffer_size_factor=10, learning_starts_factor=2, verbose=0)

    # Locate components for weight checks
    policy = agent.agent.policy
    transformer = getattr(getattr(policy, "features_extractor", None), "transformer", None)
    ess = getattr(policy, "ess_layer", None)
    attn = getattr(ess, "attention_network", None) if ess is not None else None
    strategies = list(getattr(ess, "strategies", [])) if ess is not None else []

    before = {
        "transformer": snapshot_params(transformer),
        "ess_attention": snapshot_params(attn),
    }
    # Capture any trainable strategy parameters (most classical strategies are parameter-free)
    before_strategy = []
    for i, s in enumerate(strategies):
        params = snapshot_params(s) if hasattr(s, 'parameters') else {}
        before_strategy.append(params)

    total_steps = 1200
    agent.agent.learn(total_timesteps=total_steps, log_interval=10)

    after = {
        "transformer": snapshot_params(transformer),
        "ess_attention": snapshot_params(attn),
    }
    after_strategy = []
    for i, s in enumerate(strategies):
        params = snapshot_params(s) if hasattr(s, 'parameters') else {}
        after_strategy.append(params)

    deltas = {
        "transformer_norm_delta": norm_delta(before["transformer"], after["transformer"]) if before["transformer"] and after["transformer"] else 0.0,
        "ess_attention_norm_delta": norm_delta(before["ess_attention"], after["ess_attention"]) if before["ess_attention"] and after["ess_attention"] else 0.0,
    }
    # Per-strategy deltas (only for those with parameters)
    strat_deltas = []
    for i in range(len(before_strategy)):
        try:
            strat_deltas.append(norm_delta(before_strategy[i], after_strategy[i]))
        except Exception:
            strat_deltas.append(0.0)

    report = {
        "day": str(day.date()),
        "symbols_primary": symbols,
        "symbols_all": sorted(list(all_syms)),
        "total_steps": total_steps,
        "weight_deltas": deltas,
        "strategy_weight_deltas": strat_deltas,
    }

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "daily_sanity_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Optional chart
    try:
        import matplotlib.pyplot as plt

        labels = list(deltas.keys())
        values = [deltas[k] for k in labels]
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.bar(labels, values, color=["#4e79a7", "#f28e2b"])
        plt.ylabel("Weight norm delta")
        plt.title("Transformer/ESS deltas")
        # Strategy deltas pane
        plt.subplot(1, 2, 2)
        if len(strat_deltas) > 0:
            plt.bar([f"S{i}" for i in range(len(strat_deltas))], strat_deltas, color="#59a14f")
            plt.title("Strategy deltas (parametrized)")
        else:
            plt.text(0.5, 0.5, "No parametrized\nstrategies", ha='center', va='center')
        plt.tight_layout()
        plt_path = reports_dir / "daily_sanity_weight_deltas.png"
        plt.savefig(plt_path)
        logger.info(f"Saved plot: {plt_path}")
    except Exception as e:
        logger.warning(f"Could not render plot: {e}")

    logger.info(f"Saved sanity report: {report_path}")


if __name__ == "__main__":
    main()
