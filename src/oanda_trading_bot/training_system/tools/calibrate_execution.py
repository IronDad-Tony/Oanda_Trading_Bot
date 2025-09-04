import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict

import pandas as pd


def infer_pip_size(symbol: str) -> float:
    try:
        # Simple heuristics
        if symbol.endswith("_JPY"):
            return 0.01
        if symbol.startswith("XAU_") or symbol.startswith("XAG_"):
            return 0.01
        return 0.0001
    except Exception:
        return 0.0001


def calibrate(log_csv: Path) -> Dict:
    df = pd.read_csv(log_csv)
    df = df.dropna(subset=['ref_price'])
    if df.empty:
        raise RuntimeError("No sufficient data in execution_metrics.csv for calibration.")

    # Normalize signed slippage; positive = adverse
    def signed_slip(row):
        side = (row['side'] or '').upper()
        fp = row['fill_price'] if not math.isnan(row['fill_price']) else row['ref_price']
        base = row['ref_price']
        diff = float(fp) - float(base)
        if side == 'BUY':
            return max(diff, 0.0)
        elif side == 'SELL':
            return max(-diff, 0.0)
        else:
            return abs(diff)

    df['adverse_slip'] = df.apply(signed_slip, axis=1)
    # Convert to pips
    df['pip_size'] = df['instrument'].apply(infer_pip_size)
    df['slip_pips'] = df['adverse_slip'] / df['pip_size']

    # Compute robust stats
    slip_sigma = float(df['slip_pips'].std(skipna=True)) if len(df) > 1 else 0.25
    slip_p95 = float(df['slip_pips'].quantile(0.95)) if len(df) > 1 else 0.5

    # Delay in seconds
    if 'request_ts_ms' in df.columns and 'response_ts_ms' in df.columns:
        df['latency_s'] = (df['response_ts_ms'] - df['request_ts_ms']) / 1000.0
        mean_latency_s = float(df['latency_s'].mean())
    else:
        mean_latency_s = 5.0

    # Map to steps (assume S5 ~ 5s)
    step_len_s = 5.0
    mean_steps = int(round(mean_latency_s / step_len_s))
    jitter_steps = max(1, int(round(slip_sigma / 10)))  # loose default mapping

    return {
        "simulate_order_latency": True,
        "execution_delay_mean_steps": max(0, mean_steps),
        "execution_delay_jitter_steps": max(0, jitter_steps),
        "max_slippage_pips": max(0.1, slip_p95),
        "slippage_sigma_pips": max(0.05, slip_sigma),
        "reject_if_slippage_exceeds_bound": True
    }


def main():
    ap = argparse.ArgumentParser(description="Calibrate execution simulation parameters from live logs.")
    root = Path(__file__).resolve().parents[4]
    ap.add_argument("--logs", default=str(root / 'logs' / 'execution_metrics.csv'))
    ap.add_argument("--out", default=str(root / 'configs' / 'training' / 'execution_sim_config.json'))
    args = ap.parse_args()

    cfg = calibrate(Path(args.logs))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote calibrated execution sim config to: {out_path}")


if __name__ == "__main__":
    main()
