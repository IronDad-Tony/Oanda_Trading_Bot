import os
import sys
import json
from typing import Dict, Any, List
import pandas as pd
import numpy as np

SRC_PATH = os.path.join(os.path.dirname(__file__), '..', 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from oanda_trading_bot.live_trading_system.core.oanda_client import OandaClient
from oanda_trading_bot.common.mtf_features import compute_mtf_features


def load_live_config(project_root: str) -> Dict[str, Any]:
    cfg_path = os.path.join(project_root, 'configs', 'live', 'live_config.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cfg = load_live_config(root)
    instruments: List[str] = cfg.get('trading_instruments', [])
    scalers_path: str = cfg.get('preprocessor', {}).get('scalers_path', 'data/scalers/mtf_scalers.json')
    lookback = int(cfg.get('model_lookback_window', 256))

    client = OandaClient.from_env()

    scalers: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for inst in instruments:
        records = client.get_bid_ask_candles_combined(inst, count=5000, granularity='S5') or []
        if not records:
            continue
        df = pd.DataFrame(records)
        mtf = compute_mtf_features(df)
        # Keep only finite values
        mtf = mtf.replace([np.inf, -np.inf], np.nan).dropna()
        if len(mtf) == 0:
            continue
        # Use the most recent window for scaling statistics to better match live
        mtf_tail = mtf.tail(max(lookback * 4, 1024))
        inst_scalers: Dict[str, Dict[str, List[float]]] = {}
        for col in mtf_tail.columns:
            s = mtf_tail[col].values.astype(float)
            mean = float(np.mean(s))
            std = float(np.std(s))
            if std < 1e-9:
                std = 1.0
            inst_scalers[col] = {"mean": [mean], "scale": [std]}
        scalers[inst] = inst_scalers

    ensure_dir(os.path.join(root, scalers_path))
    with open(os.path.join(root, scalers_path), 'w', encoding='utf-8') as f:
        json.dump(scalers, f, indent=2)
    print(f"Saved scalers for {len(scalers)} instruments to {scalers_path}")


if __name__ == '__main__':
    main()
