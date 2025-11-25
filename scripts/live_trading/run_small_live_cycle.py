"""
Run a very small live trading cycle in PRACTICE mode using the live system.

Behavior:
- Forces OANDA practice environment by injecting a client with environment='practice'.
- Initializes the live system components via initialize_system(mock_client=...).
- Executes N quick cycles (default 3) with the configured interval.
- Writes a brief summary report to reports/live_small_run_YYYYmmdd_HHMMSS.md

Usage:
  python scripts/run_small_live_cycle.py [--cycles 3]
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


def add_src_to_path():
    here = Path(__file__).resolve()
    root = here.parents[1]  # .../Oanda_Trading_Bot
    src = root / 'src'
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main(cycles: int = 3):
    add_src_to_path()
    from oanda_trading_bot.live_trading_system.main import initialize_system
    from oanda_trading_bot.live_trading_system.core.oanda_client import OandaClient

    # Load .env from project root
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / '.env'
    load_dotenv(dotenv_path=str(dotenv_path))

    api_key = os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_ACCOUNT_ID')
    if not api_key or not account_id:
        print('Missing OANDA credentials in .env; aborting.')
        sys.exit(1)

    # Force PRACTICE to avoid accidental live orders
    client = OandaClient(api_key=api_key, account_id=account_id, environment='practice')

    components = initialize_system(mock_client=client)
    if not components:
        print('Failed to initialize live system; aborting.')
        sys.exit(2)

    system_state = components['system_state']
    trading_logic = components['trading_logic']
    config = components['config']

    # Run a few quick cycles
    interval = int(config.get('trading_loop_interval_seconds', 5))
    system_state.is_running = True
    errors = []
    for i in range(max(1, cycles)):
        try:
            trading_logic.execute_trade_cycle()
        except Exception as e:
            errors.append(str(e))
        time.sleep(interval)
    system_state.is_running = False

    # Save a quick summary report
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = reports_dir / f'live_small_run_{ts}.md'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('# Live Small Run Summary\n')
        f.write(f'- Timestamp: {ts}\n')
        f.write(f'- Cycles: {cycles}\n')
        f.write(f"- Instruments: {config.get('trading_instruments')}\n")
        f.write(f"- Granularity: {config.get('trading_granularity')}\n")
        f.write(f"- Interval(s): {interval}\n")
        f.write(f"- Model: {(config.get('model') or {}).get('path')}\n")
        if errors:
            f.write(f"- Errors: {len(errors)} (see logs)\n")
        else:
            f.write('- Errors: 0\n')
    print(f'Small live run complete. Report: {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cycles', type=int, default=3)
    args = p.parse_args()
    main(args.cycles)

