import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from oanda_trading_bot.live_trading_system.main import initialize_system

comps = initialize_system()
assert comps, 'init failed'
print('Init OK')
ss = comps['system_state']
tl = comps['trading_logic']
cfg = comps['config']
insts = cfg.get('trading_instruments', [])
ss.set_selected_instruments(insts)
print('Warming up...')
tl.warmup_buffers(insts, max_wait_seconds=10, sleep_seconds=1)
print('Executing one cycle...')
tl.execute_trade_cycle()
print('Cycle OK')

