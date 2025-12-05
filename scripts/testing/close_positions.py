"""Close all open positions"""
import os
import sys
import time

sys.path.insert(0, r"c:\Users\tonyh\Oanda_Trading_Bot\src")

from oanda_trading_bot.common.account_profile import get_account_profile
from oanda_trading_bot.live_trading_system.core.oanda_client import OandaClient

profile = get_account_profile()
api_key = os.getenv('OANDA_API_KEY')
client = OandaClient(api_key, profile.account_id, profile.environment)

print('=== CLOSING ALL POSITIONS ===')
positions = client.get_open_positions()
if positions and 'positions' in positions:
    for pos in positions['positions']:
        sym = pos.get('instrument')
        long_units = pos.get('long', {}).get('units', '0')
        short_units = pos.get('short', {}).get('units', '0')
        
        try:
            if float(long_units) > 0:
                result = client.close_position(sym, long_units='ALL')
                print(f'  Closed LONG {sym}: {result is not None}')
            if float(short_units) < 0:
                result = client.close_position(sym, short_units='ALL')
                print(f'  Closed SHORT {sym}: {result is not None}')
        except Exception as e:
            print(f'  Error {sym}: {e}')
        time.sleep(0.3)

# Final balance
summary = client.get_account_summary()
balance = float(summary['account']['balance'])
pl = float(summary['account']['pl'])
print()
print(f'Final balance: ${balance:,.2f} AUD')
print(f'Total P/L: ${pl:,.2f}')
