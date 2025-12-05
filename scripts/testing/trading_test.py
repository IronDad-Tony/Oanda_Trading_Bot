"""
OANDA Trading Test Script
Test placing 10 trades and then closing them
"""
import os
import sys
import time

# Setup path
sys.path.insert(0, r"c:\Users\tonyh\Oanda_Trading_Bot\src")

from oanda_trading_bot.common.account_profile import get_account_profile
from oanda_trading_bot.live_trading_system.core.oanda_client import OandaClient

def main():
    # Connect to OANDA
    profile = get_account_profile()
    api_key = os.getenv('OANDA_API_KEY')
    client = OandaClient(api_key, profile.account_id, profile.environment)

    # Get account balance before
    summary = client.get_account_summary()
    balance_before = float(summary['account']['balance'])
    print('=== OANDA Trading Test ===')
    print(f'Account: {profile.account_id}')
    print(f'Balance before: ${balance_before:,.2f} AUD')
    print()

    # Test symbols - 10 different categories
    test_symbols = [
        'EUR_USD',      # Major pair
        'GBP_USD',      # Major pair
        'USD_JPY',      # Major pair
        'AUD_USD',      # Major pair
        'XAU_USD',      # Gold
        'XAG_USD',      # Silver
        'SPX500_USD',   # S&P 500
        'NAS100_USD',   # NASDAQ
        'BCO_USD',      # Brent Oil
        'WTICO_USD',    # WTI Oil
    ]

    print(f'Testing with {len(test_symbols)} symbols: {test_symbols}')
    print()

    # Place 10 BUY orders (small units)
    successful_trades = []
    for i, symbol in enumerate(test_symbols):
        units = 1  # Minimum trade size
        print(f'[{i+1}/10] Placing BUY order: {symbol} x {units} units...', end=' ')
        
        try:
            response = client.create_order(symbol, units)
            if response and 'orderFillTransaction' in response:
                fill = response['orderFillTransaction']
                price = float(fill.get('price', 0))
                print(f'FILLED @ {price}')
                successful_trades.append({
                    'symbol': symbol,
                    'price': price,
                    'units': units,
                    'time': fill.get('time')
                })
            elif response and 'orderCancelTransaction' in response:
                cancel = response['orderCancelTransaction']
                reason = cancel.get('reason', 'Unknown')
                print(f'REJECTED: {reason}')
            else:
                print('FAILED: No fill transaction')
        except Exception as e:
            print(f'ERROR: {e}')
        
        time.sleep(0.3)  # Wait for confirmation

    print()
    print('=== TRADING SUMMARY ===')
    print(f'Successful trades: {len(successful_trades)}/10')

    # Show open positions
    print()
    print('Current open positions:')
    positions = client.get_open_positions()
    if positions and 'positions' in positions and len(positions['positions']) > 0:
        for pos in positions['positions']:
            sym = pos.get('instrument')
            long_units = pos.get('long', {}).get('units', '0')
            short_units = pos.get('short', {}).get('units', '0')
            unrealized = float(pos.get('long', {}).get('unrealizedPL', 0)) + float(pos.get('short', {}).get('unrealizedPL', 0))
            print(f'  {sym}: Long={long_units}, Short={short_units}, P/L=${unrealized:.2f}')
    else:
        print('  No open positions')

    # Show new balance
    summary2 = client.get_account_summary()
    balance_after = float(summary2['account']['balance'])
    print()
    print(f'Balance after: ${balance_after:,.2f} AUD')
    print(f'Change: ${balance_after - balance_before:+,.2f}')
    
    # Ask to close positions
    if len(successful_trades) > 0:
        print()
        print('=== CLOSING ALL POSITIONS ===')
        
        # Get positions again
        positions = client.get_open_positions()
        if positions and 'positions' in positions:
            for pos in positions['positions']:
                sym = pos.get('instrument')
                try:
                    result = client.close_position(sym)
                    if result:
                        print(f'  Closed: {sym}')
                    else:
                        print(f'  Failed to close: {sym}')
                except Exception as e:
                    print(f'  Error closing {sym}: {e}')
                time.sleep(0.3)
        
        # Final balance
        summary3 = client.get_account_summary()
        balance_final = float(summary3['account']['balance'])
        print()
        print(f'Final balance: ${balance_final:,.2f} AUD')
        print(f'Total P/L: ${balance_final - balance_before:+,.2f}')

if __name__ == '__main__':
    main()
