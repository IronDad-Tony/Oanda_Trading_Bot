"""
Quick Trading Test with Trained Model
使用訓練好的模型進行交易測試
"""

import os
import sys
import time
from pathlib import Path
from collections import deque

# Setup path
script_path = Path(__file__).resolve()
project_root = script_path.parents[2]
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TradingTest")

def main():
    logger.info("=" * 60)
    logger.info("LIVE TRADING TEST WITH TRAINED MODEL")
    logger.info("=" * 60)
    
    # Import components
    from oanda_trading_bot.common.account_profile import get_account_profile
    from oanda_trading_bot.live_trading_system.core.oanda_client import OandaClient
    from stable_baselines3 import SAC
    
    # Load trained model
    model_path = project_root / "weights" / "sac_transformer_symbols10.zip"
    logger.info(f"Loading model from: {model_path}")
    
    try:
        model = SAC.load(str(model_path))
        logger.info("✅ Model loaded successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.policy.parameters())
        logger.info(f"Model parameters: {total_params:,}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Connect to OANDA
    profile = get_account_profile()
    api_key = os.getenv('OANDA_API_KEY')
    client = OandaClient(api_key, profile.account_id, profile.environment)
    
    # Get account info
    summary = client.get_account_summary()
    balance = float(summary['account']['balance'])
    logger.info(f"\nAccount: {profile.account_id}")
    logger.info(f"Balance: ${balance:,.2f} {profile.currency}")
    
    # Trading symbols
    symbols = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'XAU_USD']
    
    # Warmup - fetch historical data
    logger.info(f"\n--- WARMUP: {len(symbols)} symbols ---")
    data_buffers = {}
    
    for symbol in symbols:
        try:
            candles = client.get_bid_ask_candles_combined(symbol, count=100, granularity="S5")
            if candles:
                data_buffers[symbol] = deque(candles, maxlen=100)
                last_price = candles[-1].get('ask_close', 0)
                logger.info(f"  ✅ {symbol}: {len(candles)} candles, last price: {last_price:.5f}")
            time.sleep(0.3)
        except Exception as e:
            logger.error(f"  ❌ {symbol}: {e}")
    
    # Execute trades
    logger.info(f"\n--- TRADING: Placing test orders ---")
    trades_executed = 0
    positions = []
    
    for symbol in symbols[:3]:  # Test with first 3 symbols
        try:
            logger.info(f"Placing order for {symbol}...")
            response = client.create_order(symbol, 1)  # 1 unit
            
            if response and 'orderFillTransaction' in response:
                fill = response['orderFillTransaction']
                price = float(fill.get('price', 0))
                logger.info(f"  ✅ {symbol}: FILLED @ {price}")
                trades_executed += 1
                positions.append(symbol)
            elif response and 'orderCancelTransaction' in response:
                reason = response['orderCancelTransaction'].get('reason', 'Unknown')
                logger.warning(f"  ⚠️ {symbol}: REJECTED - {reason}")
            else:
                logger.warning(f"  ❌ {symbol}: No response")
            
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"  ❌ {symbol}: {e}")
    
    # Check positions
    logger.info(f"\n--- CURRENT POSITIONS ---")
    open_positions = client.get_open_positions()
    if open_positions and 'positions' in open_positions:
        for pos in open_positions['positions']:
            sym = pos.get('instrument')
            long_units = pos.get('long', {}).get('units', '0')
            short_units = pos.get('short', {}).get('units', '0')
            unrealized = float(pos.get('long', {}).get('unrealizedPL', 0)) + float(pos.get('short', {}).get('unrealizedPL', 0))
            logger.info(f"  {sym}: Long={long_units}, Short={short_units}, P/L=${unrealized:.2f}")
    
    # Wait a moment
    logger.info("\nWaiting 5 seconds before closing positions...")
    time.sleep(5)
    
    # Close all positions
    logger.info(f"\n--- CLOSING ALL POSITIONS ---")
    for symbol in positions:
        try:
            result = client.close_position(symbol, long_units='ALL')
            if result:
                logger.info(f"  ✅ Closed {symbol}")
            time.sleep(0.3)
        except Exception as e:
            logger.error(f"  ❌ Error closing {symbol}: {e}")
    
    # Final balance
    summary2 = client.get_account_summary()
    final_balance = float(summary2['account']['balance'])
    pnl = final_balance - balance
    
    logger.info(f"\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Trades Executed: {trades_executed}")
    logger.info(f"Starting Balance: ${balance:,.2f}")
    logger.info(f"Final Balance: ${final_balance:,.2f}")
    logger.info(f"P/L: ${pnl:+,.2f}")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
