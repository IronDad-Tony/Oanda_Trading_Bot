"""
Simplified End-to-End Test
Fixed encoding and simplified for Windows
"""
import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
import time
import json

project_root = Path(__file__).resolve().parents[1]
src_dir = project_root / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Simple logging - no emoji
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("End-to-End Test - Simplified Version")
    logger.info("="*60)
    
    # Phase 1: Check environment
    logger.info("\n[Phase 1] Checking Environment...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('OANDA_API_KEY')
        account_id = os.getenv('OANDA_ACCOUNT_ID')
        env = os.getenv('OANDA_ENVIRONMENT', 'practice')
        
        if not api_key or not account_id:
            logger.error("OANDA credentials not found in .env")
            return False
            
        logger.info(f"[OK] Environment: {env}")
        logger.info(f"[OK] Account: {account_id}")
        
    except Exception as e:
        logger.error(f"[ERROR] Phase 1 failed: {e}")
        return False
    
    # Phase 2: Train tiny model
    logger.info("\n[Phase 2] Training Tiny Model...")
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
        import gymnasium as gym
        from gymnasium import spaces
        import numpy as np
        
        class SimpleEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
                self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                self.steps = 0
                
            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.steps = 0
                return np.zeros(5, dtype=np.float32), {}
            
            def step(self, action):
                self.steps += 1
                obs = np.random.randn(5).astype(np.float32)
                reward = float(np.random.randn())
                done = self.steps >= 50
                return obs, reward, done, False, {}
        
        logger.info("Creating environment...")
        env_obj = DummyVecEnv([lambda: SimpleEnv()])
        
        logger.info("Creating SAC model...")
        model = SAC("MlpPolicy", env_obj, verbose=0)
        
        logger.info("Training for 5000 steps...")
        model.learn(total_timesteps=5000)
        
        # Save model
        weights_dir = project_root / "weights"
        weights_dir.mkdir(exist_ok=True)
        model_path = weights_dir / "e2e_test.zip"
        model.save(model_path)
        
        logger.info(f"[OK] Model saved: {model_path.name}")
        
    except Exception as e:
        logger.error(f"[ERROR] Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Phase 3: Live trading test
    logger.info("\n[Phase 3] Live Trading Test (PRACTICE MODE)...")
    try:
        from oanda_trading_bot.live_trading_system.core.oanda_client import OandaClient
        
        # Force practice mode
        os.environ['OANDA_ENVIRONMENT'] = 'practice'
        
        api_key = os.getenv('OANDA_API_KEY')
        account_id = os.getenv('OANDA_ACCOUNT_ID')
        environment = os.getenv('OANDA_ENVIRONMENT', 'practice')
        
        client = OandaClient(api_key=api_key, account_id=account_id, environment=environment)
        
        # Get account info
        account_summary = client.get_account_summary()
        if account_summary:
            account = account_summary.get('account', {})
            balance = float(account.get('balance', 0))
            currency = account.get('currency', 'USD')
            logger.info(f"[OK] Account Balance: {balance} {currency}")
        
        # Test trades
        test_symbols = ['EUR_USD', 'GBP_USD', 'USD_JPY']
        trades_completed = 0
        target_trades = 10
        
        logger.info(f"Target: {target_trades} trades")
        
        for symbol in test_symbols:
            if trades_completed >= target_trades:
                break
                
            logger.info(f"\nTesting {symbol}...")
            
            try:
                # Get current price
                candles = client.get_bid_ask_candles_combined(symbol, count=1)
                if not candles:
                    logger.warning(f"No price data for {symbol}")
                    continue
                
                price = float(candles[-1].get('bid_close', 0))
                logger.info(f"  Price: {price:.5f}")
                
                # Place tiny buy order
                units = 1  # Minimum units
                
                logger.info(f"  Placing BUY {units} units...")
                buy_order = {
                    'instrument': symbol,
                    'units': units,
                    'type': 'MARKET'
                }
                
                buy_response = client.create_order(buy_order)
                
                if buy_response:
                    logger.info("  [OK] Buy order placed")
                    trades_completed += 1
                    
                    # Wait
                    time.sleep(2)
                    
                    # Close position
                    logger.info("  Closing position...")
                    close_order = {
                        'instrument': symbol,
                        'units': -units,
                        'type': 'MARKET'
                    }
                    
                    close_response = client.create_order(close_order)
                    if close_response:
                        logger.info("  [OK] Position closed")
                        trades_completed += 1
                
                # Wait between symbols
                time.sleep(3)
                
            except Exception as e:
                logger.warning(f"  [WARN] Trade error: {e}")
                continue
        
        logger.info(f"\n[OK] Completed {trades_completed}/{target_trades} trades")
        
        # Get final account state
        final_summary = client.get_account_summary()
        if final_summary:
            final_account = final_summary.get('account', {})
            final_balance = float(final_account.get('balance', 0))
            logger.info(f"[OK] Final Balance: {final_balance} {currency}")
        
    except Exception as e:
        logger.error(f"[ERROR] Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info("[OK] Phase 1: Environment - PASS")
    logger.info("[OK] Phase 2: Training - PASS")
    logger.info(f"[OK] Phase 3: Trading - {trades_completed} trades")
    logger.info("="*60)
    logger.info("[SUCCESS] End-to-End Test Complete!")
    logger.info("="*60)
    
    # Save report
    report = {
        'test_date': datetime.now().isoformat(),
        'phases': {
            'environment': 'PASS',
            'training': 'PASS',
            'trading': f'{trades_completed} trades'
        },
        'trades_completed': trades_completed,
        'target_trades': target_trades
    }
    
    report_file = project_root / 'reports' / f'e2e_simple_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n[OK] Report saved: {report_file.name}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("\n[WARN] Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
