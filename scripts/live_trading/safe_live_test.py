"""
Safe Live Trading Test with Trained Model
Safe execution with order tracking and duplicate prevention
"""
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import time
from decimal import Decimal
from typing import Dict, Optional
import json

# Add src to path
project_root = Path(__file__).resolve().parents[2]
src_dir = project_root / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'live_trading_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OrderTracker:
    """Track orders to prevent duplicates"""
    
    def __init__(self):
        self.pending_orders = {}  # symbol -> order_id
        self.completed_trades = []
        self.order_lock = {}  # symbol -> bool (True = locked)
        
    def is_locked(self, symbol: str) -> bool:
        """Check if symbol is locked (has pending order)"""
        return self.order_lock.get(symbol, False)
    
    def lock_symbol(self, symbol: str, order_id: str):
        """Lock symbol for order"""
        logger.info(f"[LOCK] {symbol} - Order ID: {order_id}")
        self.order_lock[symbol] = True
        self.pending_orders[symbol] = order_id
    
    def unlock_symbol(self, symbol: str):
        """Unlock symbol after order completion"""
        logger.info(f"[UNLOCK] {symbol}")
        self.order_lock[symbol] = False
        if symbol in self.pending_orders:
            del self.pending_orders[symbol]
    
    def add_trade(self, trade_info: dict):
        """Record completed trade"""
        self.completed_trades.append(trade_info)
    
    def get_summary(self) -> dict:
        """Get trading summary"""
        total_trades = len(self.completed_trades)
        
        if total_trades == 0:
            return {
                'total_trades': 0,
                'symbols_traded': [],
                'total_pnl': 0,
                'trades': []
            }
        
        symbols = set(t['symbol'] for t in self.completed_trades)
        total_pnl = sum(t.get('pnl', 0) for t in self.completed_trades)
        
        return {
            'total_trades': total_trades,
            'symbols_traded': list(symbols),
            'total_pnl': total_pnl,
            'trades': self.completed_trades
        }


class SafeLiveTrader:
    """Safe live trading with model"""
    
    def __init__(self):
        self.client = None
        self.model = None
        self.tracker = OrderTracker()
        self.target_trades = 10
        
        # Popular symbols for trading
        self.symbols = [
            'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD',
            'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY', 'AUD_JPY'
        ]
        
    def initialize(self):
        """Initialize client and model"""
        logger.info("="*80)
        logger.info("INITIALIZING SAFE LIVE TRADING TEST")
        logger.info("="*80)
        
        # Load environment
        from dotenv import load_dotenv
        load_dotenv()
        
        # Force PRACTICE mode for safety
        os.environ['OANDA_ENVIRONMENT'] = 'practice'
        logger.info("[SAFETY] Forced PRACTICE mode")
        
        # Initialize OANDA client
        from oanda_trading_bot.live_trading_system.core.oanda_client import OandaClient
        
        api_key = os.getenv('OANDA_API_KEY')
        account_id = os.getenv('OANDA_ACCOUNT_ID')
        
        self.client = OandaClient(
            api_key=api_key,
            account_id=account_id,
            environment='practice'
        )
        
        logger.info("[OK] OANDA Client initialized")
        
        # Get account info
        account = self.client.get_account_summary()
        if account:
            acc_info = account.get('account', {})
            balance = float(acc_info.get('balance', 0))
            currency = acc_info.get('currency', 'USD')
            logger.info(f"[OK] Account Balance: {balance:.2f} {currency}")
        
        # Load model (we'll use simple logic if model load fails)
        try:
            from stable_baselines3 import SAC
            model_path = project_root / 'weights' / 'e2e_test.zip'
            if model_path.exists():
                self.model = SAC.load(model_path)
                logger.info(f"[OK] Model loaded: {model_path.name}")
            else:
                logger.warning("[WARN] No model found, will use simple logic")
        except Exception as e:
            logger.warning(f"[WARN] Model load failed: {e}")
        
        logger.info(f"\n[CONFIG] Target trades: {self.target_trades}")
        logger.info(f"[CONFIG] Symbols: {self.symbols}\n")
        
        return True
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            candles = self.client.get_bid_ask_candles_combined(symbol, count=1)
            
            # Debug: log what we got
            if not candles:
                logger.debug(f"[DEBUG] {symbol}: No candles returned")
                return None
            
            if len(candles) == 0:
                logger.debug(f"[DEBUG] {symbol}: Empty candles list")
                return None
            
            # Try different price fields
            last_candle = candles[-1]
            
            # Try bid_close first
            price = last_candle.get('bid_close')
            if price and float(price) > 0:
                return float(price)
            
            # Try ask_close
            price = last_candle.get('ask_close')
            if price and float(price) > 0:
                return float(price)
            
            # Try mid_close
            price = last_candle.get('mid_close')
            if price and float(price) > 0:
                return float(price)
            
            # Try close (regular)
            price = last_candle.get('close')
            if price and float(price) > 0:
                return float(price)
            
            logger.debug(f"[DEBUG] {symbol}: Candle data: {last_candle}")
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get price for {symbol}: {e}")
            return None
    
    def place_order_safe(self, symbol: str, units: int) -> Optional[dict]:
        """Place order with safety checks"""
        
        # Check if symbol is locked
        if self.tracker.is_locked(symbol):
            logger.warning(f"[BLOCKED] {symbol} is locked (pending order)")
            return None
        
        # Lock symbol
        order_id = f"temp_{symbol}_{int(time.time())}"
        self.tracker.lock_symbol(symbol, order_id)
        
        try:
            logger.info(f"[ORDER] Placing MARKET order: {symbol} {units} units")
            
            # Use correct API signature: create_order(instrument, units, ...)
            response = self.client.create_order(
                instrument=symbol,
                units=units
            )
            
            if response:
                # Extract order details
                fill_tx = response.get('orderFillTransaction', {})
                order_id = fill_tx.get('id', order_id)
                fill_price = float(fill_tx.get('price', 0))
                
                logger.info(f"[FILL] Order filled at {fill_price:.5f}")
                
                return {
                    'order_id': order_id,
                    'symbol': symbol,
                    'units': units,
                    'fill_price': fill_price,
                    'response': response
                }
            else:
                logger.error(f"[ERROR] No response from order placement")
                return None
                
        except Exception as e:
            logger.error(f"[ERROR] Order placement failed: {e}")
            return None
        finally:
            # Always unlock after attempt
            self.tracker.unlock_symbol(symbol)
    
    def execute_trade_cycle(self, symbol: str) -> bool:
        """Execute one complete trade cycle: open -> close"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRADE CYCLE: {symbol}")
        logger.info(f"{'='*80}")
        
        # Get current price
        current_price = self.get_current_price(symbol)
        if not current_price:
            logger.warning(f"[SKIP] No price available for {symbol}")
            return False
        
        logger.info(f"[PRICE] Current: {current_price:.5f}")
        
        # Small position size for safety
        units = 10  # Very small
        
        # Step 1: Open position (BUY)
        logger.info(f"\n[STEP 1] Opening position...")
        buy_order = self.place_order_safe(symbol, units)
        
        if not buy_order:
            logger.error(f"[FAILED] Could not open position")
            return False
        
        # Wait for confirmation
        time.sleep(2)
        
        # Verify position opened
        try:
            positions = self.client.get_positions()
            if positions:
                pos_list = positions.get('positions', [])
                has_position = any(
                    p.get('instrument') == symbol and float(p.get('long', {}).get('units', 0)) > 0
                    for p in pos_list
                )
                if has_position:
                    logger.info(f"[OK] Position opened and verified")
                else:
                    logger.warning(f"[WARN] Position not found in account")
        except Exception as e:
            logger.warning(f"[WARN] Could not verify position: {e}")
        
        # Step 2: Close position (SELL)
        logger.info(f"\n[STEP 2] Closing position...")
        
        # Wait a bit before closing
        time.sleep(3)
        
        close_order = self.place_order_safe(symbol, -units)
        
        if not close_order:
            logger.error(f"[FAILED] Could not close position")
            return False
        
        # Calculate P&L (approximate)
        buy_price = float(buy_order.get('fill_price', current_price))
        sell_price = float(close_order.get('fill_price', current_price))
        pnl = (sell_price - buy_price) * units
        
        # Record trade
        trade_info = {
            'symbol': symbol,
            'open_time': datetime.now().isoformat(),
            'buy_price': buy_price,
            'sell_price': sell_price,
            'units': units,
            'pnl': pnl,
            'buy_order_id': buy_order.get('order_id'),
            'sell_order_id': close_order.get('order_id')
        }
        
        self.tracker.add_trade(trade_info)
        
        logger.info(f"\n[COMPLETE] Trade finished")
        logger.info(f"  Buy: {buy_price:.5f}")
        logger.info(f"  Sell: {sell_price:.5f}")
        logger.info(f"  P&L: {pnl:.2f}")
        
        return True
    
    def run_trading_session(self):
        """Run complete trading session"""
        
        logger.info("\n" + "="*80)
        logger.info("STARTING TRADING SESSION")
        logger.info("="*80)
        logger.info(f"Target: {self.target_trades} trades")
        logger.info(f"Symbols: {len(self.symbols)}")
        
        trades_completed = 0
        symbol_index = 0
        max_attempts = len(self.symbols) * 2  # Safety limit
        attempts = 0
        
        while trades_completed < self.target_trades and attempts < max_attempts:
            attempts += 1
            
            # Get next symbol
            symbol = self.symbols[symbol_index % len(self.symbols)]
            symbol_index += 1
            
            # Execute trade
            success = self.execute_trade_cycle(symbol)
            
            if success:
                trades_completed += 1
                logger.info(f"\n[PROGRESS] Completed {trades_completed}/{self.target_trades} trades")
            
            # Rate limiting between trades
            if trades_completed < self.target_trades:
                logger.info(f"\n[WAIT] Cooling down for 5 seconds...")
                time.sleep(5)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRADING SESSION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Trades executed: {trades_completed}/{self.target_trades}")
        
        return trades_completed >= self.target_trades
    
    def generate_summary(self):
        """Generate and display trading summary"""
        
        summary = self.tracker.get_summary()
        
        logger.info("\n" + "="*80)
        logger.info("TRADING SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\nTotal Trades: {summary['total_trades']}")
        logger.info(f"Symbols Traded: {', '.join(summary['symbols_traded'])}")
        logger.info(f"Total P&L: {summary['total_pnl']:.2f}")
        
        logger.info("\n" + "-"*80)
        logger.info("TRADE DETAILS")
        logger.info("-"*80)
        
        for i, trade in enumerate(summary['trades'], 1):
            logger.info(f"\nTrade #{i}")
            logger.info(f"  Symbol: {trade['symbol']}")
            logger.info(f"  Time: {trade['open_time']}")
            logger.info(f"  Buy: {trade['buy_price']:.5f}")
            logger.info(f"  Sell: {trade['sell_price']:.5f}")
            logger.info(f"  Units: {trade['units']}")
            logger.info(f"  P&L: {trade['pnl']:.2f}")
        
        # Save summary to file
        report_file = project_root / 'reports' / f'live_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n[SAVED] Report: {report_file.name}")
        
        return summary


def main():
    """Main execution"""
    
    trader = SafeLiveTrader()
    
    try:
        # Initialize
        if not trader.initialize():
            logger.error("Initialization failed")
            return 1
        
        # Confirm with user (via log)
        logger.info("\n" + "!"*80)
        logger.info("SAFETY CONFIRMATION")
        logger.info("!"*80)
        logger.info("Mode: PRACTICE (Safe)")
        logger.info("Trade Size: 10 units (Very small)")
        logger.info("Target: 10 trades")
        logger.info("!"*80)
        
        # Small delay for user to see
        time.sleep(3)
        
        # Run trading
        success = trader.run_trading_session()
        
        # Generate summary
        summary = trader.generate_summary()
        
        logger.info("\n" + "="*80)
        if success:
            logger.info("[SUCCESS] Trading session completed successfully!")
        else:
            logger.warning("[PARTIAL] Trading session completed with some issues")
        logger.info("="*80)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.warning("\n[INTERRUPTED] Trading stopped by user")
        trader.generate_summary()
        return 1
    except Exception as e:
        logger.error(f"\n[ERROR] Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
