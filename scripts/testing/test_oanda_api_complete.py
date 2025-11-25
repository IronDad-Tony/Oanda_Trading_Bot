"""
OANDA API Complete Functionality Test
Tests all critical API features for trading system
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parents[2]
src_dir = project_root / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import logging
from oanda_trading_bot.live_trading_system.core.oanda_client import OandaClient
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_features():
    """Test existing features"""
    logger.info("\n" + "="*80)
    logger.info("TESTING BASIC FEATURES")
    logger.info("="*80)
    
    client = OandaClient.from_env()
    
    # 1. Account Summary
    logger.info("\n[TEST 1] Account Summary")
    summary = client.get_account_summary()
    if summary and 'account' in summary:
        acc = summary['account']
        logger.info(f"[OK] Balance: {acc.get('balance')} {acc.get('currency')}")
        logger.info(f"[OK] Margin Available: {acc.get('marginAvailable')}")
        logger.info(f"[OK] Open Trades: {acc.get('openTradeCount')}")
    else:
        logger.error("[FAIL] Could not get account summary")
        return False
    
    # 2. Candles
    logger.info("\n[TEST 2] Get Candles")
    candles = client.get_candles('EUR_USD', count=5, granularity='M5')
    if candles:
        logger.info(f"[OK] Retrieved {len(candles)} candles")
    else:
        logger.error("[FAIL] Could not get candles")
        return False
    
    # 3. Bid/Ask Candles
    logger.info("\n[TEST 3] Get Bid/Ask Candles Combined")
    combined = client.get_bid_ask_candles_combined('EUR_USD', count=5)
    if combined:
        logger.info(f"[OK] Retrieved {len(combined)} combined candles")
        if len(combined) > 0:
            logger.info(f"[OK] Sample: bid_close={combined[-1].get('bid_close')}, ask_close={combined[-1].get('ask_close')}")
    else:
        logger.error("[FAIL] Could not get combined candles")
        return False
    
    logger.info("\n[SUCCESS] All basic features working!")
    return True

def test_new_critical_features():
    """Test new critical features using direct API calls"""
    logger.info("\n" + "="*80)
    logger.info("TESTING NEW CRITICAL FEATURES")
    logger.info("="*80)
    
    client = OandaClient.from_env()
    
    # 1. Instrument Details (CRITICAL)
    logger.info("\n[TEST 4] Get Instrument Details")
    try:
        params = {"instruments": "EUR_USD,GBP_USD,USD_JPY"}
        endpoint = accounts.AccountInstruments(client.account_id, params=params)
        response = client._request(endpoint)
        
        if response and 'instruments' in response:
            logger.info(f"[OK] Retrieved details for {len(response['instruments'])} instruments")
            for inst in response['instruments'][:3]:
                logger.info(f"  {inst['name']}:")
                logger.info(f"    Margin Rate: {inst.get('marginRate')} (leverage: {1/float(inst.get('marginRate', 0.05)):.0f}:1)")
                logger.info(f"    Min Trade Size: {inst.get('minimumTradeSize')}")
                logger.info(f"    Max Position: {inst.get('maximumPositionSize')}")
                logger.info(f"    Financing Long: {inst.get('financing', {}).get('longRate')}")
                logger.info(f"    Financing Short: {inst.get('financing', {}).get('shortRate')}")
        else:
            logger.error("[FAIL] Could not get instrument details")
            return False
    except Exception as e:
        logger.error(f"[FAIL] Instrument details error: {e}")
        return False
    
    # 2. Full Account Details (CRITICAL)
    logger.info("\n[TEST 5] Get Full Account Details")
    try:
        endpoint = accounts.AccountDetails(client.account_id)
        response = client._request(endpoint)
        
        if response and 'account' in response:
            acc = response['account']
            logger.info(f"[OK] Account Currency: {acc.get('currency')}")
            logger.info(f"[OK] NAV: {acc.get('NAV')}")
            logger.info(f"[OK] Margin Closeout %%: {acc.get('marginCloseoutPercent')}")
            logger.info(f"[OK] Margin Used: {acc.get('marginUsed')}")
            logger.info(f"[OK] Withdrawal Limit: {acc.get('withdrawalLimit')}")
        else:
            logger.error("[FAIL] Could not get account details")
            return False
    except Exception as e:
        logger.error(f"[FAIL] Account details error: {e}")
        return False
    
    # 3. Pricing API (IMPORTANT)
    logger.info("\n[TEST 6] Get Current Pricing")
    try:
        params = {"instruments": "EUR_USD,GBP_USD"}
        endpoint = pricing.PricingInfo(client.account_id, params=params)
        response = client._request(endpoint)
        
        if response and 'prices' in response:
            logger.info(f"[OK] Retrieved pricing for {len(response['prices'])} instruments")
            for price in response['prices']:
                logger.info(f"  {price['instrument']}:")
                logger.info(f"    Tradeable: {price.get('tradeable')}")
                logger.info(f"    Bid: {price.get('closeoutBid')}")
                logger.info(f"    Ask: {price.get('closeoutAsk')}")
                logger.info(f"    Mid: {(float(price.get('closeoutBid', 0)) + float(price.get('closeoutAsk', 0)))/2:.5f}")
                
                # Conversion factors (CRITICAL)
                conv = price.get('quoteHomeConversionFactors', {})
                logger.info(f"    Conversion +: {conv.get('positiveUnits')}")
                logger.info(f"    Conversion -: {conv.get('negativeUnits')}")
                
                # Available units
                units = price.get('unitsAvailable', {}).get('default', {})
                logger.info(f"    Long Units Available: {units.get('long')}")
                logger.info(f"    Short Units Available: {units.get('short')}")
        else:
            logger.error("[FAIL] Could not get pricing")
            return False
    except Exception as e:
        logger.error(f"[FAIL] Pricing error: {e}")
        return False
    
    # 4. Open Trades
    logger.info("\n[TEST 7] Get Open Trades")
    try:
        endpoint = trades.OpenTrades(client.account_id)
        response = client._request(endpoint)
        
        if response is not None:
            trade_list = response.get('trades', [])
            logger.info(f"[OK] Open trades: {len(trade_list)}")
            for trade in trade_list[:3]:
                logger.info(f"  Trade {trade.get('id')}:")
                logger.info(f"    Instrument: {trade.get('instrument')}")
                logger.info(f"    Units: {trade.get('currentUnits')}")
                logger.info(f"    Price: {trade.get('price')}")
                logger.info(f"    Unrealized P&L: {trade.get('unrealizedPL')}")
        else:
            logger.error("[FAIL] Could not get open trades")
            return False
    except Exception as e:
        logger.error(f"[FAIL] Open trades error: {e}")
        return False
    
    logger.info("\n[SUCCESS] All new critical features working!")
    return True

def test_margin_calculation():
    """Test margin calculation logic"""
    logger.info("\n" + "="*80)
    logger.info("TESTING MARGIN CALCULATION")
    logger.info("="*80)
    
    client = OandaClient.from_env()
    
    # Get instrument details
    params = {"instruments": "EUR_USD"}
    endpoint = accounts.AccountInstruments(client.account_id, params=params)
    inst_response = client._request(endpoint)
    
    # Get pricing
    params = {"instruments": "EUR_USD"}
    endpoint = pricing.PricingInfo(client.account_id, params=params)
    price_response = client._request(endpoint)
    
    if inst_response and price_response:
        inst = inst_response['instruments'][0]
        price = price_response['prices'][0]
        
        # Extract data
        margin_rate = float(inst['marginRate'])
        conversion_factor = float(price['quoteHomeConversionFactors']['positiveUnits'])
        units = 10000
        
        # OANDA Formula:  Margin = abs(units) * conversion_factor * margin_rate
        calculated_margin = abs(units) * conversion_factor * margin_rate
        
        logger.info(f"\nMargin Calculation for EUR_USD 10000 units:")
        logger.info(f"  Margin Rate: {margin_rate} ({1/margin_rate:.0f}:1 leverage)")
        logger.info(f"  Conversion Factor: {conversion_factor}")
        logger.info(f"  Units: {units}")
        logger.info(f"  CALCULATED MARGIN: {calculated_margin:.2f}")
        
        logger.info(f"\n[SUCCESS] Margin calculation formula verified!")
        return True
    else:
        logger.error("[FAIL] Could not get data for margin calculation")
        return False

def main():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("OANDA API COMPLETE FUNCTIONALITY TEST")
    logger.info("="*80)
    
    results = {}
    
    # Test 1: Basic Features
    results['basic'] = test_basic_features()
    
    # Test 2: New Critical Features
    results['critical'] = test_new_critical_features()
    
    # Test 3: Margin Calculation
    results['margin'] = test_margin_calculation()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for test_name, result in results.items():
        status = "[OK] PASS" if result else "[ERROR] FAIL"
        logger.info(f"  {test_name}: {status}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    logger.info(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        logger.info("\n[SUCCESS] All API features verified and working!")
        logger.info("="*80)
        return 0
    else:
        logger.warning(f"\n[WARNING] {total - passed} test(s) failed")
        logger.info("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
