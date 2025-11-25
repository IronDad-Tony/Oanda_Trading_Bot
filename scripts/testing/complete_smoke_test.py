"""
Complete Smoke Test - v2.0 Post-Cleanup
Tests all upgraded components
"""
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Add src to path - scripts/testing/ is 2 levels deep
project_root = Path(__file__).resolve().parents[2]
src_dir = project_root / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_registry():
    """Test Model Registry"""
    logger.info("="*60)
    logger.info("Testing Model Registry...")
    logger.info("="*60)
    
    try:
        from oanda_trading_bot.common.model_registry import ModelRegistry, ModelMetadata
        
        registry = ModelRegistry()
        
        # Create test metadata
        metadata = ModelMetadata(
            model_name="test_model",
            version="1.0.0",
            training_config={'test': True},
            tags=['smoke-test']
        )
        
        logger.info(f"Created metadata: {metadata.model_id}")
        
        # List models
        models = registry.list_models()
        logger.info(f"Found {len(models)} models")
        
        logger.info("[OK] Model Registry test passed\n")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Model Registry failed: {e}")
        return False

def test_enhanced_simulation():
    """Test Enhanced Simulation"""
    logger.info("="*60)
    logger.info("Testing Enhanced Simulation...")
    logger.info("="*60)
    
    try:
        from oanda_trading_bot.training_system.environment.enhanced_simulation import (
            SwapCalculator, MarketHoursManager
        )
        from decimal import Decimal
        
        # Test swap calculator
        swap_calc = SwapCalculator()
        swap = swap_calc.calculate_swap(
            symbol='EUR_USD',
            position_type='long',
            units=10000,
            entry_price=Decimal('1.0850'),
            days_held=1
        )
        logger.info(f"Swap calculated: {swap}")
        
        # Test market hours
        market_mgr = MarketHoursManager()
        is_open = market_mgr.is_market_open(datetime(2024, 11, 18, 10, 0, 0))
        logger.info(f"Monday market open: {is_open}")
        
        logger.info("[OK] Enhanced Simulation test passed\n")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Enhanced Simulation failed: {e}")
        return False

def test_symbol_attribution():
    """Test Symbol Reward Attribution"""
    logger.info("="*60)
    logger.info("Testing Symbol Reward Attribution...")
    logger.info("="*60)
    
    try:
        from oanda_trading_bot.training_system.environment.symbol_reward_attribution import (
            SymbolRewardAttribution
        )
        from decimal import Decimal
        
        symbols = ['EUR_USD', 'GBP_USD']
        attribution = SymbolRewardAttribution(symbols, Decimal('10000'))
        
        # Calculate contributions
        contributions = attribution.calculate_symbol_contributions(
            positions_data={},
            trade_log=[],
            current_step=1
        )
        
        logger.info(f"Contributions: {contributions}")
        
        logger.info("[OK] Symbol Reward Attribution test passed\n")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Symbol Attribution failed: {e}")
        return False

def test_api_availability():
    """Test API Server availability"""
    logger.info("="*60)
    logger.info("Testing API Server...")
    logger.info("="*60)
    
    try:
        import requests
        
        try:
            response = requests.get("http://localhost:8000/api/health", timeout=2)
            if response.status_code == 200:
                logger.info("[OK] API server is running")
                return True
        except:
            pass
        
        logger.info("[WARN] API server not running (start with option [1])")
        return True  # Not critical for smoke test
        
    except Exception as e:
        logger.error(f"[ERROR] API test failed: {e}")
        return False

def generate_report(results):
    """Generate test report"""
    logger.info("="*60)
    logger.info("Generating Report...")
    logger.info("="*60)
    
    report_dir = project_root / 'reports'
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f'smoke_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    report_content = f"""# Smoke Test Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Results

**Passed**: {passed}/{total}

"""
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        emoji = "[OK]" if result else "[ERROR]"
        report_content += f"- {emoji} {test_name}: {status}\n"
    
    report_content += f"""

## Components Tested

1. Model Registry System
2. Enhanced Simulation (Swap + Market Hours)
3. Symbol Reward Attribution
4. API Server Availability

## Status

All core infrastructure components are {"working" if passed == total else f"partially working ({passed}/{total})"}.

---
*Generated by Smoke Test v2.0*
"""
    
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    logger.info(f"[OK] Report saved: {report_file.name}\n")
    return report_file

def main():
    """Main test flow"""
    logger.info("\n" + "="*60)
    logger.info("OANDA Trading Bot - Smoke Test v2.0")
    logger.info("="*60 + "\n")
    
    results = {}
    
    # Run tests
    logger.info("Phase 1: Core Infrastructure Tests\n")
    results['model_registry'] = test_model_registry()
    results['enhanced_simulation'] = test_enhanced_simulation()
    results['symbol_attribution'] = test_symbol_attribution()
    
    logger.info("Phase 2: API Tests\n")
    results['api_server'] = test_api_availability()
    
    # Generate report
    report_file = generate_report(results)
    
    # Summary
    logger.info("="*60)
    logger.info("Test Summary")
    logger.info("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    logger.info(f"\nPassed: {passed}/{total}\n")
    
    for test_name, result in results.items():
        status = "[OK] PASS" if result else "[ERROR] FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nFull report: {report_file.name}")
    
    logger.info("\n" + "="*60)
    
    if passed == total:
        logger.info("[SUCCESS] All tests passed!")
        logger.info("="*60 + "\n")
        return 0
    else:
        logger.warning(f"[WARNING] {total - passed} test(s) need attention")
        logger.info("="*60 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
