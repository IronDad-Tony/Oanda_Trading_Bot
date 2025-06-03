#!/usr/bin/env python3
"""
測試貨幣轉換修復
驗證 CurrencyDependencyManager 集成是否成功
"""

import sys
import os
from decimal import Decimal
from typing import Dict, Tuple

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, os.path.abspath('.'))

def test_currency_manager_import():
    """測試貨幣管理器導入"""
    try:
        from src.data_manager.currency_manager import CurrencyDependencyManager
        print("✅ CurrencyDependencyManager 導入成功")
        
        # 創建實例
        manager = CurrencyDependencyManager("AUD", apply_oanda_markup=True)
        print(f"✅ 貨幣管理器實例創建成功 - 賬戶貨幣: {manager.account_currency}")
        return manager
        
    except ImportError as e:
        print(f"❌ CurrencyDependencyManager 導入失敗: {e}")
        return None
    except Exception as e:
        print(f"❌ 貨幣管理器創建失敗: {e}")
        return None

def test_currency_conversion():
    """測試貨幣轉換功能"""
    manager = test_currency_manager_import()
    if not manager:
        return False
    
    # 模擬價格數據
    mock_prices_map: Dict[str, Tuple[Decimal, Decimal]] = {
        "EUR_USD": (Decimal("1.0850"), Decimal("1.0852")),
        "USD_JPY": (Decimal("150.20"), Decimal("150.25")),
        "GBP_USD": (Decimal("1.2750"), Decimal("1.2755")),
        "AUD_USD": (Decimal("0.6650"), Decimal("0.6655")),
        "USD_CAD": (Decimal("1.3580"), Decimal("1.3585")),
    }
    
    print("\n🧪 測試貨幣轉換功能:")
    
    # 測試案例
    test_cases = [
        ("USD", "AUD"),  # 直接轉換
        ("EUR", "AUD"),  # 通過USD中轉
        ("JPY", "AUD"),  # 通過USD中轉
        ("AUD", "AUD"),  # 相同貨幣
    ]
    
    success_count = 0
    for from_curr, to_curr in test_cases:
        try:
            rate = manager.convert_to_account_currency(from_curr, mock_prices_map)
            print(f"✅ {from_curr} → {to_curr}: {rate:.6f}")
            success_count += 1
        except Exception as e:
            print(f"❌ {from_curr} → {to_curr}: 失敗 - {e}")
    
    print(f"\n📊 測試結果: {success_count}/{len(test_cases)} 通過")
    return success_count == len(test_cases)

def test_markup_application():
    """測試OANDA標記應用"""
    print("\n🏷️ 測試OANDA標記應用:")
    
    try:
        from src.data_manager.currency_manager import CurrencyDependencyManager
        
        # 測試有標記和無標記的差異
        manager_with_markup = CurrencyDependencyManager("AUD", apply_oanda_markup=True)
        manager_without_markup = CurrencyDependencyManager("AUD", apply_oanda_markup=False)
    except ImportError as e:
        print(f"❌ CurrencyDependencyManager 導入失敗: {e}")
        return False
    
    mock_prices_map = {
        "USD_AUD": (Decimal("1.5000"), Decimal("1.5010")),
    }
    
    try:
        rate_with_markup = manager_with_markup.convert_to_account_currency("USD", mock_prices_map)
        rate_without_markup = manager_without_markup.convert_to_account_currency("USD", mock_prices_map)
        
        print(f"✅ 有標記匯率: {rate_with_markup:.6f}")
        print(f"✅ 無標記匯率: {rate_without_markup:.6f}")
        
        # 驗證標記確實應用了
        if rate_with_markup != rate_without_markup:
            markup_diff = abs(rate_with_markup - rate_without_markup)
            print(f"✅ 標記差異: {markup_diff:.6f} ({markup_diff/rate_without_markup*100:.2f}%)")
            return True
        else:
            print("❌ 標記未生效")
            return False
            
    except Exception as e:
        print(f"❌ 標記測試失敗: {e}")
        return False

def test_trading_env_integration():
    """測試交易環境集成"""
    print("\n🔧 測試交易環境集成:")
    
    try:
        # 檢查 trading_env.py 是否能正確導入並使用新的貨幣管理器
        from src.environment.trading_env import UniversalTradingEnvV4
        print("✅ UniversalTradingEnvV4 導入成功")
        
        # 注意：這裡不實際創建環境實例，因為需要複雜的依賴
        # 我們只是驗證導入是否成功
        return True
        
    except ImportError as e:
        print(f"❌ 交易環境導入失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ 交易環境測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🔄 開始貨幣轉換修復驗證...")
    print("=" * 50)
    
    results = []
    
    # 執行所有測試
    results.append(("貨幣轉換功能", test_currency_conversion()))
    results.append(("OANDA標記應用", test_markup_application()))
    results.append(("交易環境集成", test_trading_env_integration()))
    
    # 總結結果
    print("\n" + "=" * 50)
    print("📋 測試總結:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 整體結果: {passed}/{total} 測試通過")
    
    if passed == total:
        print("🎉 所有測試通過！貨幣轉換修復成功！")
        return True
    else:
        print("⚠️ 部分測試失敗，需要進一步檢查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
