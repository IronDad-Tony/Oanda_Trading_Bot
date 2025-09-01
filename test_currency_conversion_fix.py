#!/usr/bin/env python3
"""
測試腳本：驗證UK100到AUD貨幣轉換修復效果
"""
import sys
import os
from pathlib import Path
from decimal import Decimal

# 添加項目路徑
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    from src.oanda_trading_bot.training_system.data_manager.currency_manager import CurrencyDependencyManager
    import logging

    # 設置日誌
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def test_currency_conversion_fix():
        print("=== 測試UK100到AUD貨幣轉換修復效果 ===")

        # 模擬當前價格映射（基於錯誤信息）
        current_prices_map = {
            'AU200_AUD': (Decimal('7000.0'), Decimal('7010.0')),
            'AUD_USD': (Decimal('0.65'), Decimal('0.66')),
            'CN50_USD': (Decimal('15000.0'), Decimal('15010.0')),
            'GBP_AUD': (Decimal('1.85'), Decimal('1.86')),  # 這是我們需要的轉換對
            'NAS100_USD': (Decimal('15000.0'), Decimal('15010.0')),
            'SPX500_USD': (Decimal('4500.0'), Decimal('4510.0')),
            'UK100_GBP': (Decimal('8000.0'), Decimal('8010.0')),  # UK100的價格
            'US2000_USD': (Decimal('2000.0'), Decimal('2010.0'))
        }

        print(f"測試用的價格映射鍵: {list(current_prices_map.keys())}")

        # 初始化貨幣管理器
        currency_manager = CurrencyDependencyManager(account_currency='AUD')

        print("\n1. 測試正確的轉換：GBP -> AUD")
        try:
            rate_gbp_to_aud = currency_manager.convert_to_account_currency('GBP', current_prices_map)
            print(f"GBP -> AUD 轉換率: {rate_gbp_to_aud}")
        except Exception as e:
            print(f"GBP -> AUD 轉換失敗: {e}")

        print("\n2. 測試錯誤的轉換：UK100 -> AUD（應該觸發修復邏輯）")
        try:
            rate_uk100_to_aud = currency_manager.convert_to_account_currency('UK100', current_prices_map)
            print(f"UK100 -> AUD 轉換率: {rate_uk100_to_aud}")
            if rate_uk100_to_aud == Decimal('1.0'):
                print("❌ 修復失敗：仍然返回安全值1.0")
            else:
                print("✅ 修復成功：自動修復為GBP -> AUD轉換")
        except Exception as e:
            print(f"UK100 -> AUD 轉換失敗: {e}")

        print("\n3. 測試其他指數代碼")
        test_indices = ['SPX', 'NAS', 'US30', 'DE30']
        for index in test_indices:
            print(f"\n測試 {index} -> AUD:")
            try:
                rate = currency_manager.convert_to_account_currency(index, current_prices_map)
                print(f"{index} -> AUD 轉換率: {rate}")
            except Exception as e:
                print(f"{index} -> AUD 轉換失敗: {e}")

        print("\n4. 測試正常貨幣轉換")
        normal_currencies = ['USD', 'GBP', 'EUR']
        for currency in normal_currencies:
            print(f"\n測試 {currency} -> AUD:")
            try:
                rate = currency_manager.convert_to_account_currency(currency, current_prices_map)
                print(f"{currency} -> AUD 轉換率: {rate}")
            except Exception as e:
                print(f"{currency} -> AUD 轉換失敗: {e}")

    if __name__ == "__main__":
        test_currency_conversion_fix()

except ImportError as e:
    print(f"導入失敗: {e}")
    print("請確保項目路徑設置正確")
    sys.exit(1)