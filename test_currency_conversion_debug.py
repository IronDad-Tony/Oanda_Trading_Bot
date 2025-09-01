#!/usr/bin/env python3
"""
測試腳本：重現UK100到AUD貨幣轉換錯誤並收集調試信息
"""
import sys
import os
from pathlib import Path

# 添加項目路徑
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    from src.oanda_trading_bot.training_system.data_manager.currency_manager import CurrencyDependencyManager
    from src.oanda_trading_bot.common.instrument_info_manager import InstrumentInfoManager
    from decimal import Decimal
    import logging

    # 設置日誌
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def test_currency_conversion():
        print("=== 測試UK100到AUD貨幣轉換問題 ===")

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

        # 初始化貨幣管理器
        currency_manager = CurrencyDependencyManager(account_currency='AUD')

        print("\n1. 測試正確的轉換：GBP -> AUD")
        try:
            rate_gbp_to_aud = currency_manager.convert_to_account_currency('GBP', current_prices_map)
            print(f"GBP -> AUD 轉換率: {rate_gbp_to_aud}")
        except Exception as e:
            print(f"GBP -> AUD 轉換失敗: {e}")

        print("\n2. 測試錯誤的轉換：UK100 -> AUD（這應該失敗）")
        try:
            rate_uk100_to_aud = currency_manager.convert_to_account_currency('UK100', current_prices_map)
            print(f"UK100 -> AUD 轉換率: {rate_uk100_to_aud}")
        except Exception as e:
            print(f"UK100 -> AUD 轉換失敗: {e}")

        print("\n3. 測試InstrumentInfoManager")
        try:
            instrument_manager = InstrumentInfoManager()

            # 測試獲取UK100_GBP的詳細信息
            details_uk100_gbp = instrument_manager.get_details('UK100_GBP')
            if details_uk100_gbp:
                print(f"UK100_GBP 詳細信息: {details_uk100_gbp}")
                print(f"報價貨幣: {details_uk100_gbp.quote_currency}")
            else:
                print("無法獲取UK100_GBP的詳細信息")

            # 測試獲取UK100的詳細信息（這應該返回None）
            details_uk100 = instrument_manager.get_details('UK100')
            if details_uk100:
                print(f"UK100 詳細信息: {details_uk100}")
            else:
                print("無法獲取UK100的詳細信息（這是預期的）")

        except Exception as e:
            print(f"InstrumentInfoManager測試失敗: {e}")

        print("\n4. 測試完整的轉換路徑")
        try:
            # 模擬UK100_GBP的報價貨幣是GBP，然後轉換GBP到AUD
            print("UK100_GBP -> GBP -> AUD 的轉換路徑:")
            print("步驟1: UK100_GBP的報價貨幣是GBP")
            print("步驟2: 將GBP轉換為AUD")

            # 這是正確的轉換方式
            rate_gbp_to_aud = currency_manager.convert_to_account_currency('GBP', current_prices_map)
            print(f"最終轉換率 (GBP -> AUD): {rate_gbp_to_aud}")

        except Exception as e:
            print(f"完整轉換路徑測試失敗: {e}")

    if __name__ == "__main__":
        test_currency_conversion()

except ImportError as e:
    print(f"導入失敗: {e}")
    print("請確保項目路徑設置正確")
    sys.exit(1)