#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oanda規則驗證測試
驗證我們的實現是否完全符合Oanda官方規則
"""

import sys
import os
from decimal import Decimal
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_manager.currency_manager_fixed import CurrencyDependencyManager

def test_oanda_conversion_fee_formula():
    """測試Oanda貨幣轉換費用公式"""
    print("=== 測試Oanda貨幣轉換費用公式 ===")
    
    # 模擬官方例子：GBP/USD = 1.29530 (bid) / 1.29550 (ask)
    manager = CurrencyDependencyManager("USD", apply_oanda_markup=True)
    
    bid = Decimal('1.29530')
    ask = Decimal('1.29550')
    
    # 計算中間價格
    midpoint = manager.get_midpoint_rate(bid, ask)
    print(f"中間價格: {midpoint}")
    
    # 應用轉換費用（獲利情況）
    conversion_rate = manager.apply_conversion_fee(midpoint, is_credit=True)
    print(f"轉換率（獲利）: {conversion_rate}")
    
    # 根據官方公式計算期望值
    expected_rate = midpoint * (Decimal('1') - Decimal('0.005'))
    print(f"期望轉換率: {expected_rate}")
    
    # 驗證
    assert abs(conversion_rate - expected_rate) < Decimal('0.000001'), "轉換費用計算不正確"
    print("✓ 轉換費用公式驗證通過")
    
    # 測試官方例子的具體數值
    # 100 GBP -> USD，期望結果：128.89 USD
    amount_gbp = Decimal('100')
    converted_amount = amount_gbp * conversion_rate
    expected_converted = Decimal('128.89')
    
    print(f"100 GBP -> USD: {converted_amount:.2f}")
    print(f"期望結果: {expected_converted}")
    
    # 允許小數點誤差
    assert abs(converted_amount - expected_converted) < Decimal('0.1'), "官方例子驗證失敗"
    print("✓ 官方例子驗證通過")

def test_bid_ask_usage():
    """測試Bid/Ask價格使用規則"""
    print("\n=== 測試Bid/Ask價格使用規則 ===")
    
    price_data = {
        "EUR_USD": (Decimal('1.0850'), Decimal('1.0852')),  # bid, ask
        "GBP_USD": (Decimal('1.2530'), Decimal('1.2532')),
        "USD_JPY": (Decimal('149.80'), Decimal('149.82')),
    }
    
    manager = CurrencyDependencyManager("USD", apply_oanda_markup=False)
    
    # 測試直接對：買入EUR（使用ask價格）
    eur_usd_rate = manager.get_specific_rate("EUR", "USD", price_data, is_for_conversion=False)
    expected_eur_usd = price_data["EUR_USD"][1]  # ask price
    print(f"EUR/USD 交易匯率: {eur_usd_rate} (期望: {expected_eur_usd})")
    assert eur_usd_rate == expected_eur_usd, "應該使用ask價格"
    
    # 測試反向對：買入USD賣出EUR（使用bid價格的倒數）
    usd_eur_rate = manager.get_specific_rate("USD", "EUR", price_data, is_for_conversion=False)
    expected_usd_eur = Decimal('1') / price_data["EUR_USD"][0]  # 1/bid price
    print(f"USD/EUR 交易匯率: {usd_eur_rate} (期望: {expected_usd_eur})")
    assert abs(usd_eur_rate - expected_usd_eur) < Decimal('0.000001'), "應該使用bid價格的倒數"
    
    print("✓ Bid/Ask價格使用規則驗證通過")

def test_cross_rate_calculation():
    """測試交叉匯率計算"""
    print("\n=== 測試交叉匯率計算 ===")
    
    price_data = {
        "EUR_USD": (Decimal('1.0850'), Decimal('1.0852')),
        "GBP_USD": (Decimal('1.2530'), Decimal('1.2532')),
        "AUD_USD": (Decimal('0.6680'), Decimal('0.6682')),
    }
    
    manager = CurrencyDependencyManager("USD", apply_oanda_markup=False)
    
    # 測試EUR/GBP交叉匯率（通過USD）
    eur_gbp_rate = manager.get_specific_rate("EUR", "GBP", price_data, is_for_conversion=False)
    
    # 手動計算：EUR/USD ÷ GBP/USD
    eur_usd = price_data["EUR_USD"][1]  # ask
    gbp_usd = price_data["GBP_USD"][1]  # ask  
    expected_eur_gbp = eur_usd / gbp_usd
    
    print(f"EUR/GBP 計算匯率: {eur_gbp_rate}")
    print(f"手動計算結果: {expected_eur_gbp}")
    
    assert abs(eur_gbp_rate - expected_eur_gbp) < Decimal('0.000001'), "交叉匯率計算錯誤"
    print("✓ 交叉匯率計算驗證通過")

def test_multiple_account_currencies():
    """測試多種賬戶貨幣的轉換"""
    print("\n=== 測試多種賬戶貨幣的轉換 ===")
    
    price_data = {
        "EUR_USD": (Decimal('1.0850'), Decimal('1.0852')),
        "GBP_USD": (Decimal('1.2530'), Decimal('1.2532')),
        "USD_JPY": (Decimal('149.80'), Decimal('149.82')),
        "AUD_USD": (Decimal('0.6680'), Decimal('0.6682')),
        "EUR_GBP": (Decimal('0.8660'), Decimal('0.8662')),
        "EUR_JPY": (Decimal('162.50'), Decimal('162.55')),
    }
    
    # 測試不同賬戶貨幣
    test_cases = [
        ("EUR", "USD"),
        ("GBP", "EUR"), 
        ("JPY", "AUD"),
        ("USD", "GBP"),
    ]
    
    for account_currency in ["USD", "EUR", "GBP", "JPY"]:
        manager = CurrencyDependencyManager(account_currency, apply_oanda_markup=True)
        print(f"\n賬戶貨幣: {account_currency}")
        
        for from_curr in ["EUR", "GBP", "JPY", "AUD"]:
            if from_curr != account_currency:
                rate = manager.convert_to_account_currency(from_curr, price_data)
                print(f"  {from_curr} -> {account_currency}: {rate}")
                assert rate > 0, f"轉換率應該為正數: {from_curr} -> {account_currency}"
    
    print("✓ 多種賬戶貨幣轉換驗證通過")

def test_edge_cases():
    """測試邊緣情況"""
    print("\n=== 測試邊緣情況 ===")
    
    price_data = {
        "EUR_USD": (Decimal('1.0850'), Decimal('1.0852')),
        "GBP_USD": (Decimal('1.2530'), Decimal('1.2532')),
    }
    
    manager = CurrencyDependencyManager("USD", apply_oanda_markup=True)
    
    # 測試同一貨幣轉換
    same_currency_rate = manager.convert_to_account_currency("USD", price_data)
    assert same_currency_rate == Decimal('1.0'), "同一貨幣轉換應該返回1.0"
    print("✓ 同一貨幣轉換測試通過")
    
    # 測試不存在的貨幣對
    unknown_rate = manager.convert_to_account_currency("XYZ", price_data)
    assert unknown_rate == Decimal('1.0'), "不存在的貨幣對應該返回安全值1.0"
    print("✓ 不存在貨幣對測試通過")
    
    # 測試深度限制
    nested_rate = manager.get_specific_rate("ABC", "XYZ", price_data, depth=10)
    assert nested_rate is None, "超過深度限制應該返回None"
    print("✓ 深度限制測試通過")

def main():
    """運行所有驗證測試"""
    print("Oanda規則驗證測試")
    print("=" * 50)
    
    try:
        test_oanda_conversion_fee_formula()
        test_bid_ask_usage()
        test_cross_rate_calculation()
        test_multiple_account_currencies()
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("所有Oanda規則驗證測試通過！✓")
        print("您的實現完全符合Oanda官方交易規則。")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
