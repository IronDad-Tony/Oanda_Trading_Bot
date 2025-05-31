#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oanda 貨幣轉換邏輯測試腳本
測試各種交易品種組合的匯率轉換計算

根據Oanda官方規則：
1. Bid/Ask價格使用規則
2. 0.5% 貨幣轉換費用
3. 交叉匯率計算（三角套利原理）
4. 中介貨幣轉換邏輯
"""

import sys
import os
import logging
from decimal import Decimal
from typing import Dict, Tuple, List
import json
from datetime import datetime

# 添加src路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_manager.currency_manager_fixed import CurrencyDependencyManager

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CurrencyConverterTest")

class OandaCurrencyConverterTester:
    """Oanda貨幣轉換邏輯測試器"""
    
    def __init__(self):
        # 模擬Oanda常見的貨幣對及其價格
        self.mock_prices = {
            # 主要美元對
            "EUR_USD": (Decimal('1.0850'), Decimal('1.0852')),  # bid, ask
            "GBP_USD": (Decimal('1.2530'), Decimal('1.2532')), 
            "USD_JPY": (Decimal('149.80'), Decimal('149.82')),
            "AUD_USD": (Decimal('0.6680'), Decimal('0.6682')),
            "USD_CAD": (Decimal('1.3580'), Decimal('1.3582')),
            "USD_CHF": (Decimal('0.8950'), Decimal('0.8952')),
            "NZD_USD": (Decimal('0.6150'), Decimal('0.6152')),
            
            # 交叉貨幣對
            "EUR_GBP": (Decimal('0.8660'), Decimal('0.8662')),
            "EUR_JPY": (Decimal('162.50'), Decimal('162.55')),
            "GBP_JPY": (Decimal('187.80'), Decimal('187.85')),
            "AUD_JPY": (Decimal('100.10'), Decimal('100.15')),
            "EUR_AUD": (Decimal('1.6240'), Decimal('1.6245')),
            "GBP_AUD": (Decimal('1.8750'), Decimal('1.8755')),
            
            # 商品貨幣對
            "AUD_CAD": (Decimal('0.9070'), Decimal('0.9075')),
            "AUD_CHF": (Decimal('0.5980'), Decimal('0.5985')),
            "CAD_JPY": (Decimal('110.25'), Decimal('110.30')),
            "CHF_JPY": (Decimal('167.40'), Decimal('167.45')),
            
            # 其他交叉對
            "EUR_CAD": (Decimal('1.4730'), Decimal('1.4735')),
            "EUR_CHF": (Decimal('0.9710'), Decimal('0.9715')),
            "GBP_CAD": (Decimal('1.7020'), Decimal('1.7025')),
            "GBP_CHF": (Decimal('1.1220'), Decimal('1.1225')),
            "CAD_CHF": (Decimal('0.6590'), Decimal('0.6595')),
            "NZD_JPY": (Decimal('92.15'), Decimal('92.20')),
        }
        
        # 測試的交易品種組合
        self.trading_symbols = [
            "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD",
            "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD", "GBP_AUD",
            "AUD_CAD", "AUD_CHF", "CAD_JPY", "CHF_JPY", "EUR_CAD", "EUR_CHF",
            "GBP_CAD", "GBP_CHF", "CAD_CHF", "NZD_JPY"
        ]
        
        # 測試的賬戶貨幣
        self.account_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF"]
    
    def extract_currencies_from_symbol(self, symbol: str) -> Tuple[str, str]:
        """從交易品種中提取基礎貨幣和報價貨幣"""
        parts = symbol.split('_')
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            raise ValueError(f"無效的交易品種格式: {symbol}")
    
    def test_single_conversion(self, from_currency: str, account_currency: str, 
                             manager: CurrencyDependencyManager) -> Dict:
        """測試單一貨幣轉換"""
        try:
            # 測試交易匯率（無轉換費用）
            trading_rate = manager.get_trading_rate(from_currency, account_currency, self.mock_prices)
            
            # 測試貨幣轉換匯率（含0.5%費用）
            conversion_rate_credit = manager.convert_to_account_currency(
                from_currency, self.mock_prices, is_credit=True
            )
            conversion_rate_debit = manager.convert_to_account_currency(
                from_currency, self.mock_prices, is_credit=False
            )
            
            return {
                "from_currency": from_currency,
                "to_currency": account_currency,
                "trading_rate": float(trading_rate) if trading_rate else None,
                "conversion_rate_credit": float(conversion_rate_credit),
                "conversion_rate_debit": float(conversion_rate_debit),
                "markup_impact_credit": None,
                "markup_impact_debit": None,
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "from_currency": from_currency,
                "to_currency": account_currency,
                "trading_rate": None,
                "conversion_rate_credit": None,
                "conversion_rate_debit": None,
                "markup_impact_credit": None,
                "markup_impact_debit": None,
                "success": False,
                "error": str(e)
            }
    
    def calculate_markup_impact(self, trading_rate: float, conversion_rate: float) -> float:
        """計算markup對轉換率的影響"""
        if trading_rate and conversion_rate and trading_rate > 0:
            return ((trading_rate - conversion_rate) / trading_rate) * 100
        return 0.0
    
    def test_symbol_combinations(self, account_currency: str) -> List[Dict]:
        """測試所有交易品種與指定賬戶貨幣的組合"""
        manager = CurrencyDependencyManager(account_currency, apply_oanda_markup=True)
        results = []
        
        logger.info(f"測試賬戶貨幣: {account_currency}")
        
        for symbol in self.trading_symbols:
            base_curr, quote_curr = self.extract_currencies_from_symbol(symbol)
            
            # 測試基礎貨幣轉換
            if base_curr != account_currency:
                result = self.test_single_conversion(base_curr, account_currency, manager)
                result["symbol"] = symbol
                result["currency_type"] = "base"
                
                # 計算markup影響
                if result["trading_rate"] and result["conversion_rate_credit"]:
                    result["markup_impact_credit"] = self.calculate_markup_impact(
                        result["trading_rate"], result["conversion_rate_credit"]
                    )
                if result["trading_rate"] and result["conversion_rate_debit"]:
                    result["markup_impact_debit"] = self.calculate_markup_impact(
                        result["trading_rate"], result["conversion_rate_debit"]
                    )
                
                results.append(result)
            
            # 測試報價貨幣轉換
            if quote_curr != account_currency:
                result = self.test_single_conversion(quote_curr, account_currency, manager)
                result["symbol"] = symbol
                result["currency_type"] = "quote"
                
                # 計算markup影響
                if result["trading_rate"] and result["conversion_rate_credit"]:
                    result["markup_impact_credit"] = self.calculate_markup_impact(
                        result["trading_rate"], result["conversion_rate_credit"]
                    )
                if result["trading_rate"] and result["conversion_rate_debit"]:
                    result["markup_impact_debit"] = self.calculate_markup_impact(
                        result["trading_rate"], result["conversion_rate_debit"]
                    )
                
                results.append(result)
        
        return results
    
    def test_cross_rate_calculation(self) -> List[Dict]:
        """測試交叉匯率計算的準確性"""
        manager = CurrencyDependencyManager("USD", apply_oanda_markup=False)
        results = []
        
        # 測試一些已知的交叉匯率
        test_cases = [
            ("EUR", "GBP", "EUR_GBP"),  # 直接可用
            ("EUR", "AUD", "EUR_AUD"),  # 直接可用
            ("GBP", "EUR", "EUR_GBP"),  # 需要反向計算
            ("JPY", "AUD", None),       # 需要通過USD計算
            ("CHF", "CAD", None),       # 需要通過USD計算
        ]
        
        for base, quote, direct_pair in test_cases:
            try:
                calculated_rate = manager.get_specific_rate(base, quote, self.mock_prices)
                
                # 如果有直接貨幣對，比較計算結果
                expected_rate = None
                if direct_pair and direct_pair in self.mock_prices:
                    bid, ask = self.mock_prices[direct_pair]
                    expected_rate = float(ask)
                elif direct_pair:
                    # 檢查反向對
                    reverse_pair = f"{quote}_{base}"
                    if reverse_pair in self.mock_prices:
                        bid, ask = self.mock_prices[reverse_pair]
                        expected_rate = float(1 / bid)
                
                results.append({
                    "base": base,
                    "quote": quote,
                    "calculated_rate": float(calculated_rate) if calculated_rate else None,
                    "expected_rate": expected_rate,
                    "direct_pair_available": direct_pair is not None and direct_pair in self.mock_prices,
                    "calculation_method": self._determine_calculation_method(base, quote, direct_pair),
                    "success": calculated_rate is not None
                })
            except Exception as e:
                results.append({
                    "base": base,
                    "quote": quote,
                    "calculated_rate": None,
                    "expected_rate": None,
                    "direct_pair_available": False,
                    "calculation_method": "error",
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def _determine_calculation_method(self, base: str, quote: str, direct_pair: str) -> str:
        """確定匯率計算方法"""
        if direct_pair and direct_pair in self.mock_prices:
            return "direct"
        reverse_pair = f"{quote}_{base}"
        if reverse_pair in self.mock_prices:
            return "reverse"
        return "cross_rate"
    
    def generate_report(self, results: Dict) -> str:
        """生成測試報告"""
        report = []
        report.append("=" * 80)
        report.append("Oanda 貨幣轉換邏輯測試報告")
        report.append("=" * 80)
        report.append(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 總覽統計
        total_tests = 0
        successful_tests = 0
        
        for account_currency, account_results in results["conversion_tests"].items():
            total_tests += len(account_results)
            successful_tests += sum(1 for r in account_results if r["success"])
        
        report.append(f"總測試數量: {total_tests}")
        report.append(f"成功測試數量: {successful_tests}")
        report.append(f"成功率: {(successful_tests/total_tests)*100:.2f}%")
        report.append("")
        
        # 各賬戶貨幣測試結果
        for account_currency, account_results in results["conversion_tests"].items():
            report.append(f"賬戶貨幣: {account_currency}")
            report.append("-" * 40)
            
            successful = [r for r in account_results if r["success"]]
            failed = [r for r in account_results if not r["success"]]
            
            report.append(f"  成功: {len(successful)}/{len(account_results)}")
            
            if failed:
                report.append(f"  失敗案例:")
                for f in failed[:5]:  # 只顯示前5個失敗案例
                    report.append(f"    {f['from_currency']} -> {f['to_currency']}: {f['error']}")
            
            # 顯示markup影響統計
            markup_impacts = [r["markup_impact_credit"] for r in successful 
                            if r["markup_impact_credit"] is not None]
            if markup_impacts:
                avg_impact = sum(markup_impacts) / len(markup_impacts)
                report.append(f"  平均markup影響: {avg_impact:.4f}%")
            
            report.append("")
        
        # 交叉匯率測試結果
        report.append("交叉匯率計算測試")
        report.append("-" * 40)
        cross_results = results["cross_rate_tests"]
        successful_cross = [r for r in cross_results if r["success"]]
        
        for result in cross_results:
            status = "✓" if result["success"] else "✗"
            method = result.get("calculation_method", "unknown")
            rate = result.get("calculated_rate", "N/A")
            report.append(f"  {status} {result['base']}/{result['quote']}: {rate} ({method})")
        
        report.append("")
        report.append(f"交叉匯率測試成功率: {len(successful_cross)}/{len(cross_results)}")
        
        return "\n".join(report)
    
    def run_comprehensive_test(self) -> Dict:
        """運行全面測試"""
        logger.info("開始Oanda貨幣轉換邏輯測試...")
        
        results = {
            "conversion_tests": {},
            "cross_rate_tests": [],
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_currency_pairs": len(self.mock_prices),
                "total_symbols": len(self.trading_symbols),
                "account_currencies": self.account_currencies
            }
        }
        
        # 測試各賬戶貨幣的轉換
        for account_currency in self.account_currencies:
            results["conversion_tests"][account_currency] = self.test_symbol_combinations(account_currency)
        
        # 測試交叉匯率計算
        results["cross_rate_tests"] = self.test_cross_rate_calculation()
        
        return results
    
    def save_results(self, results: Dict, filename: str = None):
        """保存測試結果到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"currency_conversion_test_results_{timestamp}.json"
        
        # 轉換Decimal為float以便JSON序列化
        def decimal_to_float(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: decimal_to_float(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [decimal_to_float(item) for item in obj]
            return obj
        
        json_results = decimal_to_float(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"測試結果已保存到: {filename}")

def main():
    """主函數"""
    tester = OandaCurrencyConverterTester()
    
    # 運行測試
    results = tester.run_comprehensive_test()
    
    # 生成報告
    report = tester.generate_report(results)
    print(report)
    
    # 保存結果
    tester.save_results(results)
    
    # 保存報告到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"currency_conversion_test_report_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"測試報告已保存到: {report_filename}")

if __name__ == "__main__":
    main()
