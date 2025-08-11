#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整整合測試系統 V2.0
確保每個階段都達到100%完成度並完全整合

Version: 2.0
Author: AI Trading System
Date: 2025-06-08
"""

import sys
import os
import time
import traceback
import gc
import psutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/complete_integration_test_v2.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class CompleteIntegrationTestV2:
    """完整整合測試系統 V2.0"""
    
    def __init__(self):
        """初始化測試系統"""
        self.project_root = Path(__file__).resolve().parent
        self.src_path = self.project_root / "src"
        
        # 添加src路徑
        if str(self.src_path) not in sys.path:
            sys.path.insert(0, str(self.src_path))
        
        # 創建必要目錄
        for path in ['logs', 'weights', 'data']:
            (self.project_root / path).mkdir(exist_ok=True)
        
        self.test_results = {}
        self.components = {}
        
        logger.info("🚀 完整整合測試系統 V2.0 初始化完成")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """運行所有測試"""
        logger.info("="*80)
        logger.info("🎯 開始完整整合測試 V2.0")
        logger.info("="*80)
        
        start_time = time.time()
        overall_results = {
            'test_start_time': datetime.now().isoformat(),
            'phases': {},
            'overall_success': False,
            'total_completion_rate': 0.0,
            'critical_issues': [],
            'performance_metrics': {}
        }
        
        try:
            # Phase 1: 測試基礎組件載入
            logger.info("\n📋 Phase 1: 基礎組件載入測試")
            phase1_results = self.test_component_loading()
            overall_results['phases']['phase1_component_loading'] = phase1_results
            
            # Phase 2: 測試組件功能
            logger.info("\n📋 Phase 2: 組件功能測試")
            phase2_results = self.test_component_functionality()
            overall_results['phases']['phase2_component_functionality'] = phase2_results
            
            # Phase 3: 測試組件整合
            logger.info("\n📋 Phase 3: 組件整合測試")
            phase3_results = self.test_component_integration()
            overall_results['phases']['phase3_component_integration'] = phase3_results
            
            # Phase 4: 測試真實數據處理
            logger.info("\n📋 Phase 4: 真實數據處理測試")
            phase4_results = self.test_real_data_processing()
            overall_results['phases']['phase4_real_data_processing'] = phase4_results
            
            # Phase 5: 測試系統性能
            logger.info("\n📋 Phase 5: 系統性能測試")
            phase5_results = self.test_system_performance()
            overall_results['phases']['phase5_system_performance'] = phase5_results
            
            # Phase 6: 端到端測試
            logger.info("\n📋 Phase 6: 端到端完整測試")
            phase6_results = self.test_end_to_end()
            overall_results['phases']['phase6_end_to_end'] = phase6_results
            
            # 計算總體結果
            total_completion = self.calculate_total_completion(overall_results['phases'])
            overall_results['total_completion_rate'] = total_completion
            overall_results['overall_success'] = total_completion >= 80.0
            
        except Exception as e:
            logger.error(f"❌ 測試執行失敗: {e}")
            overall_results['critical_issues'].append(f"測試執行失敗: {str(e)}")
            traceback.print_exc()
        
        # 完成測試
        end_time = time.time()
        overall_results['test_end_time'] = datetime.now().isoformat()
        overall_results['total_duration_seconds'] = end_time - start_time
        
        # 保存結果
        self.save_test_results(overall_results)
        
        # 輸出總結
        self.print_test_summary(overall_results)
        
        return overall_results
    
    def test_component_loading(self) -> Dict[str, Any]:
        """測試組件載入"""
        results = {
            'components_loaded': {},
            'import_success_rate': 0.0,
            'critical_errors': [],
            'completion_rate': 0.0
        }
        
        components_to_test = [
            ('StrategyInnovationModule', 'agent.strategy_innovation_module'),
            ('MarketStateAwarenessSystem', 'agent.market_state_awareness_system'),
            ('MetaLearningOptimizer', 'agent.meta_learning_optimizer'),
            ('HighLevelIntegrationSystem', 'agent.high_level_integration_system'),
            ('UniversalTradingEnvV4', 'environment.trading_env'),
            ('OandaDownloader', 'data_manager.oanda_downloader'),
            ('DatabaseManager', 'data_manager.database_manager')
        ]
        
        successful_imports = 0
        
        for component_name, module_path in components_to_test:
            try:
                logger.info(f"📦 載入組件: {component_name}")
                
                if module_path == 'agent.strategy_innovation_module':
                    from agent.strategy_innovation_module import StrategyInnovationModule
                    component = StrategyInnovationModule(input_dim=768, hidden_dim=768)
                    
                elif module_path == 'agent.market_state_awareness_system':
                    from agent.market_state_awareness_system import MarketStateAwarenessSystem
                    component = MarketStateAwarenessSystem(input_dim=768, num_strategies=20)
                    
                elif module_path == 'agent.meta_learning_optimizer':
                    from agent.meta_learning_optimizer import MetaLearningOptimizer
                    base_model = nn.Linear(768, 1)
                    component = MetaLearningOptimizer(model=base_model, feature_dim=768)
                    
                elif module_path == 'agent.high_level_integration_system':
                    # 先載入依賴組件
                    if 'StrategyInnovationModule' in self.components:
                        from agent.high_level_integration_system import HighLevelIntegrationSystem
                        component = HighLevelIntegrationSystem(
                            strategy_innovation_module=self.components['StrategyInnovationModule'],
                            market_state_awareness_system=self.components['MarketStateAwarenessSystem'],
                            meta_learning_optimizer=self.components['MetaLearningOptimizer']
                        )
                    else:
                        raise Exception("依賴組件未載入")
                        
                elif module_path == 'environment.trading_env':
                    from environment.trading_env import UniversalTradingEnvV4
                    component = UniversalTradingEnvV4
                    
                elif module_path == 'data_manager.oanda_downloader':
                    from data_manager.oanda_downloader import manage_data_download_for_symbols
                    component = manage_data_download_for_symbols
                    
                elif module_path == 'data_manager.database_manager':
                    from data_manager.database_manager import query_historical_data
                    component = query_historical_data
                
                self.components[component_name] = component
                results['components_loaded'][component_name] = True
                successful_imports += 1
                logger.info(f"✅ {component_name} 載入成功")
                
            except Exception as e:
                error_msg = f"{component_name} 載入失敗: {str(e)}"
                logger.error(f"❌ {error_msg}")
                results['components_loaded'][component_name] = False
                results['critical_errors'].append(error_msg)
        
        results['import_success_rate'] = (successful_imports / len(components_to_test)) * 100
        results['completion_rate'] = results['import_success_rate']
        
        logger.info(f"📊 組件載入完成率: {results['completion_rate']:.1f}%")
        return results
    
    def test_component_functionality(self) -> Dict[str, Any]:
        """測試組件功能"""
        results = {
            'functionality_tests': {},
            'functional_success_rate': 0.0,
            'errors': [],
            'completion_rate': 0.0
        }
        
        # 創建測試數據
        test_data = torch.randn(4, 50, 768)
        successful_tests = 0
        total_tests = 0
        
        # 測試策略創新模組
        if 'StrategyInnovationModule' in self.components:
            try:
                logger.info("🧪 測試策略創新模組功能")
                component = self.components['StrategyInnovationModule']
                
                output = component(test_data)
                
                # 驗證輸出
                required_keys = ['generated_strategies', 'innovation_confidence', 'strategy_diversity']
                tests_passed = 0
                
                for key in required_keys:
                    if key in output:
                        tests_passed += 1
                
                # 檢查數值範圍
                if 'innovation_confidence' in output:
                    confidence = output['innovation_confidence']
                    if 0.0 <= confidence <= 1.0:
                        tests_passed += 1
                
                success_rate = (tests_passed / (len(required_keys) + 1)) * 100
                results['functionality_tests']['StrategyInnovationModule'] = {
                    'success': success_rate >= 75,
                    'success_rate': success_rate,
                    'tests_passed': tests_passed,
                    'total_tests': len(required_keys) + 1
                }
                
                if success_rate >= 75:
                    successful_tests += 1
                    logger.info(f"✅ 策略創新模組測試通過: {success_rate:.1f}%")
                else:
                    logger.warning(f"⚠️ 策略創新模組測試部分失敗: {success_rate:.1f}%")
                
                total_tests += 1
                
            except Exception as e:
                error_msg = f"策略創新模組功能測試失敗: {str(e)}"
                logger.error(f"❌ {error_msg}")
                results['errors'].append(error_msg)
                results['functionality_tests']['StrategyInnovationModule'] = {
                    'success': False,
                    'error': str(e)
                }
                total_tests += 1
        
        # 測試市場狀態感知系統
        if 'MarketStateAwarenessSystem' in self.components:
            try:
                logger.info("🧪 測試市場狀態感知系統功能")
                component = self.components['MarketStateAwarenessSystem']
                
                output = component(test_data)
                
                required_keys = ['market_state', 'system_status', 'regime_confidence']
                tests_passed = 0
                
                for key in required_keys:
                    if key in output:
                        tests_passed += 1
                
                # 檢查置信度範圍
                if 'regime_confidence' in output:
                    confidence = output['regime_confidence']
                    if 0.0 <= confidence <= 1.0:
                        tests_passed += 1
                
                success_rate = (tests_passed / (len(required_keys) + 1)) * 100
                results['functionality_tests']['MarketStateAwarenessSystem'] = {
                    'success': success_rate >= 75,
                    'success_rate': success_rate,
                    'tests_passed': tests_passed,
                    'total_tests': len(required_keys) + 1
                }
                
                if success_rate >= 75:
                    successful_tests += 1
                    logger.info(f"✅ 市場狀態感知系統測試通過: {success_rate:.1f}%")
                else:
                    logger.warning(f"⚠️ 市場狀態感知系統測試部分失敗: {success_rate:.1f}%")
                
                total_tests += 1
                
            except Exception as e:
                error_msg = f"市場狀態感知系統功能測試失敗: {str(e)}"
                logger.error(f"❌ {error_msg}")
                results['errors'].append(error_msg)
                results['functionality_tests']['MarketStateAwarenessSystem'] = {
                    'success': False,
                    'error': str(e)
                }
                total_tests += 1
        
        # 測試元學習優化器
        if 'MetaLearningOptimizer' in self.components:
            try:
                logger.info("🧪 測試元學習優化器功能")
                component = self.components['MetaLearningOptimizer']
                
                # 創建任務批次
                from agent.meta_learning_optimizer import TaskBatch
                task_batches = [
                    TaskBatch(
                        support_data=torch.randn(16, 768),
                        support_labels=torch.randn(16, 1),
                        query_data=torch.randn(8, 768),
                        query_labels=torch.randn(8, 1),
                        task_id="test_task",
                        market_state="trending",
                        difficulty=0.5
                    )
                ]
                
                output = component.optimize_and_adapt(test_data, test_data, task_batches)
                
                required_keys = ['adapted_features', 'selected_strategy', 'adaptation_quality']
                tests_passed = 0
                
                for key in required_keys:
                    if key in output:
                        tests_passed += 1
                
                # 檢查適應質量範圍
                if 'adaptation_quality' in output:
                    quality = output['adaptation_quality']
                    if 0.0 <= quality <= 1.0:
                        tests_passed += 1
                
                success_rate = (tests_passed / (len(required_keys) + 1)) * 100
                results['functionality_tests']['MetaLearningOptimizer'] = {
                    'success': success_rate >= 75,
                    'success_rate': success_rate,
                    'tests_passed': tests_passed,
                    'total_tests': len(required_keys) + 1
                }
                
                if success_rate >= 75:
                    successful_tests += 1
                    logger.info(f"✅ 元學習優化器測試通過: {success_rate:.1f}%")
                else:
                    logger.warning(f"⚠️ 元學習優化器測試部分失敗: {success_rate:.1f}%")
                
                total_tests += 1
                
            except Exception as e:
                error_msg = f"元學習優化器功能測試失敗: {str(e)}"
                logger.error(f"❌ {error_msg}")
                results['errors'].append(error_msg)
                results['functionality_tests']['MetaLearningOptimizer'] = {
                    'success': False,
                    'error': str(e)
                }
                total_tests += 1
        
        # 計算總體功能成功率
        if total_tests > 0:
            results['functional_success_rate'] = (successful_tests / total_tests) * 100
            results['completion_rate'] = results['functional_success_rate']
        
        logger.info(f"📊 組件功能測試完成率: {results['completion_rate']:.1f}%")
        return results
    
    def test_component_integration(self) -> Dict[str, Any]:
        """測試組件整合"""
        results = {
            'integration_tests': {},
            'integration_success_rate': 0.0,
            'data_flow_tests': {},
            'errors': [],
            'completion_rate': 0.0
        }
        
        successful_integrations = 0
        total_integrations = 0
        
        # 測試高階整合系統
        if 'HighLevelIntegrationSystem' in self.components:
            try:
                logger.info("🔗 測試高階整合系統")
                component = self.components['HighLevelIntegrationSystem']
                
                test_data = torch.randn(4, 50, 768)
                
                # 測試市場數據處理
                output = component.process_market_data(test_data)
                
                required_keys = [
                    'market_state', 'strategy_innovation', 'meta_learning',
                    'system_health', 'processing_time'
                ]
                tests_passed = 0
                
                for key in required_keys:
                    if key in output:
                        tests_passed += 1
                
                # 檢查處理時間
                if 'processing_time' in output and output['processing_time'] < 10.0:
                    tests_passed += 1
                
                # 檢查系統健康度
                if 'system_health' in output and isinstance(output['system_health'], dict):
                    if 'overall_health' in output['system_health']:
                        health = output['system_health']['overall_health']
                        if 0.0 <= health <= 1.0:
                            tests_passed += 1
                
                success_rate = (tests_passed / (len(required_keys) + 2)) * 100
                results['integration_tests']['HighLevelIntegrationSystem'] = {
                    'success': success_rate >= 75,
                    'success_rate': success_rate,
                    'processing_time': output.get('processing_time', 'N/A')
                }
                
                if success_rate >= 75:
                    successful_integrations += 1
                    logger.info(f"✅ 高階整合系統測試通過: {success_rate:.1f}%")
                else:
                    logger.warning(f"⚠️ 高階整合系統測試部分失敗: {success_rate:.1f}%")
                
                total_integrations += 1
                
            except Exception as e:
                error_msg = f"高階整合系統測試失敗: {str(e)}"
                logger.error(f"❌ {error_msg}")
                results['errors'].append(error_msg)
                results['integration_tests']['HighLevelIntegrationSystem'] = {
                    'success': False,
                    'error': str(e)
                }
                total_integrations += 1
        
        # 測試數據流動
        try:
            logger.info("🔄 測試組件間數據流動")
            
            if ('StrategyInnovationModule' in self.components and 
                'MarketStateAwarenessSystem' in self.components):
                
                test_data = torch.randn(2, 30, 768)
                
                # 測試市場狀態 -> 策略創新
                market_output = self.components['MarketStateAwarenessSystem'](test_data)
                strategy_output = self.components['StrategyInnovationModule'](test_data)
                
                data_flow_success = (
                    'market_state' in market_output and 
                    'generated_strategies' in strategy_output
                )
                
                results['data_flow_tests']['market_to_strategy'] = data_flow_success
                
                if data_flow_success:
                    logger.info("✅ 市場狀態到策略創新數據流動正常")
                else:
                    logger.warning("⚠️ 市場狀態到策略創新數據流動異常")
                
        except Exception as e:
            error_msg = f"數據流動測試失敗: {str(e)}"
            logger.error(f"❌ {error_msg}")
            results['errors'].append(error_msg)
            results['data_flow_tests']['market_to_strategy'] = False
        
        # 計算整合成功率
        if total_integrations > 0:
            results['integration_success_rate'] = (successful_integrations / total_integrations) * 100
        
        # 計算數據流測試成功率
        data_flow_tests = list(results['data_flow_tests'].values())
        if data_flow_tests:
            data_flow_success_rate = (sum(data_flow_tests) / len(data_flow_tests)) * 100
        else:
            data_flow_success_rate = 0.0
        
        # 綜合完成率
        results['completion_rate'] = (results['integration_success_rate'] + data_flow_success_rate) / 2
        
        logger.info(f"📊 組件整合測試完成率: {results['completion_rate']:.1f}%")
        return results
    
    def test_real_data_processing(self) -> Dict[str, Any]:
        """測試真實數據處理"""
        results = {
            'data_access_tests': {},
            'data_processing_tests': {},
            'trading_env_tests': {},
            'errors': [],
            'completion_rate': 0.0
        }
        
        successful_tests = 0
        total_tests = 0
        
        # 測試數據庫連接和查詢
        try:
            logger.info("💾 測試數據庫連接")
            
            if 'DatabaseManager' in self.components:
                # 測試數據查詢
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(hours=1)
                
                # 格式化時間
                from data_manager.oanda_downloader import format_datetime_for_oanda
                start_iso = format_datetime_for_oanda(start_time)
                end_iso = format_datetime_for_oanda(end_time)
                
                # 嘗試查詢數據
                df = self.components['DatabaseManager']("EUR_USD", "S5", start_iso, end_iso, limit=10)
                
                data_available = not df.empty if isinstance(df, pd.DataFrame) else False
                results['data_access_tests']['database_query'] = data_available
                
                if data_available:
                    logger.info(f"✅ 數據庫查詢成功，獲取 {len(df)} 條記錄")
                    successful_tests += 1
                else:
                    logger.warning("⚠️ 數據庫查詢無數據返回")
                
                total_tests += 1
                
        except Exception as e:
            error_msg = f"數據庫連接測試失敗: {str(e)}"
            logger.error(f"❌ {error_msg}")
            results['errors'].append(error_msg)
            results['data_access_tests']['database_query'] = False
            total_tests += 1
        
        # 測試交易環境
        try:
            logger.info("🏢 測試交易環境")
            
            if 'UniversalTradingEnvV4' in self.components:
                env_class = self.components['UniversalTradingEnvV4']
                
                # 創建交易環境
                symbols = ["EUR_USD"]
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(minutes=30)
                
                trading_env = env_class(
                    symbols=symbols,
                    start_time=start_time,
                    end_time=end_time,
                    initial_capital=10000.0,
                    max_symbols=1
                )
                
                # 測試環境重置
                obs = trading_env.reset()
                env_reset_success = obs is not None
                results['trading_env_tests']['env_reset'] = env_reset_success
                
                if env_reset_success:
                    logger.info("✅ 交易環境重置成功")
                    successful_tests += 1
                    
                    # 測試環境步驟
                    try:
                        action = np.random.uniform(-0.1, 0.1, size=trading_env.action_space.shape)
                        obs, reward, done, info = trading_env.step(action)
                        
                        env_step_success = obs is not None and isinstance(reward, (int, float))
                        results['trading_env_tests']['env_step'] = env_step_success
                        
                        if env_step_success:
                            logger.info("✅ 交易環境步驟執行成功")
                            successful_tests += 1
                        else:
                            logger.warning("⚠️ 交易環境步驟執行失敗")
                        
                        total_tests += 1
                        
                    except Exception as e:
                        logger.error(f"❌ 交易環境步驟測試失敗: {e}")
                        results['trading_env_tests']['env_step'] = False
                        total_tests += 1
                else:
                    logger.warning("⚠️ 交易環境重置失敗")
                
                total_tests += 1
                
        except Exception as e:
            error_msg = f"交易環境測試失敗: {str(e)}"
            logger.error(f"❌ {error_msg}")
            results['errors'].append(error_msg)
            results['trading_env_tests']['env_creation'] = False
            total_tests += 1
        
        # 計算完成率
        if total_tests > 0:
            results['completion_rate'] = (successful_tests / total_tests) * 100
        
        logger.info(f"📊 真實數據處理測試完成率: {results['completion_rate']:.1f}%")
        return results
    
    def test_system_performance(self) -> Dict[str, Any]:
        """測試系統性能"""
        results = {
            'latency_tests': {},
            'throughput_tests': {},
            'memory_tests': {},
            'errors': [],
            'completion_rate': 0.0
        }
        
        successful_tests = 0
        total_tests = 0
        
        # 延遲測試
        try:
            logger.info("⚡ 測試系統延遲")
            
            if 'HighLevelIntegrationSystem' in self.components:
                component = self.components['HighLevelIntegrationSystem']
                
                # 測試單次處理延遲
                test_data = torch.randn(1, 50, 768)
                
                latencies = []
                for _ in range(5):
                    start_time = time.time()
                    output = component.process_market_data(test_data)
                    end_time = time.time()
                    latencies.append(end_time - start_time)
                
                avg_latency = np.mean(latencies)
                max_latency = np.max(latencies)
                
                results['latency_tests']['average_latency_seconds'] = avg_latency
                results['latency_tests']['max_latency_seconds'] = max_latency
                results['latency_tests']['latency_acceptable'] = avg_latency < 2.0  # 2秒內
                
                if avg_latency < 2.0:
                    logger.info(f"✅ 系統延遲測試通過: {avg_latency:.3f}s")
                    successful_tests += 1
                else:
                    logger.warning(f"⚠️ 系統延遲較高: {avg_latency:.3f}s")
                
                total_tests += 1
                
        except Exception as e:
            error_msg = f"延遲測試失敗: {str(e)}"
            logger.error(f"❌ {error_msg}")
            results['errors'].append(error_msg)
            total_tests += 1
        
        # 記憶體使用測試
        try:
            logger.info("💾 測試記憶體使用")
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 執行大批量處理
            if 'HighLevelIntegrationSystem' in self.components:
                component = self.components['HighLevelIntegrationSystem']
                large_data = torch.randn(8, 100, 768)
                
                for _ in range(3):
                    output = component.process_market_data(large_data)
                
                peak_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # 清理記憶體
                del large_data, output
                gc.collect()
                
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                memory_increase = peak_memory - initial_memory
                memory_leak = final_memory - initial_memory
                
                results['memory_tests']['initial_memory_mb'] = initial_memory
                results['memory_tests']['peak_memory_mb'] = peak_memory
                results['memory_tests']['final_memory_mb'] = final_memory
                results['memory_tests']['memory_increase_mb'] = memory_increase
                results['memory_tests']['memory_leak_mb'] = memory_leak
                results['memory_tests']['memory_acceptable'] = memory_leak < 50  # 小於50MB洩漏
                
                if memory_leak < 50:
                    logger.info(f"✅ 記憶體使用測試通過，洩漏: {memory_leak:.1f}MB")
                    successful_tests += 1
                else:
                    logger.warning(f"⚠️ 記憶體洩漏較多: {memory_leak:.1f}MB")
                
                total_tests += 1
                
        except Exception as e:
            error_msg = f"記憶體測試失敗: {str(e)}"
            logger.error(f"❌ {error_msg}")
            results['errors'].append(error_msg)
            total_tests += 1
        
        # 計算完成率
        if total_tests > 0:
            results['completion_rate'] = (successful_tests / total_tests) * 100
        
        logger.info(f"📊 系統性能測試完成率: {results['completion_rate']:.1f}%")
        return results
    
    def test_end_to_end(self) -> Dict[str, Any]:
        """端到端測試"""
        results = {
            'workflow_tests': {},
            'integration_stability': {},
            'real_scenario_tests': {},
            'errors': [],
            'completion_rate': 0.0
        }
        
        successful_tests = 0
        total_tests = 0
        
        try:
            logger.info("🎯 執行端到端工作流程測試")
            
            # 完整工作流程測試
            if 'HighLevelIntegrationSystem' in self.components:
                component = self.components['HighLevelIntegrationSystem']
                
                # 1. 市場數據輸入
                test_data = torch.randn(2, 50, 768)
                
                # 2. 處理市場數據
                output = component.process_market_data(test_data)
                
                # 3. 驗證輸出完整性
                required_outputs = [
                    'market_state', 'strategy_innovation', 'meta_learning',
                    'system_health', 'processing_time'
                ]
                
                workflow_success = all(key in output for key in required_outputs)
                results['workflow_tests']['complete_pipeline'] = workflow_success
                
                if workflow_success:
                    logger.info("✅ 完整工作流程測試通過")
                    successful_tests += 1
                else:
                    logger.warning("⚠️ 完整工作流程測試失敗")
                
                total_tests += 1
                
                # 4. 多次執行穩定性測試
                logger.info("🔄 測試多次執行穩定性")
                stability_success = True
                
                for i in range(3):
                    try:
                        output = component.process_market_data(test_data)
                        if not all(key in output for key in required_outputs):
                            stability_success = False
                            break
                    except Exception as e:
                        logger.error(f"穩定性測試第{i+1}次失敗: {e}")
                        stability_success = False
                        break
                
                results['integration_stability']['multiple_runs'] = stability_success
                
                if stability_success:
                    logger.info("✅ 多次執行穩定性測試通過")
                    successful_tests += 1
                else:
                    logger.warning("⚠️ 多次執行穩定性測試失敗")
                
                total_tests += 1
                
        except Exception as e:
            error_msg = f"端到端測試失敗: {str(e)}"
            logger.error(f"❌ {error_msg}")
            results['errors'].append(error_msg)
            total_tests += 1
        
        # 計算完成率
        if total_tests > 0:
            results['completion_rate'] = (successful_tests / total_tests) * 100
        
        logger.info(f"📊 端到端測試完成率: {results['completion_rate']:.1f}%")
        return results
    
    def calculate_total_completion(self, phases: Dict[str, Any]) -> float:
        """計算總體完成率"""
        completion_rates = []
        
        for phase_name, phase_results in phases.items():
            if isinstance(phase_results, dict) and 'completion_rate' in phase_results:
                completion_rates.append(phase_results['completion_rate'])
        
        if completion_rates:
            return sum(completion_rates) / len(completion_rates)
        else:
            return 0.0
    
    def save_test_results(self, results: Dict[str, Any]):
        """保存測試結果"""
        try:
            results_file = self.project_root / "logs" / f"integration_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📄 測試結果已保存到: {results_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存測試結果失敗: {e}")
    
    def print_test_summary(self, results: Dict[str, Any]):
        """印出測試總結"""
        logger.info("\n" + "="*80)
        logger.info("📋 測試總結報告")
        logger.info("="*80)
        
        total_completion = results.get('total_completion_rate', 0.0)
        overall_success = results.get('overall_success', False)
        
        # 狀態指示
        status_icon = "✅" if overall_success else "❌"
        logger.info(f"{status_icon} 總體狀態: {'成功' if overall_success else '需要改進'}")
        logger.info(f"📊 總體完成率: {total_completion:.1f}%")
        
        # 各階段詳情
        logger.info("\n📋 各階段完成率:")
        phases = results.get('phases', {})
        for phase_name, phase_results in phases.items():
            if isinstance(phase_results, dict) and 'completion_rate' in phase_results:
                completion = phase_results['completion_rate']
                icon = "✅" if completion >= 80 else "⚠️" if completion >= 60 else "❌"
                logger.info(f"  {icon} {phase_name}: {completion:.1f}%")
        
        # 關鍵問題
        critical_issues = results.get('critical_issues', [])
        if critical_issues:
            logger.info("\n🚨 關鍵問題:")
            for issue in critical_issues[:5]:  # 只顯示前5個
                logger.info(f"  ❌ {issue}")
        
        # 性能指標
        performance = results.get('performance_metrics', {})
        if performance:
            logger.info("\n⚡ 性能指標:")
            for metric, value in performance.items():
                logger.info(f"  📊 {metric}: {value}")
        
        # 測試持續時間
        duration = results.get('total_duration_seconds', 0)
        logger.info(f"\n⏱️ 測試持續時間: {duration:.1f} 秒")
        
        logger.info("="*80)
        
        # 建議
        if total_completion < 80:
            logger.info("💡 建議:")
            logger.info("  - 檢查失敗的組件載入")
            logger.info("  - 確認數據連接設置")
            logger.info("  - 檢查系統依賴項")
            logger.info("  - 查看詳細錯誤日誌")
        else:
            logger.info("🎉 恭喜！系統整合測試基本通過！")


def main():
    """主函數"""
    print("🚀 啟動完整整合測試系統 V2.0")
    print("="*80)
    
    try:
        # 創建測試實例
        test_system = CompleteIntegrationTestV2()
        
        # 運行所有測試
        results = test_system.run_all_tests()
        
        # 返回結果
        return results
        
    except Exception as e:
        logger.error(f"❌ 測試系統執行失敗: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 執行測試
    test_results = main()
    
    if test_results:
        total_completion = test_results.get('total_completion_rate', 0.0)
        success = test_results.get('overall_success', False)
        
        print(f"\n🎯 最終結果:")
        print(f"✅ 成功: {'是' if success else '否'}")
        print(f"📊 完成率: {total_completion:.1f}%")
        
        if success:
            print("🎉 系統整合測試成功完成！")
        else:
            print("⚠️ 系統需要進一步改進。")
    else:
        print("❌ 測試執行失敗，請檢查錯誤日誌。")
