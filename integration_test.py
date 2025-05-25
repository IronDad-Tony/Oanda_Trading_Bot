#!/usr/bin/env python3
"""
OANDA AI Trading Bot - 整合測試腳本
測試所有修復的相容性和系統整體穩定性
"""

import sys
import os
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import gc

# 確保能找到src模組
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('integration_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class IntegrationTestSuite:
    """整合測試套件"""
    
    def __init__(self):
        self.test_results = {}
        self.failed_tests = []
        self.passed_tests = []
        self.start_time = datetime.now()
        
        # 測試配置
        self.test_symbols = ["EUR_USD", "USD_JPY"]
        self.test_timesteps = 1000  # 短期測試
        self.test_start_time = datetime.now(timezone.utc) - timedelta(days=7)
        self.test_end_time = datetime.now(timezone.utc) - timedelta(days=1)
        
        logger.info("=" * 60)
        logger.info("🧪 OANDA AI Trading Bot 整合測試開始")
        logger.info("=" * 60)
    
    def run_test(self, test_name: str, test_func) -> bool:
        """執行單個測試"""
        logger.info(f"🔍 執行測試: {test_name}")
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            if result:
                self.passed_tests.append(test_name)
                logger.info(f"✅ {test_name} - 通過 ({duration:.2f}秒)")
            else:
                self.failed_tests.append(test_name)
                logger.error(f"❌ {test_name} - 失敗 ({duration:.2f}秒)")
            
            self.test_results[test_name] = {
                'passed': result,
                'duration': duration,
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time if 'start_time' in locals() else 0
            self.failed_tests.append(test_name)
            logger.error(f"❌ {test_name} - 異常: {e}")
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            
            self.test_results[test_name] = {
                'passed': False,
                'duration': duration,
                'error': str(e),
                'timestamp': datetime.now()
            }
            
            return False
    
    def test_imports(self) -> bool:
        """測試所有模組導入"""
        logger.info("測試模組導入...")
        
        try:
            # 測試核心模組
            from src.common.config import DEVICE, USE_AMP, LOGS_DIR
            from src.common.logger_setup import logger as common_logger
            from src.common.shared_data_manager import get_shared_data_manager
            
            # 測試數據管理模組
            from src.data_manager.currency_manager import CurrencyDependencyManager
            from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset
            from src.data_manager.database_manager import DatabaseManager
            
            # 測試環境模組
            from src.environment.trading_env import UniversalTradingEnvV4
            
            # 測試代理模組
            from src.agent.sac_agent_wrapper import SACAgentWrapper
            
            # 測試訓練器模組
            from src.trainer.enhanced_trainer_complete import EnhancedUniversalTrainer
            
            logger.info("✅ 所有核心模組導入成功")
            return True
            
        except ImportError as e:
            logger.error(f"❌ 模組導入失敗: {e}")
            return False
    
    def test_gpu_setup(self) -> bool:
        """測試GPU設置和優化"""
        logger.info("測試GPU設置...")
        
        try:
            # 檢查CUDA可用性
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA 可用: {cuda_available}")
            
            if cuda_available:
                # 檢查GPU信息
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
                
                logger.info(f"GPU數量: {gpu_count}")
                logger.info(f"當前GPU: {current_device} - {gpu_name}")
                logger.info(f"GPU記憶體: {gpu_memory:.1f}GB")
                
                # 測試GPU記憶體清理
                torch.cuda.empty_cache()
                gc.collect()
                
                # 測試基本GPU操作
                test_tensor = torch.randn(100, 100).cuda()
                result = torch.matmul(test_tensor, test_tensor.T)
                logger.info(f"GPU計算測試: {result.shape}")
                
                # 清理測試張量
                del test_tensor, result
                torch.cuda.empty_cache()
                
            logger.info("✅ GPU設置測試完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU設置測試失敗: {e}")
            return False
    
    def test_shared_data_manager(self) -> bool:
        """測試共享數據管理器"""
        logger.info("測試共享數據管理器...")
        
        try:
            from src.common.shared_data_manager import get_shared_data_manager
            
            # 獲取共享數據管理器實例
            manager = get_shared_data_manager()
            
            # 測試狀態更新
            manager.update_training_status('running', 50)
            status = manager.get_current_status()
            assert status['status'] == 'running'
            assert status['progress'] == 50
            
            # 測試停止請求
            manager.request_stop()
            assert manager.is_stop_requested() == True
            manager.reset_stop_flag()
            assert manager.is_stop_requested() == False
            
            # 測試指標添加
            manager.add_training_metric(
                step=100,
                reward=1.5,
                portfolio_value=105000,
                actor_loss=0.1,
                critic_loss=0.2
            )
            
            metrics = manager.get_latest_metrics(1)
            assert len(metrics) == 1
            assert metrics[0]['step'] == 100
            assert metrics[0]['reward'] == 1.5
            
            # 測試交易記錄
            manager.add_trade_record(
                symbol='EUR_USD',
                action='buy',
                price=1.1000,
                quantity=10000,
                profit_loss=50.0
            )
            
            trades = manager.get_latest_trades(1)
            assert len(trades) == 1
            assert trades[0]['symbol'] == 'EUR_USD'
            
            logger.info("✅ 共享數據管理器測試通過")
            return True
            
        except Exception as e:
            logger.error(f"❌ 共享數據管理器測試失敗: {e}")
            return False
    
    def test_file_management(self) -> bool:
        """測試檔案管理和路徑配置"""
        logger.info("測試檔案管理...")
        
        try:
            # 檢查重要目錄
            required_dirs = ['src', 'logs', 'data']
            for dir_name in required_dirs:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"創建目錄: {dir_path}")
                assert dir_path.exists(), f"目錄不存在: {dir_name}"
            
            # 檢查weights目錄
            weights_dir = Path('weights')
            if not weights_dir.exists():
                weights_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"創建weights目錄: {weights_dir}")
            
            # 測試日誌檔案創建
            test_log_file = Path('logs/integration_test.log')
            test_log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(test_log_file, 'w', encoding='utf-8') as f:
                f.write(f"Integration test log - {datetime.now()}\n")
            
            assert test_log_file.exists(), "測試日誌檔案創建失敗"
            
            # 清理測試檔案
            test_log_file.unlink()
            
            logger.info("✅ 檔案管理測試通過")
            return True
            
        except Exception as e:
            logger.error(f"❌ 檔案管理測試失敗: {e}")
            return False
    
    def test_mmap_cleanup(self) -> bool:
        """測試mmap檔案清理機制"""
        logger.info("測試mmap檔案清理...")
        
        try:
            from src.data_manager.mmap_dataset import UniversalMemoryMappedDataset
            
            # 創建測試數據集（小規模）
            dataset = UniversalMemoryMappedDataset(
                symbols=self.test_symbols,
                start_time=self.test_start_time,
                end_time=self.test_end_time,
                granularity="M1",  # 使用分鐘數據減少數據量
                timesteps=50
            )
            
            # 檢查數據集是否有效
            if dataset.is_valid():
                logger.info(f"測試數據集創建成功，樣本數: {len(dataset)}")
                
                # 測試數據訪問
                if len(dataset) > 0:
                    sample = dataset[0]
                    logger.info(f"樣本形狀: {sample[0].shape if hasattr(sample[0], 'shape') else 'N/A'}")
                
                # 清理數據集
                dataset.cleanup()
                logger.info("數據集清理完成")
            else:
                logger.warning("測試數據集無效，可能是數據不足")
            
            logger.info("✅ mmap檔案清理測試通過")
            return True
            
        except Exception as e:
            logger.error(f"❌ mmap檔案清理測試失敗: {e}")
            return False
    
    def test_trainer_initialization(self) -> bool:
        """測試訓練器初始化"""
        logger.info("測試訓練器初始化...")
        
        try:
            from src.trainer.enhanced_trainer_complete import EnhancedUniversalTrainer
            
            # 創建訓練器實例
            trainer = EnhancedUniversalTrainer(
                trading_symbols=self.test_symbols,
                start_time=self.test_start_time,
                end_time=self.test_end_time,
                granularity="M1",
                total_timesteps=self.test_timesteps,
                save_freq=500,
                eval_freq=500,
                model_name_prefix="integration_test"
            )
            
            logger.info(f"訓練器創建成功")
            logger.info(f"交易品種: {trainer.trading_symbols}")
            logger.info(f"模型識別碼: {trainer.model_identifier}")
            
            # 測試清理
            trainer.cleanup()
            
            logger.info("✅ 訓練器初始化測試通過")
            return True
            
        except Exception as e:
            logger.error(f"❌ 訓練器初始化測試失敗: {e}")
            return False
    
    def test_streamlit_compatibility(self) -> bool:
        """測試Streamlit相容性"""
        logger.info("測試Streamlit相容性...")
        
        try:
            # 檢查Streamlit應用檔案
            streamlit_files = [
                'streamlit_app_complete.py',
                'streamlit_app.py'
            ]
            
            found_files = []
            for file_name in streamlit_files:
                file_path = Path(file_name)
                if file_path.exists():
                    found_files.append(file_name)
                    logger.info(f"找到Streamlit檔案: {file_name}")
            
            assert len(found_files) > 0, "未找到Streamlit應用檔案"
            
            # 測試導入Streamlit相關模組
            try:
                import streamlit as st
                import plotly.graph_objects as go
                import plotly.express as px
                logger.info("Streamlit相關模組導入成功")
            except ImportError as e:
                logger.warning(f"Streamlit模組導入失敗: {e}")
                return False
            
            logger.info("✅ Streamlit相容性測試通過")
            return True
            
        except Exception as e:
            logger.error(f"❌ Streamlit相容性測試失敗: {e}")
            return False
    
    def test_configuration_consistency(self) -> bool:
        """測試配置一致性"""
        logger.info("測試配置一致性...")
        
        try:
            from src.common.config import (
                DEVICE, USE_AMP, LOGS_DIR, TIMESTEPS,
                ACCOUNT_CURRENCY, INITIAL_CAPITAL
            )
            
            # 檢查關鍵配置
            config_checks = {
                'DEVICE': DEVICE in ['cpu', 'cuda', 'auto'],
                'USE_AMP': isinstance(USE_AMP, bool),
                'LOGS_DIR': isinstance(LOGS_DIR, (str, type(None))),
                'TIMESTEPS': isinstance(TIMESTEPS, int) and TIMESTEPS > 0,
                'ACCOUNT_CURRENCY': isinstance(ACCOUNT_CURRENCY, str) and len(ACCOUNT_CURRENCY) == 3,
                'INITIAL_CAPITAL': isinstance(INITIAL_CAPITAL, (int, float)) and INITIAL_CAPITAL > 0
            }
            
            for config_name, is_valid in config_checks.items():
                if not is_valid:
                    logger.error(f"配置檢查失敗: {config_name}")
                    return False
                logger.info(f"配置檢查通過: {config_name}")
            
            logger.info("✅ 配置一致性測試通過")
            return True
            
        except Exception as e:
            logger.error(f"❌ 配置一致性測試失敗: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """執行所有測試"""
        logger.info("開始執行完整測試套件...")
        
        # 定義測試順序
        tests = [
            ("模組導入測試", self.test_imports),
            ("GPU設置測試", self.test_gpu_setup),
            ("共享數據管理器測試", self.test_shared_data_manager),
            ("檔案管理測試", self.test_file_management),
            ("mmap清理測試", self.test_mmap_cleanup),
            ("訓練器初始化測試", self.test_trainer_initialization),
            ("Streamlit相容性測試", self.test_streamlit_compatibility),
            ("配置一致性測試", self.test_configuration_consistency)
        ]
        
        # 執行所有測試
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            time.sleep(1)  # 測試間隔
        
        # 生成測試報告
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """生成測試報告"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        total_tests = len(self.test_results)
        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)
        success_rate = (passed_count / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_count,
                'failed': failed_count,
                'success_rate': success_rate,
                'total_duration': total_duration,
                'start_time': self.start_time,
                'end_time': end_time
            },
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'detailed_results': self.test_results
        }
        
        # 輸出報告
        logger.info("=" * 60)
        logger.info("📊 整合測試報告")
        logger.info("=" * 60)
        logger.info(f"總測試數: {total_tests}")
        logger.info(f"通過測試: {passed_count}")
        logger.info(f"失敗測試: {failed_count}")
        logger.info(f"成功率: {success_rate:.1f}%")
        logger.info(f"總耗時: {total_duration:.2f}秒")
        
        if self.passed_tests:
            logger.info("\n✅ 通過的測試:")
            for test in self.passed_tests:
                duration = self.test_results[test]['duration']
                logger.info(f"  - {test} ({duration:.2f}秒)")
        
        if self.failed_tests:
            logger.info("\n❌ 失敗的測試:")
            for test in self.failed_tests:
                duration = self.test_results[test]['duration']
                error = self.test_results[test].get('error', '未知錯誤')
                logger.info(f"  - {test} ({duration:.2f}秒) - {error}")
        
        logger.info("=" * 60)
        
        if success_rate >= 80:
            logger.info("🎉 整合測試整體通過！系統準備就緒。")
        elif success_rate >= 60:
            logger.warning("⚠️ 整合測試部分通過，建議檢查失敗的測試。")
        else:
            logger.error("❌ 整合測試失敗率過高，需要修復問題後重新測試。")
        
        return report


def main():
    """主函數"""
    try:
        # 創建測試套件
        test_suite = IntegrationTestSuite()
        
        # 執行所有測試
        report = test_suite.run_all_tests()
        
        # 保存報告到檔案
        import json
        report_file = Path('integration_test_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            # 轉換datetime對象為字符串
            def datetime_converter(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            json.dump(report, f, ensure_ascii=False, indent=2, default=datetime_converter)
        
        logger.info(f"詳細測試報告已保存到: {report_file}")
        
        # 返回成功率
        success_rate = report['summary']['success_rate']
        return success_rate >= 80
        
    except Exception as e:
        logger.error(f"整合測試執行失敗: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)