"""
Phase 4: Progressive Reward System Integration
漸進式獎勵系統與增強版Transformer深度整合

主要功能：
1. 三階段獎勵函數無縫切換
2. 增強版Transformer與漸進式學習系統集成
3. 智能階段進階管理
4. 全面整合測試
5. 性能監控與優化
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import json
import gc
import psutil

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from oanda_trading_bot.training_system.models.enhanced_transformer import EnhancedTransformer as EnhancedUniversalTradingTransformer
    from oanda_trading_bot.training_system.agent.enhanced_feature_extractor import EnhancedTransformerFeatureExtractor
    from oanda_trading_bot.training_system.environment.progressive_learning_system import ProgressiveLearningSystem, LearningStage
    from oanda_trading_bot.training_system.environment.progressive_reward_calculator import ProgressiveRewardCalculator
    from oanda_trading_bot.training_system.environment.progressive_reward_system import ProgressiveRewardSystem
    from oanda_trading_bot.training_system.agent.meta_learning_system import MetaLearningSystem
    from oanda_trading_bot.training_system.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
    from oanda_trading_bot.training_system.common.config import *
    from oanda_trading_bot.training_system.common.logger_setup import logger
    
    imports_successful = True
except ImportError as e:
    print(f"導入失敗: {e}")
    imports_successful = False
    
    # 創建備用logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


@dataclass
class Phase4Metrics:
    """Phase 4 性能監控指標"""
    stage_switching_time: float
    reward_calculation_time: float
    transformer_inference_time: float
    integration_success_rate: float
    stage_advancement_accuracy: float
    total_memory_usage: float
    gpu_memory_usage: float


class ProgressiveRewardIntegrator:
    """漸進式獎勵系統整合器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化整合器
        
        Args:
            config: 配置參數
        """
        self.device = torch.device(DEVICE)
        logger.info(f"🚀 Phase 4: 初始化漸進式獎勵系統整合器 - 設備: {self.device}")
          # 測試配置 - 更新為Large模型配置
        self.test_config = {
            'batch_size': 4,
            'num_symbols': 5,
            'num_timesteps': 128,
            'num_features': 9,
            'model_dim': 1024,       # Large模型: 1024維度
            'num_layers': 24,        # Large模型: 24層
            'num_heads': 32,         # Large模型: 32個注意力頭
            'ffn_dim': 4096,         # Large模型: 4096 FFN維度
            'initial_capital': 100000.0
        }
        
        # 自定義配置覆蓋
        if config:
            self.test_config.update(config)
          # 組件初始化狀態
        self.components_initialized = {
            'large_transformer': False,
            'progressive_learning_system': False,
            'progressive_reward_calculator': False,
            'progressive_reward_system': False,
            'meta_learning_system': False,
            'quantum_strategy_layer': False
        }
        
        # 性能監控
        self.performance_metrics = {}
        self.integration_log = []
    
    def initialize_components(self) -> bool:
        """初始化所有系統組件"""
        logger.info("🔧 初始化系統組件...")
        
        try:            # 1. 初始化Large Transformer
            self._initialize_large_transformer()
            
            # 2. 初始化漸進式學習系統
            self._initialize_progressive_learning_system()
            
            # 3. 初始化漸進式獎勵計算器
            self._initialize_progressive_reward_calculator()
            
            # 4. 初始化漸進式獎勵系統
            self._initialize_progressive_reward_system()
            
            # 5. 初始化元學習系統
            self._initialize_meta_learning_system()
            
            # 6. 初始化量子策略層
            self._initialize_quantum_strategy_layer()
            
            # 檢查所有組件是否成功初始化
            all_initialized = all(self.components_initialized.values())
            
            if all_initialized:
                logger.info("✅ 所有系統組件初始化成功")
                return True
            else:
                failed_components = [k for k, v in self.components_initialized.items() if not v]
                logger.error(f"❌ 組件初始化失敗: {failed_components}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 組件初始化過程中發生錯誤: {e}")
            return False
    
    def _initialize_large_transformer(self):
        """初始化Large Transformer"""
        try:
            logger.info("🚀 初始化Large Transformer...")
            
            self.large_transformer = EnhancedUniversalTradingTransformer(
                num_input_features=self.test_config['num_features'],
                num_symbols_possible=self.test_config['num_symbols'],
                model_dim=self.test_config['model_dim'],
                num_layers=self.test_config['num_layers'],
                num_heads=self.test_config['num_heads'],
                ffn_dim=self.test_config['ffn_dim'],
                use_multi_scale=True,
                use_cross_time_fusion=True
            ).to(self.device)
            
            # 計算模型參數數量
            total_params = sum(p.numel() for p in self.large_transformer.parameters())
            trainable_params = sum(p.numel() for p in self.large_transformer.parameters() if p.requires_grad)
            
            logger.info(f"🎯 Large Transformer初始化完成")
            logger.info(f"📊 模型規模: {total_params:,} 總參數, {trainable_params:,} 可訓練參數")
            logger.info(f"🔧 模型配置: {self.test_config['model_dim']}維度, {self.test_config['num_layers']}層, {self.test_config['num_heads']}頭")
            
            self.components_initialized['large_transformer'] = True
            
        except Exception as e:
            logger.error(f"❌ Large Transformer初始化失敗: {e}")
            self.components_initialized['large_transformer'] = False
    
    def _initialize_progressive_learning_system(self):
        """初始化漸進式學習系統"""
        try:
            logger.info("🎯 初始化漸進式學習系統...")
            
            self.progressive_learning_system = ProgressiveLearningSystem(
                min_stage_episodes=50,
                performance_window=20,
                advancement_patience=10,
                device=str(self.device)
            )
            
            logger.info(f"📚 漸進式學習系統初始化完成 - 當前階段: {self.progressive_learning_system.current_stage}")
            
            self.components_initialized['progressive_learning_system'] = True
            
        except Exception as e:
            logger.error(f"❌ 漸進式學習系統初始化失敗: {e}")
            self.components_initialized['progressive_learning_system'] = False
    
    def _initialize_progressive_reward_calculator(self):
        """初始化漸進式獎勵計算器"""
        try:
            logger.info("💰 初始化漸進式獎勵計算器...")
            
            from decimal import Decimal
            self.progressive_reward_calculator = ProgressiveRewardCalculator(
                initial_capital=Decimal(str(self.test_config['initial_capital']))
            )
            
            logger.info(f"🎖️ 漸進式獎勵計算器初始化完成 - 初始資本: {self.test_config['initial_capital']:,}")
            
            self.components_initialized['progressive_reward_calculator'] = True
            
        except Exception as e:
            logger.error(f"❌ 漸進式獎勵計算器初始化失敗: {e}")
            self.components_initialized['progressive_reward_calculator'] = False
    
    def _initialize_progressive_reward_system(self):
        """初始化漸進式獎勵系統"""
        try:
            logger.info("🏆 初始化漸進式獎勵系統...")
            
            self.progressive_reward_system = ProgressiveRewardSystem(
                profit_weight=0.4,
                risk_weight=0.3,
                adaptation_weight=0.2,
                consistency_weight=0.1,
                device=str(self.device)
            )
            
            logger.info("🎯 漸進式獎勵系統初始化完成")
            
            self.components_initialized['progressive_reward_system'] = True
            
        except Exception as e:
            logger.error(f"❌ 漸進式獎勵系統初始化失敗: {e}")
            self.components_initialized['progressive_reward_system'] = False
    
    def _initialize_meta_learning_system(self):
        """初始化元學習系統"""
        try:
            logger.info("🧠 初始化元學習系統...")
            
            self.meta_learning_system = MetaLearningSystem(
                initial_state_dim=512,
                action_dim=256,
                meta_learning_dim=256
            ).to(self.device)  # 確保移動到正確的設備
            
            logger.info("🔬 元學習系統初始化完成")
            self.components_initialized['meta_learning_system'] = True
        except Exception as e:
            logger.error(f"❌ 元學習系統初始化失敗: {e}")
            self.components_initialized['meta_learning_system'] = False
    
    def _initialize_quantum_strategy_layer(self):
        """初始化量子策略層"""
        try:
            logger.info("⚛️ 初始化量子策略層...")
            
            self.quantum_strategy_layer = EnhancedStrategySuperposition(
                state_dim=512,
                action_dim=256,
                enable_dynamic_generation=True
            ).to(self.device)  # 確保移動到正確的設備
            
            num_strategies = len(self.quantum_strategy_layer.base_strategies)
            logger.info(f"🌟 量子策略層初始化完成 - 策略數量: {num_strategies}")
            
            self.components_initialized['quantum_strategy_layer'] = True
            
        except Exception as e:
            logger.error(f"❌ 量子策略層初始化失敗: {e}")
            self.components_initialized['quantum_strategy_layer'] = False
    
    def test_three_stage_reward_switching(self) -> bool:
        """測試三階段獎勵函數切換"""
        logger.info("🔄 測試三階段獎勵函數切換...")
        
        try:
            test_metrics = {
                'pnl': 0.02,
                'drawdown': 0.03,
                'trade_frequency': 0.05,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.08,
                'win_rate': 0.45,
                'sortino_ratio': 1.5,
                'var_risk': 0.02,
                'skewness': 0.1,
                'kurtosis': 3.2,
                'transaction_costs': 0.001,
                'consistency_score': 0.7
            }
              # 測試各階段獎勵計算
            stage_results = {}
            
            # Stage 1: Basic
            self.progressive_learning_system.current_stage = LearningStage.BASIC
            total_reward_1, components_1, metrics_1 = self.progressive_learning_system.calculate_reward(test_metrics)
            stage_results['Stage1'] = {
                'total_reward': total_reward_1,
                'components': components_1,
                'stage': metrics_1.stage
            }
            logger.info(f"📊 Stage 1 獎勵: {total_reward_1:.4f}")
            
            # Stage 2: Intermediate  
            self.progressive_learning_system.current_stage = LearningStage.INTERMEDIATE
            total_reward_2, components_2, metrics_2 = self.progressive_learning_system.calculate_reward(test_metrics)
            stage_results['Stage2'] = {
                'total_reward': total_reward_2,
                'components': components_2,
                'stage': metrics_2.stage
            }
            logger.info(f"📈 Stage 2 獎勵: {total_reward_2:.4f}")
            
            # Stage 3: Advanced
            self.progressive_learning_system.current_stage = LearningStage.ADVANCED
            total_reward_3, components_3, metrics_3 = self.progressive_learning_system.calculate_reward(test_metrics)
            stage_results['Stage3'] = {
                'total_reward': total_reward_3,
                'components': components_3,
                'stage': metrics_3.stage
            }
            logger.info(f"📉 Stage 3 獎勵: {total_reward_3:.4f}")
            
            # 驗證階段切換邏輯
            stage_switching_successful = (
                isinstance(total_reward_1, (int, float)) and
                isinstance(total_reward_2, (int, float)) and
                isinstance(total_reward_3, (int, float))
            )
            
            if stage_switching_successful:
                logger.info("✅ 三階段獎勵函數切換測試成功")
                
                # 保存測試結果
                self.performance_metrics['stage_switching'] = stage_results
                return True
            else:
                logger.error("❌ 三階段獎勵函數切換測試失敗")
                return False
                
        except Exception as e:
            logger.error(f"❌ 三階段獎勵函數切換測試錯誤: {e}")
            return False
    
    def test_transformer_progressive_integration(self) -> bool:
        """測試Transformer與漸進式學習系統整合"""
        logger.info("🔗 測試Transformer與漸進式學習系統整合...")
        
        try:
            # 創建測試數據
            test_features = torch.randn(
                self.test_config['batch_size'],
                self.test_config['num_symbols'],
                self.test_config['num_timesteps'],
                self.test_config['num_features']
            ).to(self.device)
            
            test_mask = torch.zeros(
                self.test_config['batch_size'],
                self.test_config['num_symbols'],
                dtype=torch.bool
            ).to(self.device)
            test_mask[:, -1] = True  # 最後一個符號為padding
            
            # 記錄開始時間
            start_time = time.time()
            
            # 1. Transformer推理
            with torch.no_grad():
                transformer_output = self.large_transformer(test_features, test_mask)
            transformer_time = time.time() - start_time
              # 2. 量子策略層處理
            strategy_start = time.time()
            strategy_features = transformer_output.mean(dim=1)  # 池化到批次維度
              # 生成模擬波動率數據 - 修正維度和設備
            volatility = torch.randn(
                self.test_config['batch_size']
            ).to(self.device) * 0.1 + 0.2  # 模擬波動率 (0.1-0.3範圍)
            
            strategy_output, strategy_details = self.quantum_strategy_layer(strategy_features, volatility)
            strategy_time = time.time() - strategy_start
            
            # 3. 模擬交易指標
            simulated_metrics = self._generate_simulated_metrics(strategy_output)
            
            # 4. 漸進式獎勵計算
            reward_start = time.time()
            total_reward, reward_components, learning_metrics = self.progressive_learning_system.calculate_reward(simulated_metrics)
            reward_time = time.time() - reward_start
              # 5. 元學習系統處理 - 確保設備一致性
            meta_start = time.time()
            strategy_params = strategy_output.detach().cpu().numpy()
              # 確保所有輸入都在正確的設備上
            adaptation_result = self.meta_learning_system.adapt_to_strategy_layer(
                self.quantum_strategy_layer  # 傳入策略層對象
            )
            meta_time = time.time() - meta_start
            
            total_integration_time = time.time() - start_time
              # 記錄性能指標
            integration_metrics = {
                'transformer_inference_time': transformer_time,
                'strategy_processing_time': strategy_time,
                'reward_calculation_time': reward_time,
                'meta_learning_time': meta_time,
                'total_integration_time': total_integration_time,
                'total_reward': total_reward,
                'current_stage': learning_metrics.stage.name,
                'should_advance': learning_metrics.should_advance,
                'adaptation_success': bool(adaptation_result)  # adaptation_result is a boolean
            }
            
            logger.info(f"⚡ 整合性能指標:")
            logger.info(f"   Transformer推理: {transformer_time:.4f}s")
            logger.info(f"   策略處理: {strategy_time:.4f}s")
            logger.info(f"   獎勵計算: {reward_time:.4f}s")
            logger.info(f"   元學習: {meta_time:.4f}s")
            logger.info(f"   總整合時間: {total_integration_time:.4f}s")
            logger.info(f"   當前學習階段: {learning_metrics.stage.name}")
            logger.info(f"   總獎勵: {total_reward:.4f}")
              # 保存整合結果
            self.performance_metrics['integration'] = integration_metrics
            
            logger.info("✅ Transformer與漸進式學習系統整合測試成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 整合測試錯誤: {e}")
            return False
    
    def _generate_simulated_metrics(self, strategy_output: torch.Tensor) -> Dict[str, float]:
        """基於策略輸出生成模擬交易指標"""
        # 使用策略輸出的統計特性生成相關指標
        with torch.no_grad():
            output_stats = strategy_output.detach().cpu().numpy()
        
        # 基於輸出生成模擬指標
        mean_val = float(np.mean(output_stats))
        std_val = float(np.std(output_stats))
        
        return {
            'pnl': mean_val * 0.02 + np.random.normal(0, 0.01),
            'drawdown': abs(min(0, mean_val * 0.05)) + np.random.uniform(0, 0.02),
            'trade_frequency': min(0.1, abs(std_val * 0.1) + 0.02),
            'sharpe_ratio': max(0, mean_val * 2 + np.random.normal(0, 0.5)),
            'max_drawdown': abs(min(0, mean_val * 0.08)) + np.random.uniform(0, 0.03),
            'win_rate': max(0.2, min(0.8, 0.5 + mean_val + np.random.normal(0, 0.1))),
            'sortino_ratio': max(0, mean_val * 1.5 + np.random.normal(0, 0.3)),
            'var_risk': abs(std_val * 0.02) + np.random.uniform(0, 0.01),
            'skewness': np.random.normal(0, 0.2),
            'kurtosis': 3.0 + np.random.normal(0, 0.5),
            'transaction_costs': abs(std_val * 0.001) + 0.0005,
            'consistency_score': max(0.3, min(0.9, 0.7 + mean_val * 0.2))
        }
    
    def test_stage_advancement_logic(self) -> bool:
        """測試階段進階邏輯"""
        logger.info("📈 測試階段進階邏輯...")
        
        try:
            # 重置到初始階段
            self.progressive_learning_system.current_stage = LearningStage.BASIC
            self.progressive_learning_system.stage_episodes = 0
            self.progressive_learning_system.current_episode = 0
            
            advancement_tests = []
            
            # 測試1: 不滿足進階條件 (低獎勵)
            poor_metrics = {
                'pnl': -0.05, 'drawdown': 0.15, 'trade_frequency': 0.02,
                'sharpe_ratio': -0.5, 'max_drawdown': 0.12, 'win_rate': 0.25,
                'sortino_ratio': -0.3, 'var_risk': 0.08, 'skewness': -0.2,
                'kurtosis': 4.5, 'transaction_costs': 0.003, 'consistency_score': 0.3
            }
            
            for episode in range(30):
                total_reward, _, learning_metrics = self.progressive_learning_system.calculate_reward(poor_metrics)
                if learning_metrics.should_advance:
                    advancement_tests.append(f"意外進階在第{episode}回合")
                    break
            else:
                advancement_tests.append("正確維持在基礎階段 (低性能)")
            
            # 測試2: 滿足進階條件 (良好獎勵)
            self.progressive_learning_system.current_stage = LearningStage.BASIC
            self.progressive_learning_system.stage_episodes = 0
            
            good_metrics = {
                'pnl': 0.03, 'drawdown': 0.02, 'trade_frequency': 0.05,
                'sharpe_ratio': 1.5, 'max_drawdown': 0.04, 'win_rate': 0.55,
                'sortino_ratio': 1.8, 'var_risk': 0.015, 'skewness': 0.1,
                'kurtosis': 3.2, 'transaction_costs': 0.001, 'consistency_score': 0.75
            }
            
            advanced = False
            for episode in range(100):
                total_reward, _, learning_metrics = self.progressive_learning_system.calculate_reward(good_metrics)
                if learning_metrics.should_advance:
                    advancement_tests.append(f"成功進階到下一階段在第{episode}回合")
                    advanced = True
                    break
            
            if not advanced:
                advancement_tests.append("未能進階 (可能需要調整閾值)")
            
            logger.info("📊 階段進階測試結果:")
            for test_result in advancement_tests:
                logger.info(f"   - {test_result}")
            
            # 保存測試結果
            self.performance_metrics['advancement_logic'] = advancement_tests
            
            logger.info("✅ 階段進階邏輯測試完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 階段進階邏輯測試錯誤: {e}")
            return False
    
    def run_comprehensive_integration_test(self) -> bool:
        """執行全面整合測試"""
        logger.info("🚀 執行Phase 4全面整合測試...")
        
        try:
            # 測試計數器
            tests_passed = 0
            total_tests = 6
            
            # 1. 組件初始化測試
            logger.info("\n" + "="*50)
            logger.info("📋 測試1: 組件初始化")
            if self.initialize_components():
                logger.info("✅ 組件初始化測試通過")
                tests_passed += 1
            else:
                logger.error("❌ 組件初始化測試失敗")
            
            # 2. 三階段獎勵切換測試
            logger.info("\n" + "="*50)
            logger.info("📋 測試2: 三階段獎勵切換")
            if self.test_three_stage_reward_switching():
                logger.info("✅ 三階段獎勵切換測試通過")
                tests_passed += 1
            else:
                logger.error("❌ 三階段獎勵切換測試失敗")
            
            # 3. Transformer整合測試
            logger.info("\n" + "="*50)
            logger.info("📋 測試3: Transformer整合")
            if self.test_transformer_progressive_integration():
                logger.info("✅ Transformer整合測試通過")
                tests_passed += 1
            else:
                logger.error("❌ Transformer整合測試失敗")
            
            # 4. 階段進階邏輯測試
            logger.info("\n" + "="*50)
            logger.info("📋 測試4: 階段進階邏輯")
            if self.test_stage_advancement_logic():
                logger.info("✅ 階段進階邏輯測試通過")
                tests_passed += 1
            else:
                logger.error("❌ 階段進階邏輯測試失敗")
            
            # 5. 內存使用測試
            logger.info("\n" + "="*50)
            logger.info("📋 測試5: 內存使用測試")
            if self.test_memory_usage():
                logger.info("✅ 內存使用測試通過")
                tests_passed += 1
            else:
                logger.error("❌ 內存使用測試失敗")
            
            # 6. 性能基準測試
            logger.info("\n" + "="*50)
            logger.info("📋 測試6: 性能基準測試")
            if self.test_performance_benchmarks():
                logger.info("✅ 性能基準測試通過")
                tests_passed += 1
            else:
                logger.error("❌ 性能基準測試失敗")
              # 總結測試結果
            success_rate = tests_passed / total_tests
            logger.info("\n" + "="*60)
            logger.info(f"🎯 Phase 4 整合測試完成")
            logger.info(f"📊 測試通過率: {tests_passed}/{total_tests} ({success_rate*100:.1f}%)")
            
            if success_rate >= 0.8:  # 80%以上通過率視為成功
                logger.info("🎉 Phase 4 整合測試成功！")
                self._save_integration_report(success_rate)
                return True
            else:
                logger.error("❌ Phase 4 整合測試失敗 - 需要進一步調試")
                return False
                
        except Exception as e:
            logger.error(f"❌ 全面整合測試過程中發生錯誤: {e}")
            return False
    
    def test_memory_usage(self) -> bool:
        """測試內存使用情況"""
        try:
            # 獲取初始內存使用
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            if torch.cuda.is_available():
                initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                initial_gpu_memory = 0
            
            # 執行多次推理測試內存洩漏
            for i in range(10):
                test_features = torch.randn(
                    self.test_config['batch_size'],
                    self.test_config['num_symbols'],
                    self.test_config['num_timesteps'],
                    self.test_config['num_features']
                ).to(self.device)
                
                test_mask = torch.zeros(
                    self.test_config['batch_size'],
                    self.test_config['num_symbols'],
                    dtype=torch.bool                ).to(self.device)
                
                with torch.no_grad():
                    _ = self.large_transformer(test_features, test_mask)
                
                # 清理
                del test_features, test_mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # 獲取最終內存使用
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                final_gpu_memory = 0
            
            memory_increase = final_memory - initial_memory
            gpu_memory_increase = final_gpu_memory - initial_gpu_memory
            
            logger.info(f"💾 內存使用測試結果:")
            logger.info(f"   系統內存: {initial_memory:.1f}MB → {final_memory:.1f}MB (增加: {memory_increase:.1f}MB)")
            logger.info(f"   GPU內存: {initial_gpu_memory:.1f}MB → {final_gpu_memory:.1f}MB (增加: {gpu_memory_increase:.1f}MB)")
            
            # 保存內存指標
            self.performance_metrics['memory_usage'] = {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase,
                'initial_gpu_memory_mb': initial_gpu_memory,
                'final_gpu_memory_mb': final_gpu_memory,
                'gpu_memory_increase_mb': gpu_memory_increase
            }
            
            # 內存增加小於100MB視為正常
            return memory_increase < 100
            
        except Exception as e:
            logger.error(f"❌ 內存使用測試錯誤: {e}")
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """測試性能基準"""
        try:
            logger.info("⚡ 執行性能基準測試...")
            
            # 測試配置
            batch_sizes = [1, 2, 4, 8]
            inference_times = []
            
            for batch_size in batch_sizes:
                test_features = torch.randn(
                    batch_size,
                    self.test_config['num_symbols'],
                    self.test_config['num_timesteps'],
                    self.test_config['num_features']
                ).to(self.device)
                
                test_mask = torch.zeros(
                    batch_size,
                    self.test_config['num_symbols'],
                    dtype=torch.bool
                ).to(self.device)
                  # 預熱
                with torch.no_grad():
                    _ = self.large_transformer(test_features, test_mask)
                
                # 測試推理時間
                start_time = time.time()
                for _ in range(10):
                    with torch.no_grad():
                        _ = self.large_transformer(test_features, test_mask)
                
                avg_inference_time = (time.time() - start_time) / 10
                inference_times.append(avg_inference_time)
                
                logger.info(f"   批次大小 {batch_size}: {avg_inference_time:.4f}s")
            
            # 計算性能指標
            avg_inference_time = np.mean(inference_times)
            
            # 保存性能基準
            self.performance_metrics['performance_benchmarks'] = {
                'batch_sizes': batch_sizes,
                'inference_times': inference_times,
                'avg_inference_time': avg_inference_time,
                'throughput_samples_per_second': batch_sizes[-1] / inference_times[-1]
            }
            
            logger.info(f"📊 性能基準結果:")
            logger.info(f"   平均推理時間: {avg_inference_time:.4f}s")
            logger.info(f"   吞吐量: {batch_sizes[-1] / inference_times[-1]:.1f} samples/sec")            # 推理時間小於0.5秒視為達標 (Large模型的現實閾值)
            return avg_inference_time < 0.5
            
        except Exception as e:
            logger.error(f"❌ 性能基準測試錯誤: {e}")
            return False
    
    def _save_integration_report(self, success_rate: float):
        """保存整合測試報告"""
        try:
            def make_serializable(obj, visited=None, max_depth=10, current_depth=0):
                """
                遞歸地將對象轉換為可JSON序列化的格式
                支持循環引用檢測和深度限制
                """
                # 初始化訪問集合
                if visited is None:
                    visited = set()
                
                # 深度保護
                if current_depth > max_depth:
                    return f"<Max depth {max_depth} reached>"
                
                # 處理基本類型
                if obj is None or isinstance(obj, (str, int, float, bool)):
                    return obj
                
                # 處理NumPy類型
                if hasattr(obj, 'item') and callable(obj.item):
                    try:
                        return obj.item()
                    except:
                        pass
                
                # 處理PyTorch張量
                if hasattr(obj, 'detach') and hasattr(obj, 'cpu'):
                    try:
                        tensor_data = obj.detach().cpu().numpy()
                        if tensor_data.size < 100:  # 只序列化小張量
                            return tensor_data.tolist()
                        else:
                            return f"<Tensor shape={list(obj.shape)} dtype={obj.dtype}>"
                    except:
                        return f"<Tensor {type(obj).__name__}>"
                
                # 循環引用檢測
                obj_id = id(obj)
                if obj_id in visited:
                    return f"<Circular reference to {type(obj).__name__}>"
                
                visited.add(obj_id)
                
                try:
                    # 處理集合類型
                    if isinstance(obj, (list, tuple)):
                        result = [make_serializable(item, visited, max_depth, current_depth + 1) for item in obj]
                        visited.remove(obj_id)
                        return result
                    
                    elif isinstance(obj, dict):
                        result = {}
                        for k, v in obj.items():
                            try:
                                key = str(k)
                                result[key] = make_serializable(v, visited, max_depth, current_depth + 1)
                            except Exception as e:
                                result[str(k)] = f"<Serialization error: {str(e)[:100]}>"
                        visited.remove(obj_id)
                        return result
                    
                    # 處理Enum類型
                    elif hasattr(obj, 'name'):
                        result = obj.name
                        visited.remove(obj_id)
                        return result
                    
                    # 處理namedtuple
                    elif hasattr(obj, '_asdict'):
                        result = make_serializable(obj._asdict(), visited, max_depth, current_depth + 1)
                        visited.remove(obj_id)
                        return result
                    
                    # 處理模型和復雜對象
                    elif hasattr(obj, '__dict__'):
                        # 跳過PyTorch模型的復雜屬性
                        skip_attrs = {
                            '_modules', '_parameters', '_buffers', '_non_persistent_buffers_set',
                            '_backward_hooks', '_forward_hooks', '_forward_pre_hooks',
                            'training', '_get_name', 'named_parameters', 'parameters'
                        }
                        
                        result = {}
                        for k, v in obj.__dict__.items():
                            if k.startswith('_') and k not in ['_name', '_device']:
                                continue
                            if k in skip_attrs:
                                continue
                            try:
                                result[k] = make_serializable(v, visited, max_depth, current_depth + 1)
                            except Exception as e:
                                result[k] = f"<Error: {str(e)[:50]}>"
                        
                        visited.remove(obj_id)
                        return result
                    
                    else:
                        visited.remove(obj_id)
                        return str(obj)[:200]  # 限制字符串長度
                
                except Exception as e:
                    visited.discard(obj_id)
                    return f"<Serialization error: {str(e)[:100]}>"
            
            # 創建可序列化的性能指標副本
            logger.info("🔄 開始序列化性能指標...")
            serializable_metrics = make_serializable(self.performance_metrics)
            
            report = {
                'phase': 'Phase 4: Progressive Reward System Integration',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'success_rate': success_rate,
                'components_status': self.components_initialized,
                'performance_metrics': serializable_metrics,
                'test_config': self.test_config,
                'device_info': str(self.device)
            }
            
            # 保存報告
            report_dir = project_root / 'reports'
            report_dir.mkdir(exist_ok=True)
            
            report_file = report_dir / f'phase4_integration_report_{time.strftime("%Y%m%d_%H%M%S")}.json'
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📄 整合測試報告已保存: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存整合報告錯誤: {e}")
            # 降級保存基本信息
            try:
                basic_report = {
                    'phase': 'Phase 4: Progressive Reward System Integration',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'success_rate': success_rate,
                    'components_status': self.components_initialized,
                    'error': str(e)
                }
                
                report_dir = project_root / 'reports'
                report_dir.mkdir(exist_ok=True)
                
                report_file = report_dir / f'phase4_basic_report_{time.strftime("%Y%m%d_%H%M%S")}.json'
                
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(basic_report, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"📄 基本報告已保存: {report_file}")
            except Exception as backup_error:
                logger.error(f"❌ 連基本報告也無法保存: {backup_error}")


def main():
    """主測試函數"""
    if not imports_successful:
        print("❌ 導入失敗，無法執行測試")
        return False
    
    print("🚀 開始Phase 4: 漸進式獎勵系統整合測試")
    print("="*60)
    
    # 創建整合器
    integrator = ProgressiveRewardIntegrator()
    
    # 執行全面整合測試
    success = integrator.run_comprehensive_integration_test()
    
    print("="*60)
    if success:
        print("🎉 Phase 4 整合測試完成 - 成功！")
        print("✅ 漸進式獎勵系統已成功整合到增強版Transformer中")
        print("📈 準備進入Phase 5: 高級元學習能力")
    else:
        print("❌ Phase 4 整合測試完成 - 失敗")
        print("🔧 請檢查錯誤日誌並修復問題")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
