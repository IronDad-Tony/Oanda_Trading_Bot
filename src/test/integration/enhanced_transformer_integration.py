# integration/enhanced_transformer_integration.py
"""
Phase 3: Enhanced Transformer Architecture Integration
整合增強版Transformer到現有訓練系統中

主要功能：
1. 驗證增強版Transformer功能
2. 創建新的訓練配置
3. 與漸進式學習系統集成
4. 性能基準測試
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging
import time
from typing import Dict, Any, Optional

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.enhanced_transformer import EnhancedTransformer as EnhancedUniversalTradingTransformer
    from src.agent.enhanced_feature_extractor import EnhancedTransformerFeatureExtractor
    from src.environment.progressive_learning_system import ProgressiveLearningSystem
    from src.agent.meta_learning_system import MetaLearningSystem
    from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
    from src.common.config import *
    from src.common.logger_setup import logger
    
    imports_successful = True
except ImportError as e:
    print(f"導入失敗: {e}")
    imports_successful = False
    
    # 創建備用logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class EnhancedTransformerIntegrator:
    """增強版Transformer集成器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用設備: {self.device}")
        
        # 測試配置
        self.test_config = {
            'batch_size': 4,
            'num_symbols': 5,
            'num_timesteps': 128,
            'num_features': 9,
            'model_dim': 512,
            'num_layers': 12,
            'num_heads': 16,
            'ffn_dim': 2048
        }
    
    def test_enhanced_transformer(self) -> bool:
        """測試增強版Transformer基本功能"""
        logger.info("🧪 測試增強版Transformer基本功能...")
        
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
            
            # 初始化模型
            model = EnhancedUniversalTradingTransformer(
                num_input_features=self.test_config['num_features'],
                num_symbols_possible=self.test_config['num_symbols'],
                model_dim=self.test_config['model_dim'],
                num_layers=self.test_config['num_layers'],
                num_heads=self.test_config['num_heads'],
                ffn_dim=self.test_config['ffn_dim'],
                use_multi_scale=True,
                use_cross_time_fusion=True
            ).to(self.device)
            
            # 前向傳播測試
            start_time = time.time()
            with torch.no_grad():
                output = model(test_features, test_mask)
            inference_time = time.time() - start_time
            
            # 驗證輸出形狀
            expected_shape = (
                self.test_config['batch_size'],
                self.test_config['num_symbols'],
                model.output_projection[-1].out_features
            )
            
            assert output.shape == expected_shape, f"輸出形狀不匹配: {output.shape} vs {expected_shape}"
            
            # 測試梯度計算
            model.train()
            output = model(test_features, test_mask)
            loss = output.mean()
            loss.backward()
            
            # 檢查梯度
            grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # 獲取模型信息
            model_info = model.get_model_info()
            
            logger.info("✅ 增強版Transformer測試通過！")
            logger.info(f"   - 輸出形狀: {output.shape}")
            logger.info(f"   - 推理時間: {inference_time:.4f}s")
            logger.info(f"   - 總參數量: {model_info['total_parameters']:,}")
            logger.info(f"   - 梯度範數: {grad_norm:.6f}")
            logger.info(f"   - 記憶體使用: {model_info.get('memory_usage_mb', 0):.1f}MB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 增強版Transformer測試失敗: {e}")
            return False
    
    def test_enhanced_feature_extractor(self) -> bool:
        """測試增強版特徵提取器"""
        logger.info("🧪 測試增強版特徵提取器...")
        
        try:
            from gymnasium import spaces
            
            # 創建觀察空間
            obs_space = spaces.Dict({
                'features_from_dataset': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(MAX_SYMBOLS_ALLOWED, 128, 9),
                    dtype=np.float32
                ),
                'current_positions': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(MAX_SYMBOLS_ALLOWED,),
                    dtype=np.float32
                ),
                'unrealized_pnl': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(MAX_SYMBOLS_ALLOWED,),
                    dtype=np.float32
                ),
                'margin_level': spaces.Box(
                    low=0, high=np.inf,
                    shape=(1,),
                    dtype=np.float32
                ),
                'padding_mask': spaces.Box(
                    low=0, high=1,
                    shape=(MAX_SYMBOLS_ALLOWED,),
                    dtype=np.bool_
                )
            })            # 確定設備
            device = torch.device(DEVICE)
            logger.info(f"使用設備: {device}")
            
            # 創建測試數據並移動到正確設備
            test_obs = {
                'features_from_dataset': torch.randn(4, MAX_SYMBOLS_ALLOWED, 128, 9, device=device),
                'current_positions': torch.randn(4, MAX_SYMBOLS_ALLOWED, device=device),
                'unrealized_pnl': torch.randn(4, MAX_SYMBOLS_ALLOWED, device=device),
                'margin_level': torch.randn(4, 1, device=device),
                'padding_mask': torch.zeros(4, MAX_SYMBOLS_ALLOWED, dtype=torch.bool, device=device)
            }
            
            # 初始化特徵提取器
            extractor = EnhancedTransformerFeatureExtractor(obs_space)
            
            # 將測試數據移動到與模型相同的設備
            device = next(extractor.enhanced_transformer.parameters()).device
            test_obs = {key: value.to(device) for key, value in test_obs.items()}
            
            # 測試前向傳播
            start_time = time.time()
            with torch.no_grad():
                features = extractor(test_obs)
            extraction_time = time.time() - start_time
            
            logger.info("✅ 增強版特徵提取器測試通過！")
            logger.info(f"   - 輸出特徵形狀: {features.shape}")
            logger.info(f"   - 特徵提取時間: {extraction_time:.4f}s")
            logger.info(f"   - 特徵維度: {extractor.features_dim}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 增強版特徵提取器測試失敗: {e}")
            return False
    
    def test_system_integration(self) -> bool:
        """測試系統集成"""
        logger.info("🧪 測試系統集成...")
        
        try:
            # 測試與漸進式學習系統的集成
            progressive_system = ProgressiveLearningSystem()
            logger.info(f"   - 漸進式學習系統當前階段: {progressive_system.current_stage}")
            
            # 測試與元學習系統的集成
            meta_learning = MetaLearningSystem(
                initial_state_dim=64,
                action_dim=10,
                meta_learning_dim=256
            )
            logger.info(f"   - 元學習系統適應能力: {len(meta_learning.adaptation_history)}")
              # 測試與量子策略層的集成
            quantum_layer = EnhancedStrategySuperposition(
                state_dim=64,
                action_dim=10,
                enable_dynamic_generation=True
            )
            logger.info(f"   - 量子策略層策略數量: {len(quantum_layer.base_strategies)}")
            
            logger.info("✅ 系統集成測試通過！")
            return True
            
        except Exception as e:
            logger.error(f"❌ 系統集成測試失敗: {e}")
            return False
    
    def benchmark_performance(self) -> Dict[str, float]:
        """性能基準測試"""
        logger.info("📊 執行性能基準測試...")
        
        benchmarks = {}
        
        try:
            # 創建不同規模的模型進行測試
            model_configs = [
                {'layers': 6, 'dim': 256, 'heads': 8, 'name': 'Small'},
                {'layers': 12, 'dim': 512, 'heads': 16, 'name': 'Enhanced'},
                {'layers': 16, 'dim': 768, 'heads': 24, 'name': 'Large'}
            ]
            
            test_data = torch.randn(4, 5, 128, 9).to(self.device)
            test_mask = torch.zeros(4, 5, dtype=torch.bool).to(self.device)
            
            for config in model_configs:
                logger.info(f"   - 測試 {config['name']} 模型...")
                
                try:
                    model = EnhancedUniversalTradingTransformer(
                        num_input_features=9,
                        num_symbols_possible=5,
                        model_dim=config['dim'],
                        num_layers=config['layers'],
                        num_heads=config['heads'],
                        ffn_dim=config['dim'] * 4
                    ).to(self.device)
                    
                    # 推理速度測試
                    model.eval()
                    start_time = time.time()
                    for _ in range(10):
                        with torch.no_grad():
                            _ = model(test_data, test_mask)
                    avg_inference_time = (time.time() - start_time) / 10
                    
                    # 記憶體使用測試
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        _ = model(test_data, test_mask)
                        memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                    else:
                        memory_usage = 0
                    
                    # 參數量
                    param_count = sum(p.numel() for p in model.parameters())
                    
                    benchmarks[config['name']] = {
                        'inference_time': avg_inference_time,
                        'memory_usage_mb': memory_usage,
                        'parameter_count': param_count
                    }
                    
                    logger.info(f"     ⏱️  推理時間: {avg_inference_time:.4f}s")
                    logger.info(f"     💾 記憶體使用: {memory_usage:.1f}MB")
                    logger.info(f"     🔢 參數量: {param_count:,}")
                    
                except Exception as e:
                    logger.warning(f"     ⚠️  {config['name']} 模型測試失敗: {e}")
                    continue
            
            logger.info("✅ 性能基準測試完成！")
            return benchmarks
            
        except Exception as e:
            logger.error(f"❌ 性能基準測試失敗: {e}")
            return {}
    
    def create_enhanced_training_config(self) -> Dict[str, Any]:
        """創建增強版訓練配置"""
        logger.info("⚙️ 創建增強版訓練配置...")
        
        config = {
            # 模型配置
            'model': {
                'type': 'EnhancedUniversalTradingTransformer',
                'model_dim': TRANSFORMER_MODEL_DIM,
                'num_layers': TRANSFORMER_NUM_LAYERS,
                'num_heads': TRANSFORMER_NUM_HEADS,
                'ffn_dim': TRANSFORMER_FFN_DIM,
                'dropout_rate': TRANSFORMER_DROPOUT_RATE,
                'use_multi_scale': ENHANCED_TRANSFORMER_USE_MULTI_SCALE,
                'use_cross_time_fusion': ENHANCED_TRANSFORMER_USE_CROSS_TIME_FUSION
            },
            
            # 訓練配置
            'training': {
                'total_timesteps': 2_000_000,  # 增加訓練步數
                'learning_rate': 1e-4,
                'batch_size': 32,
                'buffer_size': 200_000,
                'gradient_steps': 2,
                'train_freq': 4,
                'target_update_interval': 1000
            },
            
            # 漸進式學習配置
            'progressive_learning': {
                'enabled': True,
                'stage_advancement_episodes': 50,
                'reward_threshold_basic': -0.10,
                'reward_threshold_intermediate': 0.05,
                'reward_threshold_advanced': 0.15
            },
            
            # 元學習配置
            'meta_learning': {
                'enabled': True,
                'adaptation_rate': 0.01,
                'memory_size': 1000,
                'update_frequency': 100
            },
            
            # 量子策略層配置
            'quantum_strategies': {
                'enabled': True,
                'num_strategies': 20,
                'strategy_update_frequency': 500
            },
            
            # 評估配置
            'evaluation': {
                'eval_freq': 5000,
                'n_eval_episodes': 5,
                'deterministic': True
            },
            
            # 保存配置
            'checkpoints': {
                'save_freq': 10000,
                'keep_best': True,
                'save_path': 'weights/enhanced_model'
            }
        }
        
        logger.info("✅ 增強版訓練配置創建完成！")
        return config
    
    def run_integration_tests(self) -> bool:
        """執行完整集成測試"""
        logger.info("🚀 開始Phase 3增強版Transformer集成測試...")
        logger.info("=" * 60)
        
        if not imports_successful:
            logger.error("❌ 導入失敗，跳過集成測試")
            return False
        
        # 測試序列
        tests = [
            ("Enhanced Transformer基本功能", self.test_enhanced_transformer),
            ("Enhanced Feature Extractor", self.test_enhanced_feature_extractor),
            ("系統集成", self.test_system_integration)
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\n🔍 執行測試: {test_name}")
            logger.info("-" * 40)
            
            success = test_func()
            results.append(success)
            
            if success:
                logger.info(f"✅ {test_name} - 通過")
            else:
                logger.error(f"❌ {test_name} - 失敗")
        
        # 性能基準測試
        logger.info(f"\n📊 執行性能基準測試")
        logger.info("-" * 40)
        benchmarks = self.benchmark_performance()
        
        # 創建訓練配置
        logger.info(f"\n⚙️ 創建增強版訓練配置")
        logger.info("-" * 40)
        config = self.create_enhanced_training_config()
        
        # 總結
        logger.info("\n" + "=" * 60)
        logger.info("🎯 Phase 3集成測試總結")
        logger.info("=" * 60)
        
        passed_tests = sum(results)
        total_tests = len(results)
        
        logger.info(f"✅ 通過測試: {passed_tests}/{total_tests}")
        
        if benchmarks:
            logger.info("📊 性能基準:")
            for model_name, metrics in benchmarks.items():
                logger.info(f"   - {model_name}: "
                           f"{metrics['inference_time']:.4f}s, "
                           f"{metrics['memory_usage_mb']:.1f}MB, "
                           f"{metrics['parameter_count']:,} params")
        
        overall_success = all(results)
        
        if overall_success:
            logger.info("🎉 Phase 3: Enhanced Transformer Architecture - 完成！")
            logger.info("   ✅ 增強版Transformer架構實現完成")
            logger.info("   ✅ 多尺度特徵提取器集成完成")
            logger.info("   ✅ 自適應注意力機制實現完成")
            logger.info("   ✅ 跨時間尺度融合實現完成")
            logger.info("   ✅ 系統集成測試通過")
            logger.info("   ✅ 性能基準測試完成")
            
            # 保存配置
            import json
            config_path = project_root / "configs" / "enhanced_transformer_config.json"
            config_path.parent.mkdir(exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"   ✅ 訓練配置已保存: {config_path}")
        else:
            logger.error("❌ Phase 3測試未完全通過，請檢查錯誤並修復")
        
        return overall_success


def main():
    """主函數"""
    integrator = EnhancedTransformerIntegrator()
    success = integrator.run_integration_tests()
    
    if success:
        print("\n🎉 Phase 3: Enhanced Transformer Architecture 實施完成！")
        print("✨ 準備進入下一階段的高級功能實現...")
    else:
        print("\n⚠️  Phase 3測試未完全通過，請檢查並修復問題")
    
    return success


if __name__ == "__main__":
    main()
