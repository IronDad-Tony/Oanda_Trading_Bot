# src/agent/meta_learning_system.py
"""
自適應元學習系統實現
能夠動態檢測策略數量和維度，自動適應量子策略層的模組化變化

主要功能：
1. 自適應策略編碼器：動態調整輸入維度，支持權重遷移
2. 自動配置檢測：檢測策略數量和狀態維度變化
3. 安全參數載入：處理維度不匹配的模型載入
4. 維度變化歷史追蹤：記錄系統演化過程
"""

import sys
import os
from pathlib import Path

# 修復導入路徑問題
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import json
import pickle
from datetime import datetime

try:
    from src.common.logger_setup import logger
    from src.common.config import DEVICE, MAX_SYMBOLS_ALLOWED
    from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
    from src.environment.progressive_reward_system import ProgressiveRewardSystem
except ImportError as e:
    # 基礎日誌設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    DEVICE = "cpu"
    MAX_SYMBOLS_ALLOWED = 5
    
    # 創建空的替代類以避免錯誤
    class EnhancedStrategySuperposition:
        pass
    
    class ProgressiveRewardSystem:
        def __init__(self, *args, **kwargs):
            pass
    
    logger.warning(f"導入錯誤，使用基礎配置: {e}")


@dataclass
class ModelConfiguration:
    """模型配置信息"""
    state_dim: int
    action_dim: int
    num_strategies: int
    latent_dim: int
    strategy_names: List[str]
    timestamp: str
    version: str = "1.0"


@dataclass
class DimensionChange:
    """維度變化記錄"""
    old_config: ModelConfiguration
    new_config: ModelConfiguration
    change_type: str  # "strategy_count", "state_dim", "action_dim"
    timestamp: str
    migration_success: bool


class AdaptiveStrategyEncoder(nn.Module):
    """
    自適應策略編碼器
    能夠動態調整輸入維度並支持權重遷移
    """
    
    def __init__(self, initial_input_dim: int, output_dim: int, 
                 adaptive_layers: List[int] = None):
        super().__init__()
        self.initial_input_dim = initial_input_dim
        self.current_input_dim = initial_input_dim
        self.output_dim = output_dim
        
        # 默認自適應層配置
        if adaptive_layers is None:
            adaptive_layers = [512, 256, 128]
        
        self.adaptive_layers = adaptive_layers
        
        # 構建初始網絡
        self._build_network(initial_input_dim)
        
        # 記錄維度變化歷史
        self.dimension_history = []
        
    def _build_network(self, input_dim: int):
        """構建或重建網絡結構"""
        layers = []
        current_dim = input_dim
        
        # 輸入適應層
        layers.append(nn.Linear(current_dim, self.adaptive_layers[0]))
        layers.append(nn.GELU())
        layers.append(nn.LayerNorm(self.adaptive_layers[0]))
        layers.append(nn.Dropout(0.1))
        current_dim = self.adaptive_layers[0]
        
        # 中間層
        for hidden_dim in self.adaptive_layers[1:]:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.1))
            current_dim = hidden_dim
        
        # 輸出層
        layers.append(nn.Linear(current_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def adapt_input_dimension(self, new_input_dim: int, 
                            preserve_weights: bool = True) -> bool:
        """
        自適應調整輸入維度
        
        Args:
            new_input_dim: 新的輸入維度
            preserve_weights: 是否保留現有權重
            
        Returns:
            bool: 是否成功適應
        """
        if new_input_dim == self.current_input_dim:
            logger.info(f"輸入維度未改變: {new_input_dim}")
            return True
            
        logger.info(f"適應輸入維度變化: {self.current_input_dim} -> {new_input_dim}")
        
        old_network = None
        if preserve_weights:
            # 保存當前網絡狀態
            old_network = {
                'state_dict': self.network.state_dict().copy(),
                'input_dim': self.current_input_dim
            }
        
        # 記錄維度變化
        change_record = {
            'old_dim': self.current_input_dim,
            'new_dim': new_input_dim,
            'timestamp': datetime.now().isoformat(),
            'preserve_weights': preserve_weights
        }
        self.dimension_history.append(change_record)
        
        # 重建網絡
        self.current_input_dim = new_input_dim
        self._build_network(new_input_dim)
        
        # 權重遷移
        if preserve_weights and old_network:
            migration_success = self._migrate_weights(old_network, new_input_dim)
            change_record['migration_success'] = migration_success
            
            if not migration_success:
                logger.warning("權重遷移失敗，使用隨機初始化")
        
        return True
        
    def _migrate_weights(self, old_network: Dict, new_input_dim: int) -> bool:
        """
        遷移現有權重到新的網絡結構
        
        Args:
            old_network: 舊網絡信息
            new_input_dim: 新輸入維度
            
        Returns:
            bool: 遷移是否成功
        """
        try:
            old_state = old_network['state_dict']
            old_input_dim = old_network['input_dim']
            new_state = self.network.state_dict()
            
            # 遷移除第一層外的所有權重
            for name, param in old_state.items():
                if name in new_state:
                    if name == '0.weight':  # 第一層權重需要特殊處理
                        old_weight = param  # [output_dim, old_input_dim]
                        new_weight = new_state[name]  # [output_dim, new_input_dim]
                        
                        if new_input_dim >= old_input_dim:
                            # 維度增加：填充零或復制現有權重
                            new_weight[:, :old_input_dim] = old_weight
                            # 新增維度使用小隨機值初始化
                            if new_input_dim > old_input_dim:
                                nn.init.normal_(new_weight[:, old_input_dim:], 
                                              mean=0, std=0.01)
                        else:
                            # 維度減少：截取現有權重
                            new_weight[:] = old_weight[:, :new_input_dim]
                            
                        new_state[name] = new_weight
                        
                    elif name == '0.bias':  # 第一層偏置直接復制
                        new_state[name] = param
                        
                    elif param.shape == new_state[name].shape:
                        # 其他層形狀相同的參數直接復制
                        new_state[name] = param
            
            # 載入遷移後的權重
            self.network.load_state_dict(new_state)
            logger.info(f"權重遷移成功: {old_input_dim} -> {new_input_dim}")
            return True
            
        except Exception as e:
            logger.error(f"權重遷移失敗: {e}")
            return False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        if x.size(-1) != self.current_input_dim:
            logger.warning(f"輸入維度不匹配: 期望 {self.current_input_dim}, 得到 {x.size(-1)}")
            # 自動適應
            self.adapt_input_dimension(x.size(-1))
            
        return self.network(x)
    
    def get_dimension_history(self) -> List[Dict]:
        """獲取維度變化歷史"""
        return self.dimension_history.copy()


class MetaLearningSystem(nn.Module):
    """
    自適應元學習系統
    整合自適應編碼器、策略檢測和動態配置管理
    """
    
    def __init__(self, initial_state_dim: int, action_dim: int,
                 meta_learning_dim: int = 256,
                 config_detection_methods: List[str] = None):
        super().__init__()
        
        self.initial_state_dim = initial_state_dim
        self.action_dim = action_dim
        self.meta_learning_dim = meta_learning_dim
        
        # 配置檢測方法
        if config_detection_methods is None:
            config_detection_methods = [
                "parameter_analysis",
                "forward_pass_test", 
                "attribute_inspection",
                "layer_structure_analysis"
            ]
        self.config_detection_methods = config_detection_methods
        
        # 自適應策略編碼器
        self.strategy_encoder = AdaptiveStrategyEncoder(
            initial_input_dim=initial_state_dim,
            output_dim=meta_learning_dim
        )
        
        # 元學習控制器
        self.meta_controller = nn.Sequential(
            nn.Linear(meta_learning_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
        # 策略重要性評估器
        self.strategy_importance = nn.Sequential(
            nn.Linear(meta_learning_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 配置追蹤
        self.current_config = None
        self.config_history = []
        self.dimension_changes = []
        
        # 性能指標
        self.register_buffer('adaptation_success_rate', torch.tensor(0.0))
        self.register_buffer('total_adaptations', torch.tensor(0))
        
        logger.info(f"初始化自適應元學習系統 - 狀態維度: {initial_state_dim}, "
                   f"動作維度: {action_dim}, 元學習維度: {meta_learning_dim}")
    
    def detect_model_configuration(self, strategy_layer: nn.Module) -> ModelConfiguration:
        """
        檢測模型配置信息
        使用多種方法確保檢測準確性
        
        Args:
            strategy_layer: 量子策略層模型
            
        Returns:
            ModelConfiguration: 檢測到的配置
        """
        config_results = {}
        
        for method in self.config_detection_methods:
            try:
                if method == "parameter_analysis":
                    result = self._detect_by_parameter_analysis(strategy_layer)
                elif method == "forward_pass_test":
                    result = self._detect_by_forward_pass(strategy_layer)
                elif method == "attribute_inspection":
                    result = self._detect_by_attribute_inspection(strategy_layer)
                elif method == "layer_structure_analysis":
                    result = self._detect_by_layer_structure(strategy_layer)
                else:
                    continue
                    
                config_results[method] = result
                logger.debug(f"配置檢測方法 {method} 結果: {result}")
                
            except Exception as e:
                logger.warning(f"配置檢測方法 {method} 失敗: {e}")
                continue
        
        # 聚合檢測結果
        final_config = self._aggregate_detection_results(config_results)
        
        # 記錄配置變化
        if self.current_config is None or not self._configs_equal(self.current_config, final_config):
            if self.current_config is not None:
                change = DimensionChange(
                    old_config=self.current_config,
                    new_config=final_config,
                    change_type=self._get_change_type(self.current_config, final_config),
                    timestamp=datetime.now().isoformat(),
                    migration_success=True
                )
                self.dimension_changes.append(change)
                logger.info(f"檢測到配置變化: {change.change_type}")
                
            self.current_config = final_config
            self.config_history.append(final_config)
        
        return final_config
    
    def _detect_by_parameter_analysis(self, model: nn.Module) -> Dict[str, Any]:
        """通過參數分析檢測配置"""
        result = {
            'method': 'parameter_analysis',
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # 嘗試分析第一層參數形狀
        try:
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) == 2:
                    result['first_layer_input_dim'] = param.shape[1]
                    result['first_layer_output_dim'] = param.shape[0]
                    break
        except:
            pass
            
        return result
    
    def _detect_by_forward_pass(self, model: nn.Module) -> Dict[str, Any]:
        """通過前向傳播測試檢測配置"""
        result = {'method': 'forward_pass_test'}
        
        # 測試不同輸入維度
        test_dims = [32, 64, 96, 128, 256]
        working_dims = []
        
        model.eval()
        with torch.no_grad():
            for dim in test_dims:
                try:
                    test_input = torch.randn(2, dim)
                    test_volatility = torch.rand(2) * 0.5
                    
                    # 嘗試不同的調用方式
                    if hasattr(model, 'forward'):
                        output = model(test_input, test_volatility)
                        if isinstance(output, tuple):
                            action_output = output[0]
                        else:
                            action_output = output
                            
                        if action_output is not None and action_output.numel() > 0:
                            working_dims.append({
                                'input_dim': dim,
                                'output_shape': list(action_output.shape),
                                'output_dim': action_output.shape[-1] if len(action_output.shape) > 1 else action_output.numel()
                            })
                            
                except Exception as e:
                    logger.debug(f"維度 {dim} 測試失敗: {e}")
                    continue
        
        result['working_dimensions'] = working_dims
        if working_dims:
            result['recommended_state_dim'] = working_dims[0]['input_dim']
            result['action_dim'] = working_dims[0]['output_dim']
            
        return result
    
    def _detect_by_attribute_inspection(self, model: nn.Module) -> Dict[str, Any]:
        """通過屬性檢查檢測配置"""
        result = {'method': 'attribute_inspection'}
        
        # 檢查常見屬性
        common_attributes = [
            'state_dim', 'action_dim', 'num_strategies', 'latent_dim',
            'input_dim', 'output_dim', 'num_base_strategies'
        ]
        
        found_attributes = {}
        for attr in common_attributes:
            if hasattr(model, attr):
                value = getattr(model, attr)
                found_attributes[attr] = value
                
        result['attributes'] = found_attributes
        
        # 檢查策略列表
        if hasattr(model, 'base_strategies') and hasattr(model.base_strategies, '__len__'):
            try:
                strategy_names = []
                for i, strategy in enumerate(model.base_strategies):
                    if hasattr(strategy, 'get_strategy_name'):
                        name = strategy.get_strategy_name()
                        strategy_names.append(name)
                    else:
                        strategy_names.append(f"Strategy_{i}")
                        
                result['strategy_names'] = strategy_names
                result['num_strategies'] = len(strategy_names)
            except:
                pass
                
        return result
    
    def _detect_by_layer_structure(self, model: nn.Module) -> Dict[str, Any]:
        """通過層結構分析檢測配置"""
        result = {'method': 'layer_structure_analysis'}
        
        layer_info = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_info.append({
                    'name': name,
                    'input_dim': module.in_features,
                    'output_dim': module.out_features,
                    'has_bias': module.bias is not None
                })
        
        result['linear_layers'] = layer_info
        
        # 推斷輸入和輸出維度
        if layer_info:
            result['inferred_input_dim'] = layer_info[0]['input_dim']
            # 尋找最後一個可能的輸出層
            for layer in reversed(layer_info):
                if layer['output_dim'] <= 20:  # 假設動作維度不會太大
                    result['inferred_action_dim'] = layer['output_dim']
                    break
                    
        return result
    
    def _aggregate_detection_results(self, results: Dict[str, Dict]) -> ModelConfiguration:
        """聚合檢測結果"""
        # 提取關鍵信息
        state_dim = self.initial_state_dim
        action_dim = self.action_dim
        num_strategies = 15  # 默認值
        latent_dim = 256
        strategy_names = []
        
        # 從各方法結果中提取信息
        for method, result in results.items():
            if 'recommended_state_dim' in result:
                state_dim = result['recommended_state_dim']
            elif 'first_layer_input_dim' in result:
                state_dim = result['first_layer_input_dim']
            elif 'inferred_input_dim' in result:
                state_dim = result['inferred_input_dim']
                
            if 'action_dim' in result:
                action_dim = result['action_dim']
            elif 'inferred_action_dim' in result:
                action_dim = result['inferred_action_dim']
                
            if 'num_strategies' in result:
                num_strategies = result['num_strategies']
                
            if 'strategy_names' in result:
                strategy_names = result['strategy_names']
                
            # 從屬性中提取
            if 'attributes' in result:
                attrs = result['attributes']
                if 'state_dim' in attrs:
                    state_dim = attrs['state_dim']
                if 'action_dim' in attrs:
                    action_dim = attrs['action_dim']
                if 'num_strategies' in attrs:
                    num_strategies = attrs['num_strategies']
                if 'latent_dim' in attrs:
                    latent_dim = attrs['latent_dim']
        
        # 生成默認策略名稱
        if not strategy_names:
            strategy_names = [f"Strategy_{i}" for i in range(num_strategies)]
            
        return ModelConfiguration(
            state_dim=state_dim,
            action_dim=action_dim,
            num_strategies=num_strategies,
            latent_dim=latent_dim,
            strategy_names=strategy_names,
            timestamp=datetime.now().isoformat()
        )
    
    def _configs_equal(self, config1: ModelConfiguration, config2: ModelConfiguration) -> bool:
        """比較兩個配置是否相等"""
        return (config1.state_dim == config2.state_dim and
                config1.action_dim == config2.action_dim and
                config1.num_strategies == config2.num_strategies)
    
    def _get_change_type(self, old_config: ModelConfiguration, new_config: ModelConfiguration) -> str:
        """確定配置變化類型"""
        if old_config.num_strategies != new_config.num_strategies:
            return "strategy_count"
        elif old_config.state_dim != new_config.state_dim:
            return "state_dim"
        elif old_config.action_dim != new_config.action_dim:
            return "action_dim"
        else:
            return "other"
    
    def adapt_to_strategy_layer(self, strategy_layer: nn.Module) -> bool:
        """
        自適應到策略層配置
        
        Args:
            strategy_layer: 量子策略層
            
        Returns:
            bool: 適應是否成功
        """
        try:
            # 檢測配置
            config = self.detect_model_configuration(strategy_layer)
            
            # 適應編碼器維度
            adaptation_success = self.strategy_encoder.adapt_input_dimension(
                config.state_dim, preserve_weights=True
            )
            
            # 更新統計
            self.total_adaptations += 1
            if adaptation_success:
                self.adaptation_success_rate = (
                    (self.adaptation_success_rate * (self.total_adaptations - 1) + 1.0) / 
                    self.total_adaptations
                )
            else:
                self.adaptation_success_rate = (
                    (self.adaptation_success_rate * (self.total_adaptations - 1)) / 
                    self.total_adaptations
                )
            
            logger.info(f"自適應完成 - 成功率: {self.adaptation_success_rate:.2%}")
            return adaptation_success
            
        except Exception as e:
            logger.error(f"自適應失敗: {e}")
            return False
    
    def forward(self, state: torch.Tensor, 
                strategy_info: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        前向傳播
        
        Args:
            state: 輸入狀態
            strategy_info: 策略信息（可選）
            
        Returns:
            Tuple[動作輸出, 詳細信息]
        """
        # 編碼狀態
        encoded_state = self.strategy_encoder(state)
        
        # 元學習控制
        meta_action = self.meta_controller(encoded_state)
        
        # 策略重要性評估
        importance_score = self.strategy_importance(encoded_state)
        
        # 構建詳細信息
        info = {
            'encoded_state': encoded_state,
            'importance_score': importance_score,
            'current_config': self.current_config,
            'adaptation_success_rate': self.adaptation_success_rate.item(),
            'total_adaptations': self.total_adaptations.item()
        }
        
        return meta_action, info
    
    def save_configuration_history(self, filepath: str):
        """保存配置歷史"""
        try:
            history_data = {
                'config_history': [
                    {
                        'state_dim': cfg.state_dim,
                        'action_dim': cfg.action_dim,
                        'num_strategies': cfg.num_strategies,
                        'strategy_names': cfg.strategy_names,
                        'timestamp': cfg.timestamp,
                        'version': cfg.version
                    }
                    for cfg in self.config_history
                ],
                'dimension_changes': [
                    {
                        'old_config': {
                            'state_dim': change.old_config.state_dim,
                            'num_strategies': change.old_config.num_strategies
                        },
                        'new_config': {
                            'state_dim': change.new_config.state_dim,
                            'num_strategies': change.new_config.num_strategies
                        },
                        'change_type': change.change_type,
                        'timestamp': change.timestamp,
                        'migration_success': change.migration_success
                    }
                    for change in self.dimension_changes
                ],
                'encoder_dimension_history': self.strategy_encoder.get_dimension_history(),
                'adaptation_success_rate': self.adaptation_success_rate.item(),
                'total_adaptations': self.total_adaptations.item()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"配置歷史已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存配置歷史失敗: {e}")
    
    def load_configuration_history(self, filepath: str):
        """載入配置歷史"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"配置歷史文件不存在: {filepath}")
                return
                
            with open(filepath, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            # 恢復統計信息
            if 'adaptation_success_rate' in history_data:
                self.adaptation_success_rate.data.fill_(history_data['adaptation_success_rate'])
            if 'total_adaptations' in history_data:
                self.total_adaptations.data.fill_(history_data['total_adaptations'])
                
            logger.info(f"配置歷史已載入: {len(history_data.get('config_history', []))} 個配置記錄")
            
        except Exception as e:
            logger.error(f"載入配置歷史失敗: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """獲取系統狀態"""
        return {
            'current_config': self.current_config.__dict__ if self.current_config else None,
            'total_config_changes': len(self.config_history),
            'total_dimension_changes': len(self.dimension_changes),
            'adaptation_success_rate': self.adaptation_success_rate.item(),
            'total_adaptations': self.total_adaptations.item(),
            'encoder_current_dim': self.strategy_encoder.current_input_dim,
            'encoder_dimension_changes': len(self.strategy_encoder.dimension_history),
            'supported_detection_methods': self.config_detection_methods
        }


# 測試函數
def test_adaptive_meta_learning_system():
    """測試自適應元學習系統"""
    logger.info("開始測試自適應元學習系統...")
    
    try:
        # 初始化系統
        initial_state_dim = 64
        action_dim = 10
        
        meta_system = MetaLearningSystem(
            initial_state_dim=initial_state_dim,
            action_dim=action_dim,
            meta_learning_dim=256
        )
        
        logger.info(f"元學習系統參數數量: {sum(p.numel() for p in meta_system.parameters()):,}")
        
        # 測試基本前向傳播
        test_state = torch.randn(4, initial_state_dim)
        
        with torch.no_grad():
            output, info = meta_system(test_state)
            
        logger.info(f"基本前向傳播測試通過")
        logger.info(f"輸入形狀: {test_state.shape}")
        logger.info(f"輸出形狀: {output.shape}")
        logger.info(f"重要性分數形狀: {info['importance_score'].shape}")
        
        # 測試維度適應
        logger.info("測試維度適應...")
        
        # 模擬維度變化：64 -> 96
        new_state_dim = 96
        adaptation_success = meta_system.strategy_encoder.adapt_input_dimension(new_state_dim)
        
        if adaptation_success:
            logger.info(f"維度適應成功: {initial_state_dim} -> {new_state_dim}")
            
            # 測試新維度的前向傳播
            new_test_state = torch.randn(4, new_state_dim)
            with torch.no_grad():
                new_output, new_info = meta_system(new_test_state)
                
            logger.info(f"新維度前向傳播測試通過: {new_test_state.shape} -> {new_output.shape}")
        else:
            logger.error("維度適應失敗")
        
        # 測試配置檢測（使用模擬的策略層）
        logger.info("測試配置檢測...")
        
        # 創建模擬策略層
        class MockStrategyLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.state_dim = 96
                self.action_dim = 10
                self.num_strategies = 20
                self.base_strategies = nn.ModuleList([
                    nn.Linear(96, 10) for _ in range(20)
                ])
                
            def forward(self, state, volatility):
                return torch.randn(state.size(0), 10), {}
        
        mock_strategy_layer = MockStrategyLayer()
        
        # 給策略添加get_strategy_name方法
        for i, strategy in enumerate(mock_strategy_layer.base_strategies):
            strategy.get_strategy_name = lambda idx=i: f"MockStrategy_{idx}"
        
        # 檢測配置
        detected_config = meta_system.detect_model_configuration(mock_strategy_layer)
        logger.info(f"檢測到的配置: 狀態維度={detected_config.state_dim}, "
                   f"策略數量={detected_config.num_strategies}")
        
        # 測試適應到策略層
        adaptation_success = meta_system.adapt_to_strategy_layer(mock_strategy_layer)
        logger.info(f"策略層適應成功: {adaptation_success}")
        
        # 測試系統狀態
        status = meta_system.get_system_status()
        logger.info(f"系統狀態:")
        for key, value in status.items():
            logger.info(f"  {key}: {value}")
        
        # 測試配置歷史保存和載入
        config_file = "test_meta_learning_config.json"
        meta_system.save_configuration_history(config_file)
        
        # 創建新系統並載入歷史
        new_meta_system = MetaLearningSystem(
            initial_state_dim=initial_state_dim,
            action_dim=action_dim
        )
        new_meta_system.load_configuration_history(config_file)
        
        # 清理測試文件
        if os.path.exists(config_file):
            os.remove(config_file)
            
        # 測試梯度計算
        logger.info("測試梯度計算...")
        meta_system.train()
        test_state = torch.randn(4, meta_system.strategy_encoder.current_input_dim, requires_grad=True)
        output, _ = meta_system(test_state)
        loss = output.abs().mean()
        loss.backward()
        
        # 檢查梯度
        total_grad_norm = 0
        grad_count = 0
        for param in meta_system.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2) ** 2
                grad_count += 1
        
        total_grad_norm = total_grad_norm ** 0.5
        logger.info(f"梯度計算測試通過 - 總梯度範數: {total_grad_norm:.6f}, 有梯度參數數: {grad_count}")
        
        logger.info("自適應元學習系統測試完成！")
        return True
        
    except Exception as e:
        logger.error(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 運行測試
    success = test_adaptive_meta_learning_system()
    if success:
        print("\n✅ 自適應元學習系統測試成功！")
    else:
        print("\n❌ 自適應元學習系統測試失敗！")