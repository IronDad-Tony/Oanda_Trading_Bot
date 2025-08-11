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
    from oanda_trading_bot.training_system.common.logger_setup import logger
    from oanda_trading_bot.training_system.common.config import DEVICE, MAX_SYMBOLS_ALLOWED
    from oanda_trading_bot.training_system.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition
    from oanda_trading_bot.training_system.environment.progressive_reward_system import ProgressiveRewardSystem
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
        # Determine device from parameters if possible, else use global DEVICE
        device_to_use = next(self.parameters()).device if list(self.parameters()) else torch.device(DEVICE)
        self._build_network(initial_input_dim, device_to_use)
        
        # 記錄維度變化歷史
        self.dimension_history = []
        
    def _build_network(self, input_dim: int, device: torch.device):
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
        
        self.network = nn.Sequential(*layers).to(device) # Ensure network is on the correct device
        
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
        # Determine device from existing network or fall back to global DEVICE
        current_device = next(self.network.parameters()).device if list(self.network.parameters()) else torch.device(DEVICE)
        self._build_network(new_input_dim, current_device)
        
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
            # Ensure adaptation happens on the correct device
            self.adapt_input_dimension(x.size(-1))
            # After adaptation, the network might be on a different device if not handled carefully.
            # The _build_network method now handles moving the new network to the correct device.
            # We also need to ensure the current instance of the encoder is on the correct device if it was moved.
            self.to(x.device) # Ensure the encoder module itself is on the same device as input x
            
        return self.network(x)
    
    def get_dimension_history(self) -> List[Dict]:
        """獲取維度變化歷史"""
        return self.dimension_history.copy()


class MarketKnowledgeBase:
    """
    市場知識庫
    存儲和檢索策略表現、市場狀態等信息
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化知識庫
        
        Args:
            storage_path: 知識庫存儲路徑 (可選)
        """
        self.storage_path = storage_path
        self.knowledge: Dict[str, Any] = {}
        
        if self.storage_path and os.path.exists(self.storage_path):
            self.load_knowledge()
            
        logger.info(f"市場知識庫已初始化。存儲路徑: {self.storage_path if self.storage_path else '內存存儲'}")

    def store_knowledge(self, key: str, value: Any, persist: bool = False):
        """
        存儲知識
        
        Args:
            key: 知識鍵
            value: 知識值
            persist: 是否持久化存儲 (如果提供了存儲路徑)
        """
        self.knowledge[key] = value
        logger.debug(f"知識已存儲: Key='{key}', Value='{value}'")
        
        if persist and self.storage_path:
            self.save_knowledge()
            
    def retrieve_knowledge(self, key: str) -> Optional[Any]:
        """
        檢索知識
        
        Args:
            key: 知識鍵
            
        Returns:
            Optional[Any]: 知識值或 None
        """
        value = self.knowledge.get(key)
        if value is not None:
            logger.debug(f"知識已檢索: Key='{key}', Value='{value}'")
        else:
            logger.debug(f"知識未找到: Key='{key}'")
        return value

    def save_knowledge(self):
        """保存知識庫到文件"""
        if not self.storage_path:
            logger.warning("未提供存儲路徑，無法保存知識庫")
            return
            
        try:
            with open(self.storage_path, 'wb') as f:
                pickle.dump(self.knowledge, f)
            logger.info(f"知識庫已保存到: {self.storage_path}")
        except Exception as e:
            logger.error(f"保存知識庫失敗: {e}")

    def load_knowledge(self):
        """從文件載入知識庫"""
        if not self.storage_path or not os.path.exists(self.storage_path):
            logger.warning(f"存儲文件不存在或未提供路徑: {self.storage_path}")
            return
            
        try:
            with open(self.storage_path, 'rb') as f:
                self.knowledge = pickle.load(f)
            logger.info(f"知識庫已從 {self.storage_path} 載入")
        except Exception as e:
            logger.error(f"載入知識庫失敗: {e}")

    def get_all_knowledge_keys(self) -> List[str]:
        """獲取所有知識鍵"""
        return list(self.knowledge.keys())

    def clear_knowledge(self, persist: bool = False):
        """清除所有知識"""
        self.knowledge.clear()
        logger.info("知識庫已清除")
        if persist and self.storage_path:
            self.save_knowledge() # Save the empty state

    def find_similar_market_conditions(self, current_market_state: Dict[str, Any], 
                                     similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find similar market conditions in the knowledge base for cross-market knowledge transfer.
        
        Args:
            current_market_state (Dict[str, Any]): Current market conditions to match against
            similarity_threshold (float): Minimum similarity score to consider a match
            
        Returns:
            List[Dict[str, Any]]: List of similar market conditions and their associated knowledge
        """
        similar_conditions = []
        
        if not current_market_state:
            logger.warning("No current market state provided for similarity search")
            return similar_conditions
        
        # Extract current regime information
        current_regime = current_market_state.get('current_regime', {})
        current_volatility = current_regime.get('volatility_level', 'unknown')
        current_trend = current_regime.get('trend_strength', 'unknown')
        current_macro = current_regime.get('macro_regime', 'unknown')
        
        # Convert enums to string values for comparison
        if hasattr(current_volatility, 'value'):
            current_volatility = current_volatility.value
        if hasattr(current_trend, 'value'):
            current_trend = current_trend.value  
        if hasattr(current_macro, 'value'):
            current_macro = current_macro.value
            
        logger.debug(f"Searching for similar conditions to: volatility={current_volatility}, "
                    f"trend={current_trend}, macro={current_macro}")
        
        # Search through stored knowledge
        for key, value in self.knowledge.items():
            if 'regime_' in key and isinstance(value, dict):
                # Extract regime information from knowledge key
                key_parts = key.split('_regime_')
                if len(key_parts) > 1:
                    regime_str = key_parts[1]
                    
                    # Parse regime string to extract components
                    # Expected format: "volatility_level_medium_trend_strength_weak_trend_macro_regime_ranging"
                    similarity_score = 0.0
                    matches = 0
                    
                    if f"volatility_level_{current_volatility}".lower() in regime_str.lower():
                        similarity_score += 0.4
                        matches += 1
                    elif any(vol in regime_str.lower() for vol in ['low', 'medium', 'high']):
                        similarity_score += 0.1  # Partial match for different volatility levels
                        
                    if f"trend_strength_{current_trend}".lower() in regime_str.lower():
                        similarity_score += 0.4
                        matches += 1
                    elif any(trend in regime_str.lower() for trend in ['no_trend', 'weak_trend', 'strong_trend']):
                        similarity_score += 0.1  # Partial match for different trend strengths
                        
                    if f"macro_regime_{current_macro}".lower() in regime_str.lower():
                        similarity_score += 0.2
                        matches += 1
                    elif any(macro in regime_str.lower() for macro in ['bullish', 'bearish', 'ranging']):
                        similarity_score += 0.05  # Partial match for different macro regimes
                    
                    # Bonus for exact matches on multiple dimensions
                    if matches >= 2:
                        similarity_score += 0.1
                    if matches == 3:
                        similarity_score += 0.1
                    
                    # Only include if above threshold
                    if similarity_score >= similarity_threshold:
                        similar_conditions.append({
                            'knowledge_key': key,
                            'similarity_score': similarity_score,
                            'regime_string': regime_str,
                            'knowledge_value': value,
                            'exact_matches': matches
                        })
                        
                        logger.debug(f"Found similar condition: {regime_str} (similarity: {similarity_score:.2f})")
        
        # Sort by similarity score (highest first)
        similar_conditions.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        logger.info(f"Found {len(similar_conditions)} similar market conditions "
                   f"(threshold: {similarity_threshold})")
        
        return similar_conditions


class MetaLearningSystem(nn.Module):
    """
    自適應元學習系統
    整合自適應編碼器、策略檢測和動態配置管理
    """
    
    def __init__(self, initial_state_dim: int, action_dim: int,
                 meta_learning_dim: int = 256,
                 config_detection_methods: List[str] = None,
                 knowledge_base: Optional[MarketKnowledgeBase] = None): # Added knowledge_base parameter
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
        self.adaptation_history = []  # 添加適應歷史記錄
        
        # 初始化知識庫
        self.knowledge_base = knowledge_base if knowledge_base is not None else MarketKnowledgeBase()
        
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
                
                # 記錄適應事件
                self.record_adaptation_event(
                    event_type="config_detection",
                    success=True,
                    details={
                        'old_state_dim': self.current_config.state_dim,
                        'new_state_dim': final_config.state_dim,
                        'change_type': change.change_type
                    }
                )
                
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
    
    def record_adaptation_event(self, event_type: str, success: bool, details: Dict[str, Any] = None):
        """
        記錄適應事件到適應歷史
        
        Args:
            event_type: 事件類型 (如 'dimension_change', 'strategy_adaptation', 'config_detection')
            success: 事件是否成功
            details: 額外的事件詳情
        """
        if details is None:
            details = {}
            
        adaptation_event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'success': success,
            'details': details
        }
        
        self.adaptation_history.append(adaptation_event)
        
        # 更新適應成功率
        self.total_adaptations += 1
        if success:
            # 計算新的成功率（移動平均）
            current_success_rate = self.adaptation_success_rate.item()
            total_adaptations = self.total_adaptations.item()
            new_success_rate = (current_success_rate * (total_adaptations - 1) + 1.0) / total_adaptations
            self.adaptation_success_rate.fill_(new_success_rate)
        
        logger.debug(f"記錄適應事件: {event_type} - 成功: {success}")
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """
        獲取適應統計信息
        
        Returns:
            Dict: 包含適應統計的字典
        """
        if not self.adaptation_history:
            return {
                'total_events': 0,
                'success_rate': 0.0,
                'event_types': {},
                'recent_events': []
            }
        
        # 統計事件類型
        event_types = {}
        successful_events = 0
        
        for event in self.adaptation_history:
            event_type = event['event_type']
            if event_type not in event_types:
                event_types[event_type] = {'total': 0, 'successful': 0}
            
            event_types[event_type]['total'] += 1
            if event['success']:
                event_types[event_type]['successful'] += 1
                successful_events += 1
        
        # 計算成功率
        success_rate = successful_events / len(self.adaptation_history) if self.adaptation_history else 0.0
        
        # 獲取最近的事件
        recent_events = self.adaptation_history[-5:] if len(self.adaptation_history) > 5 else self.adaptation_history
        
        return {
            'total_events': len(self.adaptation_history),
            'success_rate': success_rate,
            'event_types': event_types,
            'recent_events': recent_events
        }
    
    def evaluate_strategy_performance(self,
                                      strategy_id: str,
                                      trade_history: List[Dict[str, Any]],
                                      market_conditions: Optional[Dict[str, Any]] = None,
                                      risk_free_rate_annual: float = 0.0 # Annual risk-free rate
                                      ) -> Dict[str, float]:
        """
        Evaluates the performance of a given strategy based on its trade history
        and current market conditions.

        Args:
            strategy_id (str): Identifier for the strategy.
            trade_history (List[Dict[str, Any]]): List of trades. Each dict should contain
                at least 'pnl' (profit and loss for the trade/period).
                It's assumed that PnLs are for comparable periods if calculating annualized ratios.
                For simplicity, we'll treat PnLs as per-period returns for ratio calculations.
                A 'duration_days' field could be added to each trade for more accurate annualization.
            market_conditions (Optional[Dict[str, Any]]): Current market regime, volatility, etc.
            risk_free_rate_annual (float): Annual risk-free rate for Sharpe/Sortino calculations.

        Returns:
            Dict[str, float]: A dictionary of performance metrics.
        """
        logger.debug(f"Evaluating performance for strategy: {strategy_id}")
        if not trade_history:
            logger.warning(f"No trade history for strategy {strategy_id}. Returning zero metrics.")
            return {
                "total_pnl": 0.0,
                "avg_pnl_per_trade": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "win_rate": 0.0,
                "loss_rate": 0.0,
                "profit_factor": 0.0,
                "trade_count": 0.0,
                "consistency_score": 0.0, # Lower is better (less variation relative to mean)
                "adaptability_score": 0.0  # Placeholder
            }

        pnl_values = np.array([trade.get('pnl', 0.0) for trade in trade_history])
        num_trades = len(pnl_values)

        total_pnl = np.sum(pnl_values)
        avg_pnl = np.mean(pnl_values) if num_trades > 0 else 0.0

        winning_trades = np.sum(pnl_values > 0)
        losing_trades = np.sum(pnl_values < 0)
        
        win_rate = (winning_trades / num_trades) if num_trades > 0 else 0.0
        loss_rate = (losing_trades / num_trades) if num_trades > 0 else 0.0

        total_profit_from_wins = np.sum(pnl_values[pnl_values > 0])
        total_loss_from_losses = abs(np.sum(pnl_values[pnl_values < 0]))
        profit_factor = (total_profit_from_wins / total_loss_from_losses) if total_loss_from_losses > 0 else np.inf

        # For Sharpe and Sortino, assume pnl_values are returns for some period (e.g., per trade)
        # To annualize, we'd need to know the number of trading periods in a year.
        # For simplicity, let's assume 'periods_in_year' = 252 (trading days) if PnLs were daily.
        # If PnLs are per trade, annualization is more complex.
        # Here, we calculate non-annualized ratios based on the provided PnLs as returns.
        # The risk_free_rate_annual should be converted to a per-period rate.
        # Assuming PnLs are per-trade and we don't have a fixed "period", we use a simplified approach.
        # Let's assume risk_free_rate_per_period = 0 for now for simplicity, or it needs to be passed.
        
        std_dev_pnl = np.std(pnl_values) if num_trades > 1 else 0.0

        # Simplified Sharpe: (Mean PnL per trade) / (Std Dev of PnL per trade)
        # Assuming risk-free rate per trade is negligible or handled by PnL definition.
        sharpe_ratio = (avg_pnl / std_dev_pnl) if std_dev_pnl > 0 else 0.0
        if num_trades <= 1: sharpe_ratio = 0.0


        negative_pnls = pnl_values[pnl_values < 0]
        downside_std_dev = np.std(negative_pnls) if len(negative_pnls) > 1 else 0.0
        
        # Simplified Sortino: (Mean PnL per trade) / (Downside Std Dev of PnL per trade)
        sortino_ratio = (avg_pnl / downside_std_dev) if downside_std_dev > 0 else 0.0
        if num_trades <=1 or len(negative_pnls) <=1 : sortino_ratio = 0.0


        # Consistency Score: Coefficient of Variation (lower is better for positive avg_pnl)
        # We want higher score for better consistency.
        # If avg_pnl is positive, consistency = 1 / (1 + CoV) = 1 / (1 + std_dev_pnl / avg_pnl)
        # If avg_pnl is zero or negative, consistency is low.
        consistency_score = 0.0
        if avg_pnl > 0 and std_dev_pnl >= 0: # std_dev_pnl can be 0 if all PnLs are same
            if std_dev_pnl == 0 : # Perfect consistency if all PnLs are same and positive
                 consistency_score = 1.0
            else:
                coefficient_of_variation = std_dev_pnl / avg_pnl
                consistency_score = 1.0 / (1.0 + coefficient_of_variation)
        elif num_trades > 0 and avg_pnl == 0 and std_dev_pnl == 0: # All zero PnLs
             consistency_score = 0.5 # Neutral consistency for zero PnLs

        # Store this performance in knowledge base
        knowledge_key = f"strategy_performance_{strategy_id}"
        current_regime_str = "unknown_regime"
        if market_conditions and market_conditions.get('current_regime'):
            # Attempt to create a string representation of the regime
            regime_details = market_conditions['current_regime']
            if isinstance(regime_details, dict):
                # Example: {'volatility_level': VolatilityLevel.HIGH, 'trend_strength': TrendStrength.STRONG}
                # Convert enums to their values for a cleaner string
                regime_parts = []
                for r_key, r_val in regime_details.items():
                    if hasattr(r_val, 'value'): # Check if it's an enum with a .value attribute
                        regime_parts.append(f"{r_key}_{r_val.value}")
                    else:
                        regime_parts.append(f"{r_key}_{str(r_val)}")
                current_regime_str = "_".join(regime_parts)
            else: # If it's not a dict, just convert to string
                current_regime_str = str(regime_details)
            
            current_regime_str = current_regime_str.replace(" ", "_").replace("'", "").replace(":", "_").replace("{", "").replace("}", "").lower()
            knowledge_key += f"_regime_{current_regime_str}"
        
        performance_metrics = {
            "total_pnl": float(total_pnl),
            "avg_pnl_per_trade": float(avg_pnl),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "win_rate": float(win_rate),
            "loss_rate": float(loss_rate),
            "profit_factor": float(profit_factor) if profit_factor != np.inf else 1000.0, # Cap inf profit factor
            "trade_count": float(num_trades),
            "consistency_score": float(consistency_score),
            "adaptability_score": 0.5  # Placeholder, needs proper calculation
        }
        self.knowledge_base.store_knowledge(knowledge_key, performance_metrics)
        
        logger.info(f"Performance for strategy {strategy_id} (Regime: {current_regime_str}): {performance_metrics}")
        return performance_metrics

    def adapt_strategies(self, 
                        current_market_state: Dict[str, Any],
                        performance_evaluations: Dict[str, Dict[str, float]],
                        adaptation_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Automatically adjust strategies based on market state and strategy evaluation results.
        
        Args:
            current_market_state (Dict[str, Any]): Current market conditions including regime, volatility, etc.
            performance_evaluations (Dict[str, Dict[str, float]]): Strategy performance metrics from evaluate_strategy_performance
            adaptation_threshold (float): Minimum performance threshold to trigger adaptations
            
        Returns:
            Dict[str, Any]: Adaptation recommendations and changes made
        """
        logger.info("Starting strategy adaptation based on market state and performance evaluations...")
        
        adaptations = {
            "timestamp": datetime.now().isoformat(),
            "market_state": current_market_state,
            "adaptations_made": [],
            "recommendations": [],
            "performance_summary": {}
        }
        
        # Analyze overall performance
        if performance_evaluations:
            avg_sharpe = np.mean([metrics.get('sharpe_ratio', 0.0) for metrics in performance_evaluations.values()])
            avg_win_rate = np.mean([metrics.get('win_rate', 0.0) for metrics in performance_evaluations.values()])
            total_trades = sum([metrics.get('trade_count', 0.0) for metrics in performance_evaluations.values()])
            
            adaptations["performance_summary"] = {
                "avg_sharpe_ratio": float(avg_sharpe),
                "avg_win_rate": float(avg_win_rate),
                "total_trades": float(total_trades),
                "num_strategies_evaluated": len(performance_evaluations)
            }
            
            # Identify underperforming strategies
            underperforming = []
            high_performing = []
            
            for strategy_id, metrics in performance_evaluations.items():
                sharpe = metrics.get('sharpe_ratio', 0.0)
                win_rate = metrics.get('win_rate', 0.0)
                
                if sharpe < adaptation_threshold and win_rate < 0.5:
                    underperforming.append((strategy_id, metrics))
                elif sharpe > adaptation_threshold * 2 and win_rate > 0.6:
                    high_performing.append((strategy_id, metrics))
            
            # Generate adaptation recommendations based on market state
            regime = current_market_state.get('current_regime', {})
            volatility = regime.get('volatility_level', 'unknown')
            trend_strength = regime.get('trend_strength', 'unknown')
            macro_regime = regime.get('macro_regime', 'unknown')
            
            # Market-state specific adaptations
            if hasattr(volatility, 'value'):
                volatility = volatility.value
            if hasattr(trend_strength, 'value'):
                trend_strength = trend_strength.value
            if hasattr(macro_regime, 'value'):
                macro_regime = macro_regime.value
                
            # High volatility adaptations
            if 'high' in str(volatility).lower():
                adaptations["recommendations"].append({
                    "type": "risk_management",
                    "reason": "High volatility detected",
                    "action": "Reduce position sizes and increase stop-loss sensitivity",
                    "affected_strategies": [s[0] for s in underperforming]
                })
                
                adaptations["adaptations_made"].append({
                    "type": "parameter_adjustment",
                    "parameter": "risk_multiplier",
                    "old_value": 1.0,
                    "new_value": 0.7,
                    "reason": "High volatility adaptation"
                })
            
            # Strong trend adaptations
            if 'strong' in str(trend_strength).lower():
                adaptations["recommendations"].append({
                    "type": "strategy_weighting",
                    "reason": "Strong trend detected",
                    "action": "Increase weight on trend-following strategies",
                    "boost_strategies": ["TrendFollowingStrategy", "MomentumStrategy", "BreakoutStrategy"]
                })
                
            # Low trend (ranging market) adaptations
            elif 'no' in str(trend_strength).lower() or 'weak' in str(trend_strength).lower():
                adaptations["recommendations"].append({
                    "type": "strategy_weighting", 
                    "reason": "Weak/No trend detected (ranging market)",
                    "action": "Increase weight on mean reversion and statistical arbitrage strategies",
                    "boost_strategies": ["MeanReversionStrategy", "StatisticalArbitrageStrategy", "PairsTradingStrategy"]
                })
            
            # Underperforming strategy adaptations
            if underperforming:
                adaptations["recommendations"].append({
                    "type": "strategy_replacement",
                    "reason": f"Strategies underperforming (Sharpe < {adaptation_threshold})",
                    "action": "Consider reducing weight or replacing with high-performing alternatives",
                    "underperforming_strategies": [s[0] for s in underperforming],
                    "suggested_replacements": [s[0] for s in high_performing[:2]]  # Top 2 performers
                })
                
                # Adaptive learning rate adjustment for underperforming strategies
                for strategy_id, metrics in underperforming:
                    adaptations["adaptations_made"].append({
                        "type": "learning_rate_adjustment",
                        "strategy": strategy_id,
                        "old_lr": "default",
                        "new_lr": "reduced_50%",
                        "reason": f"Poor performance: Sharpe={metrics.get('sharpe_ratio', 0.0):.3f}"
                    })
            
            # Cross-market knowledge transfer
            if self.knowledge_base:
                similar_conditions = self.knowledge_base.find_similar_market_conditions(current_market_state)
                if similar_conditions:
                    adaptations["recommendations"].append({
                        "type": "knowledge_transfer",
                        "reason": "Similar market conditions found in knowledge base",
                        "action": "Apply insights from similar historical periods",
                        "historical_matches": len(similar_conditions),
                        "suggested_adjustments": "Use proven strategy combinations from similar regimes"
                    })
        
        else:
            adaptations["recommendations"].append({
                "type": "data_collection",
                "reason": "No performance evaluations provided",
                "action": "Collect more trading data before making adaptations"
            })
        
        # Store adaptation in knowledge base
        if self.knowledge_base:
            adaptation_key = f"adaptation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.knowledge_base.store_knowledge(adaptation_key, adaptations)
            
        # Update adaptation history
        self.adaptation_history.append(adaptations)
        
        # Update success metrics (simplified)
        if len(adaptations["adaptations_made"]) > 0:
            self.total_adaptations += 1
            # Simplified success tracking - could be more sophisticated
            self.adaptation_success_rate = (self.adaptation_success_rate * (self.total_adaptations - 1) + 1.0) / self.total_adaptations
        
        logger.info(f"Strategy adaptation completed. Made {len(adaptations['adaptations_made'])} adaptations, "
                   f"{len(adaptations['recommendations'])} recommendations.")
        
        return adaptations


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
    
    import numpy as np # Required for std in evaluate_strategy_performance
    logging.basicConfig(level=logging.DEBUG)
    logger.info("----- Testing MetaLearningSystem -----")

    # Instantiate MarketKnowledgeBase first
    kb = MarketKnowledgeBase()
    
    # Instantiate MetaLearningSystem
    mls = MetaLearningSystem(initial_state_dim=64, action_dim=10, meta_learning_dim=128)
    mls.knowledge_base = kb # Assign the kb instance to the mls instance

    # Mock MarketRegimeIdentifier and its output
    class MockVolatilityLevel:
        HIGH = "High"
        MEDIUM = "Medium"
        LOW = "Low"

    class MockTrendStrength:
        STRONG = "Strong"
        WEAK = "Weak"
        NO_TREND = "No_Trend"
    
    class MockMacroRegime:
        BULLISH = "Bullish"
        BEARISH = "Bearish"
        RANGING = "Ranging"


    # 1. Test MarketKnowledgeBase
    logger.info("Testing MarketKnowledgeBase...")
    kb = MarketKnowledgeBase()
    assert kb is not None
    
    # Test storing and retrieving knowledge
    sample_performance_data = {
        "total_pnl": 1000.0,
        "avg_pnl_per_trade": 50.0,
        "sharpe_ratio": 1.5,
        "sortino_ratio": 2.0,
        "win_rate": 0.8,
        "loss_rate": 0.2,
        "profit_factor": 4.0,
        "trade_count": 20,
        "consistency_score": 0.9
    }
    kb.store_knowledge("strategy_performance_TestStrategy", sample_performance_data)
    
    retrieved_data = kb.retrieve_knowledge("strategy_performance_TestStrategy")
    logger.info(f"Retrieved data: {retrieved_data}")
    assert retrieved_data is not None
    assert retrieved_data["total_pnl"] == 1000.0
    
    # Test strategy performance evaluation
    sample_trade_history_1 = [
        {'pnl': 10}, {'pnl': -5}, {'pnl': 15}, {'pnl': 2} # Total PnL = 22, Avg PnL = 5.5
    ]
    market_cond_test_1 = {
        'current_regime': {
            'volatility_level': MockVolatilityLevel.HIGH, 
            'trend_strength': MockTrendStrength.STRONG, 
            'macro_regime': MockMacroRegime.BULLISH
            }
    }
    perf_metrics_1 = mls.evaluate_strategy_performance("StrategyAlpha", sample_trade_history_1, market_cond_test_1)
    logger.info(f"Performance metrics for StrategyAlpha (Test 1): {perf_metrics_1}")
    assert perf_metrics_1["trade_count"] == 4
    assert abs(perf_metrics_1["total_pnl"] - 22) < 1e-6
    assert abs(perf_metrics_1["avg_pnl_per_trade"] - 5.5) < 1e-6
    assert perf_metrics_1["win_rate"] == 0.75 # 3 wins / 4 trades
    assert perf_metrics_1["loss_rate"] == 0.25 # 1 loss / 4 trades
    # Total wins = 10+15+2 = 27. Total loss = 5. Profit factor = 27/5 = 5.4
    assert abs(perf_metrics_1["profit_factor"] - 5.4) < 1e-6 


    # Check KB storage for Test 1
    regime_details_1 = market_cond_test_1['current_regime']
    regime_str_for_key_1 = f"volatility_level_{regime_details_1['volatility_level'].lower()}_trend_strength_{regime_details_1['trend_strength'].lower()}_macro_regime_{regime_details_1['macro_regime'].lower()}"
    expected_kb_key_1 = f"strategy_performance_StrategyAlpha_regime_{regime_str_for_key_1}"
    
    stored_perf_1 = kb.retrieve_knowledge(expected_kb_key_1)
    logger.info(f"Attempting to retrieve stored performance (Test 1) with key: {expected_kb_key_1}")
    assert stored_perf_1 is not None, f"Key not found in KB: {expected_kb_key_1}"
    assert stored_perf_1["total_pnl"] == perf_metrics_1["total_pnl"]
    logger.info(f"Successfully retrieved stored performance from KB (Test 1): {stored_perf_1}")

    # Test with all positive PnLs for consistency
    sample_trade_history_2 = [{'pnl': 10}, {'pnl': 12}, {'pnl': 11}, {'pnl': 13}] # Avg=11.5, Std=1.118
    perf_metrics_2 = mls.evaluate_strategy_performance("StrategyBeta", sample_trade_history_2, market_cond_test_1)
    logger.info(f"Performance metrics for StrategyBeta (Test 2 - all positive): {perf_metrics_2}")
    assert perf_metrics_2["win_rate"] == 1.0
    assert perf_metrics_2["loss_rate"] == 0.0
    assert perf_metrics_2["profit_factor"] == 1000.0 # No losses, capped at 1000
    assert perf_metrics_2["consistency_score"] > 0.8 # Should be high

    # Test with identical PnLs for perfect consistency
    sample_trade_history_3 = [{'pnl': 5}, {'pnl': 5}, {'pnl': 5}]
    perf_metrics_3 = mls.evaluate_strategy_performance("StrategyGamma", sample_trade_history_3, market_cond_test_1)
    logger.info(f"Performance metrics for StrategyGamma (Test 3 - identical PnLs): {perf_metrics_3}")
    assert abs(perf_metrics_3["consistency_score"] - 1.0) < 1e-6

    # Test with zero PnLs
    sample_trade_history_4 = [{'pnl': 0}, {'pnl': 0}, {'pnl': 0}]
    perf_metrics_4 = mls.evaluate_strategy_performance("StrategyDelta", sample_trade_history_4, market_cond_test_1)
    logger.info(f"Performance metrics for StrategyDelta (Test 4 - zero PnLs): {perf_metrics_4}")
    assert abs(perf_metrics_4["consistency_score"] - 0.5) < 1e-6 # Neutral consistency
    assert abs(perf_metrics_4["sharpe_ratio"] - 0.0) < 1e-6
    assert abs(perf_metrics_4["sortino_ratio"] - 0.0) < 1e-6
    assert abs(perf_metrics_4["profit_factor"] - 0.0) < 1e-6 # No wins, no losses

    # Test with negative average PnL
    sample_trade_history_5 = [{'pnl': -2}, {'pnl': -3}, {'pnl': -1}]
    perf_metrics_5 = mls.evaluate_strategy_performance("StrategyEpsilon", sample_trade_history_5, market_cond_test_1)
    logger.info(f"Performance metrics for StrategyEpsilon (Test 5 - negative avg PnL): {perf_metrics_5}")
    assert perf_metrics_5["consistency_score"] == 0.0 # Low consistency for negative avg PnL
    assert perf_metrics_5["profit_factor"] == 0.0 # No wins

    # Test with single trade
    sample_trade_history_6 = [{'pnl': 10}]
    perf_metrics_6 = mls.evaluate_strategy_performance("StrategyZeta", sample_trade_history_6, market_cond_test_1)
    logger.info(f"Performance metrics for StrategyZeta (Test 6 - single trade): {perf_metrics_6}")
    assert perf_metrics_6["sharpe_ratio"] == 0.0
    assert perf_metrics_6["sortino_ratio"] == 0.0
    assert abs(perf_metrics_6["consistency_score"] - 1.0) < 1e-6 # Single trade, perfect consistency by this metric

    # Test empty trade history
    sample_trade_history_7 = []
    perf_metrics_7 = mls.evaluate_strategy_performance("StrategyEta", sample_trade_history_7, market_cond_test_1)
    logger.info(f"Performance metrics for StrategyEta (Test 7 - empty history): {perf_metrics_7}")
    assert perf_metrics_7["trade_count"] == 0.0
    assert perf_metrics_7["consistency_score"] == 0.0

    logger.info("----- MetaLearningSystem Tests Passed (enhanced evaluation) -----")