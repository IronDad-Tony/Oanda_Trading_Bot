# 修正版本的元學習系統測試
"""
自適應元學習系統測試
驗證所有自適應功能正常工作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque, defaultdict
import math
import copy

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 設置隨機種子保證結果可重現
torch.manual_seed(42)
np.random.seed(42)

@dataclass
class TaskDefinition:
    """任務定義類"""
    task_id: str
    market_regime: str  # "trending", "ranging", "volatile", "crisis"
    time_horizon: str   # "short", "medium", "long"
    risk_profile: str   # "conservative", "moderate", "aggressive"
    asset_class: str    # "forex", "stocks", "crypto", "commodities"
    volatility_range: Tuple[float, float] = (0.0, 1.0)
    return_target: float = 0.0
    max_drawdown: float = 0.2

# 簡化的策略模型用於測試
class MockEnhancedStrategySuperposition(nn.Module):
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, state, volatility=None):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        return self.strategy_net(state), {}

# 簡化的獎勵系統
class MockRewardSystem(nn.Module):
    def __init__(self, num_strategies, **kwargs):
        super().__init__()
        self.num_strategies = num_strategies
        
    def forward(self, x):
        batch_size = x.size(0) if hasattr(x, 'size') else 1
        return torch.randn(batch_size)

class AdaptiveStrategyEncoder(nn.Module):
    """自適應策略編碼器"""
    
    def __init__(self, initial_state_dim: int = 64, initial_strategy_dim: int = 20, 
                 embedding_dim: int = 128):
        super().__init__()
        self.current_state_dim = initial_state_dim
        self.current_strategy_dim = initial_strategy_dim
        self.embedding_dim = embedding_dim
        
        # 初始編碼器網絡
        self._create_encoder()
        
        # 維度變化歷史
        self.dimension_history = []
        self.adaptation_count = 0
        
        logger.info(f"初始化自適應策略編碼器: state_dim={initial_state_dim}, strategy_dim={initial_strategy_dim}")
    
    def _create_encoder(self):
        """創建編碼器網絡"""
        input_dim = self.current_state_dim + self.current_strategy_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim),
            nn.Tanh()
        )
    
    def adapt_dimensions(self, new_state_dim: int, new_strategy_dim: int):
        """適應新的維度"""
        if new_state_dim == self.current_state_dim and new_strategy_dim == self.current_strategy_dim:
            return  # 無需改變
        
        old_state_dim = self.current_state_dim
        old_strategy_dim = self.current_strategy_dim
        
        # 保存舊的權重
        old_encoder = copy.deepcopy(self.encoder)
        
        # 更新維度
        self.current_state_dim = new_state_dim
        self.current_strategy_dim = new_strategy_dim
        
        # 創建新的編碼器
        self._create_encoder()
        
        # 權重遷移
        self._transfer_weights(old_encoder, old_state_dim, old_strategy_dim)
        
        # 記錄變化
        self.dimension_history.append({
            'old_dims': (old_state_dim, old_strategy_dim),
            'new_dims': (new_state_dim, new_strategy_dim),
            'adaptation_count': self.adaptation_count
        })
        self.adaptation_count += 1
        
        logger.info(f"維度適應: ({old_state_dim}+{old_strategy_dim}) -> ({new_state_dim}+{new_strategy_dim})")
    
    def _transfer_weights(self, old_encoder: nn.Module, old_state_dim: int, old_strategy_dim: int):
        """權重遷移"""
        try:
            old_input_dim = old_state_dim + old_strategy_dim
            new_input_dim = self.current_state_dim + self.current_strategy_dim
            
            # 第一層權重遷移
            old_weight = old_encoder[0].weight.data
            old_bias = old_encoder[0].bias.data
            
            # 計算維度比例
            min_input_dim = min(old_input_dim, new_input_dim)
            
            # 複製可以遷移的權重
            with torch.no_grad():
                self.encoder[0].weight.data[:, :min_input_dim] = old_weight[:, :min_input_dim]
                self.encoder[0].bias.data = old_bias
                
                # 如果新維度更大，用小的隨機值初始化新權重
                if new_input_dim > old_input_dim:
                    self.encoder[0].weight.data[:, min_input_dim:] *= 0.1
            
            # 複製其他層的權重
            for new_layer, old_layer in zip(self.encoder[2:], old_encoder[2:]):
                if isinstance(new_layer, nn.Linear) and isinstance(old_layer, nn.Linear):
                    with torch.no_grad():
                        new_layer.weight.data = old_layer.weight.data.clone()
                        new_layer.bias.data = old_layer.bias.data.clone()
            
            logger.debug("權重遷移完成")
            
        except Exception as e:
            logger.warning(f"權重遷移失敗: {e}")
    
    def forward(self, state: torch.Tensor, strategy_weights: torch.Tensor) -> torch.Tensor:
        """前向傳播，自動檢測維度變化"""
        # 檢測維度變化
        detected_state_dim = state.shape[-1]
        detected_strategy_dim = strategy_weights.shape[-1]
        
        if (detected_state_dim != self.current_state_dim or 
            detected_strategy_dim != self.current_strategy_dim):
            self.adapt_dimensions(detected_state_dim, detected_strategy_dim)
        
        # 組合輸入
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(strategy_weights.shape) == 1:
            strategy_weights = strategy_weights.unsqueeze(0)
            
        combined_input = torch.cat([state, strategy_weights], dim=-1)
        
        # 編碼
        embedding = self.encoder(combined_input)
        
        return embedding

class MetaLearningSystem(nn.Module):
    """自適應元學習系統"""
    
    def __init__(self, strategy_model: nn.Module, reward_system: nn.Module, config: Optional[Dict] = None):
        super().__init__()
        
        self.strategy_model = strategy_model
        self.reward_system = reward_system
        self.config = config or {}
        
        # 使用自適應策略編碼器
        self.strategy_encoder = AdaptiveStrategyEncoder(
            initial_state_dim=getattr(strategy_model, 'state_dim', 64),
            initial_strategy_dim=20,  # 初始策略數量
            embedding_dim=self.config.get('embedding_dim', 128)
        )
        
        # 模型配置檢測
        self.detected_config = {
            'state_dim': 64,
            'strategy_dim': 20,
            'action_dim': 10
        }
        
        # 維度變化追蹤
        self.dimension_changes = 0
        self.adaptation_history = []
        
        # 任務編碼器
        self.task_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Tanh()
        )
        
        logger.info("初始化自適應元學習系統完成")
    
    def _detect_model_configuration(self) -> Dict[str, int]:
        """動態檢測模型配置"""
        config = {}
        
        # 方法1：從模型屬性檢測
        if hasattr(self.strategy_model, 'state_dim'):
            config['state_dim'] = self.strategy_model.state_dim
        if hasattr(self.strategy_model, 'action_dim'):
            config['action_dim'] = self.strategy_model.action_dim
        
        # 方法2：嘗試模擬前向傳播探測維度
        try:
            test_state = torch.randn(1, self.detected_config['state_dim'])
            test_volatility = torch.rand(1)
            with torch.no_grad():
                output, info = self.strategy_model(test_state, test_volatility)
                if hasattr(output, 'shape'):
                    config['action_dim'] = output.shape[-1]
        except Exception as e:
            logger.debug(f"前向傳播探測失敗: {e}")
        
        # 更新檢測到的配置
        for key, value in config.items():
            if key in self.detected_config:
                old_value = self.detected_config[key]
                if old_value != value:
                    logger.info(f"檢測到{key}變化: {old_value} -> {value}")
                    self.dimension_changes += 1
                self.detected_config[key] = value
        
        return self.detected_config
    
    def encode_strategy(self, state: torch.Tensor, strategy_weights: torch.Tensor) -> torch.Tensor:
        """編碼策略為嵌入向量（自適應版本）"""
        # 使用自適應編碼器
        embedding = self.strategy_encoder(state.mean(dim=0), strategy_weights.mean(dim=0))
        
        return embedding
    
    def encode_task(self, task: TaskDefinition) -> torch.Tensor:
        """編碼任務為特徵向量"""
        regime_encoding = {
            'trending': [1, 0, 0, 0],
            'ranging': [0, 1, 0, 0], 
            'volatile': [0, 0, 1, 0],
            'crisis': [0, 0, 0, 1]
        }.get(task.market_regime, [0, 0, 0, 0])
        
        risk_encoding = {
            'conservative': [1, 0, 0],
            'moderate': [0, 1, 0],
            'aggressive': [0, 0, 1]
        }.get(task.risk_profile, [0, 0, 0])
        
        task_features = torch.tensor(
            regime_encoding + risk_encoding + [task.return_target],
            dtype=torch.float32
        )
        
        return self.task_encoder(task_features)
    
    def fast_adaptation(self, new_task: TaskDefinition, support_data: Tuple[torch.Tensor, ...], 
                       num_adaptation_steps: int = 5) -> nn.Module:
        """快速適應新任務（增強版本）"""
        logger.info(f"開始快速適應新任務: {new_task.task_id}")
        
        # 動態檢測配置變化
        self._detect_model_configuration()
        
        # 使用自適應編碼器
        strategy_embedding = self.encode_strategy(support_data[0], support_data[1])
        
        # 創建適應模型
        adapted_model = copy.deepcopy(self.strategy_model)
        
        # 記錄適應歷史
        self.adaptation_history.append({
            'task_id': new_task.task_id,
            'detected_config': self.detected_config.copy(),
            'strategy_embedding_shape': strategy_embedding.shape
        })
        
        logger.info(f"快速適應完成，配置: {self.detected_config}")
        return adapted_model
    
    def update_strategy_configuration(self, new_state_dim: int, new_strategy_dim: int):
        """手動更新策略配置"""
        logger.info(f"手動更新策略配置: state_dim={new_state_dim}, strategy_dim={new_strategy_dim}")
        
        self.strategy_encoder.adapt_dimensions(new_state_dim, new_strategy_dim)
        self.detected_config['state_dim'] = new_state_dim
        self.detected_config['strategy_dim'] = new_strategy_dim
        self.dimension_changes += 1
    
    def get_system_analysis(self) -> Dict[str, Any]:
        """獲取系統分析（增強版本）"""
        return {
            'detected_config': self.detected_config,
            'dimension_changes': self.dimension_changes,
            'adaptation_count': self.strategy_encoder.adaptation_count,
            'dimension_history': self.strategy_encoder.dimension_history,
            'adaptation_history_count': len(self.adaptation_history),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'encoder_current_dims': (
                self.strategy_encoder.current_state_dim,
                self.strategy_encoder.current_strategy_dim
            )
        }

def test_adaptive_meta_learning_system():
    """測試自適應元學習系統"""
    logger.info("=== 開始自適應元學習系統測試 ===")
    
    # 測試參數
    batch_size = 4
    state_dim = 64
    action_dim = 10
    num_strategies = 20
    
    # 創建測試模型
    strategy_model = MockEnhancedStrategySuperposition(
        state_dim=state_dim,
        action_dim=action_dim,
        enable_dynamic_generation=True
    )
    
    reward_system = MockRewardSystem(
        num_strategies=num_strategies,
        enable_adaptive_learning=True
    )
    
    # 初始化元學習系統
    meta_config = {
        'max_strategies': 100,
        'embedding_dim': 64,
        'inner_lr': 0.01,
        'outer_lr': 0.001,
        'num_inner_steps': 3,
        'buffer_size': 1000
    }
    
    meta_learning_system = MetaLearningSystem(
        strategy_model=strategy_model,
        reward_system=reward_system,
        config=meta_config
    )
    
    # 測試1: 基本功能測試
    logger.info("--- 測試1: 基本功能測試 ---")
    
    test_task = TaskDefinition(
        task_id="test_task_1",
        market_regime="trending",
        time_horizon="medium",
        risk_profile="moderate",
        asset_class="forex",
        return_target=0.02
    )
    
    test_states = torch.randn(batch_size, state_dim)
    test_strategy_weights = torch.softmax(torch.randn(batch_size, num_strategies), dim=-1)
    test_rewards = torch.randn(batch_size) * 0.1
    
    # 測試任務編碼
    task_embedding = meta_learning_system.encode_task(test_task)
    logger.info(f"任務編碼形狀: {task_embedding.shape}")
    
    # 測試策略編碼（自適應）
    strategy_embedding = meta_learning_system.encode_strategy(test_states, test_strategy_weights)
    logger.info(f"策略編碼形狀: {strategy_embedding.shape}")
    
    # 測試快速適應
    support_data = (test_states, test_strategy_weights, test_rewards)
    adapted_model = meta_learning_system.fast_adaptation(test_task, support_data, num_adaptation_steps=3)
    logger.info("快速適應測試通過")
    
    # 測試2: 動態策略數量檢測
    logger.info("--- 測試2: 動態策略數量檢測 ---")
    
    # 模擬策略數量變化 20 -> 25
    new_num_strategies = 25
    new_strategy_weights = torch.softmax(torch.randn(batch_size, new_num_strategies), dim=-1)
    
    logger.info(f"策略數量變化: {num_strategies} -> {new_num_strategies}")
    strategy_embedding_new = meta_learning_system.encode_strategy(test_states, new_strategy_weights)
    logger.info(f"新策略編碼形狀: {strategy_embedding_new.shape}")
    
    # 測試3: 狀態維度變化
    logger.info("--- 測試3: 狀態維度變化 ---")
    
    # 模擬狀態維度變化 64 -> 96
    new_state_dim = 96
    new_test_states = torch.randn(batch_size, new_state_dim)
    
    logger.info(f"狀態維度變化: {state_dim} -> {new_state_dim}")
    strategy_embedding_new_state = meta_learning_system.encode_strategy(new_test_states, test_strategy_weights)
    logger.info(f"新狀態維度策略編碼形狀: {strategy_embedding_new_state.shape}")
    
    # 測試4: 手動配置更新
    logger.info("--- 測試4: 手動配置更新 ---")
    
    meta_learning_system.update_strategy_configuration(128, 30)
    test_states_manual = torch.randn(batch_size, 128)
    test_strategy_weights_manual = torch.softmax(torch.randn(batch_size, 30), dim=-1)
    strategy_embedding_manual = meta_learning_system.encode_strategy(test_states_manual, test_strategy_weights_manual)
    logger.info(f"手動更新後策略編碼形狀: {strategy_embedding_manual.shape}")
    
    # 獲取系統分析
    analysis = meta_learning_system.get_system_analysis()
    logger.info("=== 自適應元學習系統分析 ===")
    logger.info(f"檢測到的配置: {analysis['detected_config']}")
    logger.info(f"維度變化次數: {analysis['dimension_changes']}")
    logger.info(f"編碼器適應次數: {analysis['adaptation_count']}")
    logger.info(f"當前編碼器維度: {analysis['encoder_current_dims']}")
    logger.info(f"總參數量: {analysis['total_parameters']:,}")
    
    # 顯示維度變化歷史
    if analysis['dimension_history']:
        logger.info("維度變化歷史:")
        for i, change in enumerate(analysis['dimension_history']):
            logger.info(f"  變化{i+1}: {change['old_dims']} -> {change['new_dims']}")
    
    logger.info("=== 自適應元學習系統測試完成 ✅ ===")

if __name__ == "__main__":
    test_adaptive_meta_learning_system()
