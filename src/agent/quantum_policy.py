# src/agent/quantum_policy.py
"""
量子策略政策層 - 與SAC代理集成的包裝器
這個文件作為量子策略層與SAC代理之間的橋樑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import logging

try:
    from src.agent.quantum_strategy_layer import QuantumTradingLayer
    from src.common.logger_setup import logger
    from src.common.config import DEVICE
except ImportError:
    logger = logging.getLogger(__name__)
    DEVICE = "cpu"
    # 如果導入失敗，創建一個簡單的佔位符
    class QuantumTradingLayer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.linear = nn.Linear(128, 5)
        
        def forward(self, x, training=True):
            return self.linear(x), {}


class QuantumPolicyWrapper(nn.Module):
    """
    量子策略政策包裝器
    為SAC代理提供標準接口，內部使用完整的量子策略層
    """
    
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 256,
                 num_strategies: int = 3, num_energy_levels: int = 8):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # 初始化完整的量子策略層
        self.quantum_layer = QuantumTradingLayer(
            input_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            num_strategies=num_strategies,
            num_energy_levels=num_energy_levels
        )
        
        # SAC需要的額外層用於輸出動作分佈參數
        # 均值網絡
        self.action_mean = nn.Linear(action_dim, action_dim)
        
        # 對數標準差網絡  
        self.action_log_std = nn.Linear(action_dim, action_dim)
        
        # 約束log_std的範圍
        self.log_std_min = -20
        self.log_std_max = 2
        
        # 內部狀態追蹤
        self.last_quantum_info: Optional[Dict[str, Any]] = None
    
    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播，返回動作均值和對數標準差
        
        Args:
            state: 狀態張量
            deterministic: 是否確定性輸出
            
        Returns:
            Tuple[動作均值, 動作對數標準差]
        """
        # 使用量子策略層處理
        quantum_output, quantum_info = self.quantum_layer(state, training=self.training)
        
        # 保存量子信息用於監控
        self.last_quantum_info = quantum_info
        
        # 計算動作分佈參數
        action_mean = self.action_mean(quantum_output)
        action_log_std = self.action_log_std(quantum_output)
        
        # 約束log_std
        action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
        
        if deterministic:
            # 確定性模式：返回均值和零標準差
            return action_mean, torch.zeros_like(action_log_std)
        else:
            return action_mean, action_log_std
    
    def get_action_distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        """
        獲取動作的正態分佈
        
        Args:
            state: 狀態張量
            
        Returns:
            動作的正態分佈
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)
    
    def sample_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        採樣動作
        
        Args:
            state: 狀態張量
            deterministic: 是否確定性輸出
            
        Returns:
            Tuple[動作, 對數概率]
        """
        if deterministic:
            mean, _ = self.forward(state, deterministic=True)
            return torch.tanh(mean), torch.zeros(mean.shape[0], 1, device=mean.device)
        
        # 獲取動作分佈
        distribution = self.get_action_distribution(state)
        
        # 重參數化技巧採樣
        action_sample = distribution.rsample()
        
        # 計算tanh變換後的對數概率
        log_prob = distribution.log_prob(action_sample)
        
        # tanh變換的雅可比行列式修正
        action_tanh = torch.tanh(action_sample)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action_tanh, log_prob
    
    def get_quantum_metrics(self) -> Dict[str, float]:
        """獲取量子策略的性能指標"""
        if hasattr(self.quantum_layer, 'get_quantum_metrics'):
            return self.quantum_layer.get_quantum_metrics()
        return {}
    
    def get_last_quantum_info(self) -> Optional[Dict[str, Any]]:
        """獲取最後一次量子處理的詳細信息"""
        return self.last_quantum_info
    
    def reset_quantum_state(self):
        """重置量子狀態"""
        if hasattr(self.quantum_layer, 'reset_annealing'):
            self.quantum_layer.reset_annealing()
            
    def set_training_mode(self, mode: bool):
        """設置訓練模式"""
        self.train(mode)
        if hasattr(self.quantum_layer, 'train'):
            self.quantum_layer.train(mode)


# 為了向後兼容，提供原有的類名
QuantumPolicyLayer = QuantumPolicyWrapper


if __name__ == "__main__":
    # 測試量子策略政策包裝器
    logger.info("開始測試量子策略政策包裝器...")
    
    # 測試參數
    batch_size = 16
    state_dim = 128
    action_dim = 5
    
    # 創建測試數據
    test_state = torch.randn(batch_size, state_dim)
    
    # 初始化量子策略政策
    quantum_policy = QuantumPolicyWrapper(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=256
    )
    
    logger.info(f"量子策略政策參數數量: {sum(p.numel() for p in quantum_policy.parameters()):,}")
    
    # 測試前向傳播
    with torch.no_grad():
        mean, log_std = quantum_policy(test_state)
        logger.info(f"動作均值形狀: {mean.shape}")
        logger.info(f"動作log_std形狀: {log_std.shape}")
        
        # 測試動作採樣
        action, log_prob = quantum_policy.sample_action(test_state)
        logger.info(f"採樣動作形狀: {action.shape}")
        logger.info(f"對數概率形狀: {log_prob.shape}")
        logger.info(f"動作範圍: [{action.min().item():.3f}, {action.max().item():.3f}]")
        
        # 測試確定性輸出
        det_action, det_log_prob = quantum_policy.sample_action(test_state, deterministic=True)
        logger.info(f"確定性動作形狀: {det_action.shape}")
        
        # 獲取量子指標
        quantum_metrics = quantum_policy.get_quantum_metrics()
        if quantum_metrics:
            logger.info(f"量子指標: {quantum_metrics}")
    
    logger.info("量子策略政策包裝器測試完成！")