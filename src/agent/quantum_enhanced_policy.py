"""
量子增強策略
在SAC策略基礎上引入量子啟發式探索機制

核心創新:
1. 量子相位隨機疊加
2. 波函數坍縮模擬
3. 可調探索強度
"""
import torch
import math
from typing import Any, Dict, Tuple
from .sac_policy import CustomSACPolicy

class QuantumEnhancedPolicy(CustomSACPolicy):
    """
    量子增強探索策略
    
    參數:
        *args: SAC標準參數
        quantum_scale: 量子探索強度係數 (默認0.3)
        **kwargs: SAC標準參數
    """
    def __init__(self, *args, quantum_scale: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantum_scale = quantum_scale
    
    def quantum_explore(self, action_probs: torch.Tensor) -> torch.Tensor:
        """
        應用量子疊加原理的探索機制
        
        參數:
            action_probs: 原始動作概率分布 (batch_size, action_dim)
            
        返回:
            量子探索擾動 (batch_size, action_dim)
        """
        # 生成隨機相位 (0~2π)
        phase = torch.rand_like(action_probs) * (2 * math.pi)
        
        # 創建複數概率: p * e^(i*phase)
        complex_probs = action_probs * torch.exp(1j * phase)
        
        # 波函數坍縮: |ψ|^2
        quantum_probs = torch.abs(complex_probs)**2
        
        # 縮放探索強度
        return quantum_probs * self.quantum_scale

    def predict(self, observation: torch.Tensor, 
                deterministic: bool = False) -> torch.Tensor:
        """
        預測動作，加入量子探索
        
        參數:
            observation: 環境觀察值
            deterministic: 是否使用確定性策略
            
        返回:
            動作張量
        """
        # 原始動作預測
        action = super().predict(observation, deterministic)
        
        if not deterministic:
            # 量子增強探索
            exploration = self.quantum_explore(action)
            action += exploration
            
            # 限制動作在合法範圍
            action = torch.clamp(action, -1.0, 1.0)
        return action
    
    def set_quantum_scale(self, scale: float):
        """動態調整量子探索強度"""
        self.quantum_scale = max(0.0, min(scale, 1.0))