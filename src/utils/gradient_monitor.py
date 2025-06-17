#!/usr/bin/env python3
"""
梯度和參數監控工具
用於實時監控模型訓練過程中的梯度流和參數更新
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from collections import defaultdict

class GradientMonitor:
    """梯度和參數監控工具"""
    
    def __init__(self, model, logger: Optional[logging.Logger] = None):
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.parameter_history = defaultdict(list)
        self.gradient_history = defaultdict(list)
        self.iteration_count = 0
        
        # 檢查模型是否有效
        if self.model is None:
            self.logger.warning("⚠️ 嘗試監控 None 模型")
            self.is_valid = False
        else:
            try:
                # 嘗試訪問 named_parameters 來驗證模型
                param_count = sum(1 for _ in self.model.named_parameters())
                self.is_valid = param_count > 0
                if not self.is_valid:
                    self.logger.warning(f"⚠️ 模型沒有參數: {type(self.model)}")
                else:                    self.logger.info(f"✓ 梯度監控器已設置，模型參數數量: {param_count}")
            except Exception as e:
                self.logger.warning(f"⚠️ 無法訪問模型參數: {e}")
                self.is_valid = False
    
    def record_parameters(self, step: int = None):
        """記錄當前參數狀態"""
        if not self.is_valid:
            return {}
            
        if step is None:
            step = self.iteration_count
            self.iteration_count += 1
            
        param_norms = {}
        for name, param in self.model.named_parameters():
            if param.data is not None:
                norm = param.data.norm().item()
                param_norms[name] = norm
                self.parameter_history[name].append((step, norm))
                
        return param_norms
    
    def record_gradients(self, step: int = None):
        """記錄當前梯度狀態"""
        if not self.is_valid:
            return {}
            
        if step is None:
            step = self.iteration_count
            
        grad_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                grad_norms[name] = norm
                self.gradient_history[name].append((step, norm))
            else:
                grad_norms[name] = 0.0
                self.gradient_history[name].append((step, 0.0))
                
        return grad_norms
    
    def check_gradient_flow(self) -> Dict[str, Any]:
        """檢查梯度流狀況"""
        if not self.is_valid:
            return {
                'has_gradients': False,
                'gradient_norms': {},
                'parameter_norms': {},
                'zero_gradients': [],
                'large_gradients': [],
                'gradient_flow_ok': False,
                'total_gradient_norm': 0.0,
                'zero_gradient_ratio': 1.0
            }
        
        results = {
            'has_gradients': False,
            'gradient_norms': {},
            'parameter_norms': {},
            'zero_gradients': [],
            'large_gradients': [],
            'gradient_flow_ok': False
        }
        
        total_grad_norm = 0.0
        param_count = 0
        zero_grad_count = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_count += 1
                
                # 參數範數
                param_norm = param.data.norm().item()
                results['parameter_norms'][name] = param_norm
                
                # 梯度範數
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    results['gradient_norms'][name] = grad_norm
                    total_grad_norm += grad_norm
                    
                    if grad_norm == 0.0:
                        zero_grad_count += 1
                        results['zero_gradients'].append(name)
                    elif grad_norm > 10.0:  # 大梯度閾值
                        results['large_gradients'].append(name)
                else:
                    results['gradient_norms'][name] = 0.0
                    zero_grad_count += 1
                    results['zero_gradients'].append(name)
        
        results['has_gradients'] = total_grad_norm > 0
        results['total_gradient_norm'] = total_grad_norm
        results['zero_gradient_ratio'] = zero_grad_count / max(param_count, 1)
        results['gradient_flow_ok'] = (
            results['has_gradients'] and
            results['zero_gradient_ratio'] < 0.5
        )
        
        return results
    
    def log_gradient_status(self, prefix: str = ""):
        """記錄梯度狀態到日誌"""
        status = self.check_gradient_flow()
        
        self.logger.info(f"{prefix}梯度流檢查結果:")
        self.logger.info(f"  - 是否有梯度: {status['has_gradients']}")
        self.logger.info(f"  - 總梯度範數: {status['total_gradient_norm']:.6f}")
        self.logger.info(f"  - 零梯度比例: {status['zero_gradient_ratio']:.2%}")
        self.logger.info(f"  - 梯度流正常: {status['gradient_flow_ok']}")
        
        if status['zero_gradients']:
            self.logger.warning(f"  - 零梯度參數數量: {len(status['zero_gradients'])}")
            if len(status['zero_gradients']) <= 5:
                self.logger.warning(f"  - 零梯度參數: {status['zero_gradients']}")
        
        if status['large_gradients']:
            self.logger.warning(f"  - 大梯度參數: {status['large_gradients']}")
    
    def compare_parameters(self, step1: int, step2: int) -> Dict[str, float]:
        """比較兩個步驟之間的參數變化"""
        changes = {}
        
        for name in self.parameter_history:
            history = self.parameter_history[name]
            
            # 找到對應步驟的參數值
            val1 = None
            val2 = None
            
            for step, value in history:
                if step == step1:
                    val1 = value
                if step == step2:
                    val2 = value
            
            if val1 is not None and val2 is not None:
                if val1 != 0:
                    changes[name] = abs(val2 - val1) / val1  # 相對變化
                else:
                    changes[name] = abs(val2 - val1)  # 絕對變化
        
        return changes
    
    def get_parameter_update_ratio(self) -> float:
        """計算參數更新比例"""
        if len(self.parameter_history) == 0:
            return 0.0
        
        total_change = 0.0
        param_count = 0
        
        for name in self.parameter_history:
            history = self.parameter_history[name]
            if len(history) >= 2:
                recent_val = history[-1][1]
                prev_val = history[-2][1]
                
                if prev_val != 0:
                    change_ratio = abs(recent_val - prev_val) / prev_val
                    total_change += change_ratio
                    param_count += 1
        
        return total_change / max(param_count, 1)
    
    def reset_history(self):
        """重置監控歷史"""
        self.parameter_history.clear()
        self.gradient_history.clear()
        self.iteration_count = 0
        self.logger.info("梯度監控歷史已重置")

def create_gradient_callback(monitor: GradientMonitor):
    """創建用於stable-baselines3的梯度監控回調"""
    from stable_baselines3.common.callbacks import BaseCallback
    
    class GradientMonitorCallback(BaseCallback):
        def __init__(self, gradient_monitor: GradientMonitor, log_freq: int = 100):
            super().__init__()
            self.gradient_monitor = gradient_monitor
            self.log_freq = log_freq
        
        def _on_step(self) -> bool:
            if self.num_timesteps % self.log_freq == 0:
                # 記錄參數和梯度
                self.gradient_monitor.record_parameters(self.num_timesteps)
                self.gradient_monitor.record_gradients(self.num_timesteps)
                
                # 記錄狀態
                self.gradient_monitor.log_gradient_status(f"Step {self.num_timesteps}: ")
                
                # 計算參數更新比例
                update_ratio = self.gradient_monitor.get_parameter_update_ratio()
                self.gradient_monitor.logger.info(f"參數更新比例: {update_ratio:.6f}")
            
            return True
    
    return GradientMonitorCallback(monitor)
