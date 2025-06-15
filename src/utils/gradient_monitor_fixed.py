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
                else:
                    self.logger.info(f"✓ 梯度監控器已設置，模型參數數量: {param_count}")
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
            results['zero_gradient_ratio'] < 0.8  # 允許最多80%的參數無梯度
        )
        
        return results
    
    def get_parameter_update_ratio(self) -> float:
        """計算參數更新比例"""
        if not self.is_valid or len(self.parameter_history) < 2:
            return 0.0
            
        total_change = 0.0
        total_params = 0.0
        
        for param_name, history in self.parameter_history.items():
            if len(history) >= 2:
                # 比較最近兩次記錄
                _, old_norm = history[-2]
                _, new_norm = history[-1]
                
                change = abs(new_norm - old_norm)
                total_change += change
                total_params += max(old_norm, 1e-8)  # 避免除零
        
        if total_params > 0:
            return total_change / total_params
        return 0.0
    
    def get_gradient_statistics(self) -> Dict[str, float]:
        """獲取梯度統計信息"""
        if not self.is_valid:
            return {
                'mean_gradient': 0.0,
                'max_gradient': 0.0,
                'min_gradient': 0.0,
                'std_gradient': 0.0
            }
            
        all_gradients = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                all_gradients.append(grad_norm)
        
        if not all_gradients:
            return {
                'mean_gradient': 0.0,
                'max_gradient': 0.0,
                'min_gradient': 0.0,
                'std_gradient': 0.0
            }
        
        all_gradients = np.array(all_gradients)
        
        return {
            'mean_gradient': float(np.mean(all_gradients)),
            'max_gradient': float(np.max(all_gradients)),
            'min_gradient': float(np.min(all_gradients)),
            'std_gradient': float(np.std(all_gradients))
        }
    
    def print_gradient_summary(self):
        """打印梯度摘要"""
        if not self.is_valid:
            self.logger.warning("⚠️ 無法打印梯度摘要：模型無效")
            return
            
        gradient_stats = self.get_gradient_statistics()
        flow_status = self.check_gradient_flow()
        update_ratio = self.get_parameter_update_ratio()
        
        self.logger.info("=" * 50)
        self.logger.info("🔍 梯度流分析摘要")
        self.logger.info("=" * 50)
        self.logger.info(f"梯度流狀態: {'✅ 正常' if flow_status['gradient_flow_ok'] else '❌ 異常'}")
        self.logger.info(f"總梯度範數: {flow_status['total_gradient_norm']:.6f}")
        self.logger.info(f"零梯度比例: {flow_status['zero_gradient_ratio']:.2%}")
        self.logger.info(f"參數更新比例: {update_ratio:.8f}")
        self.logger.info(f"平均梯度: {gradient_stats['mean_gradient']:.6f}")
        self.logger.info(f"最大梯度: {gradient_stats['max_gradient']:.6f}")
        self.logger.info(f"最小梯度: {gradient_stats['min_gradient']:.6f}")
        self.logger.info("=" * 50)
