# src/optimization/performance_optimizer.py
"""
系統性能優化器
基於RTX 4060 Ti 16GB GPU優化訓練性能

主要功能：
1. GPU記憶體優化
2. 批次大小動態調整
3. 混合精度訓練優化
4. 梯度累積策略
5. 模型並行化
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import psutil
import gc

try:
    from oanda_trading_bot.training_system.common.logger_setup import logger
    from oanda_trading_bot.training_system.common.config import DEVICE, USE_AMP
except ImportError:
    logger = logging.getLogger(__name__)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True


class PerformanceOptimizer:
    """性能優化器"""
    
    def __init__(self, model: nn.Module, target_gpu_utilization: float = 0.85):
        self.model = model
        self.target_gpu_utilization = target_gpu_utilization
        self.device = DEVICE
        self.gpu_info = self._get_gpu_info()
        self.optimal_batch_size = None
        self.optimal_gradient_accumulation = 1
        
    def _get_gpu_info(self) -> Dict[str, Any]:
        """獲取GPU信息"""
        if not torch.cuda.is_available():
            return {"available": False}
            
        gpu_info = {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "compute_capability": torch.cuda.get_device_properties(0).major
        }
        
        logger.info(f"GPU信息: {gpu_info['name']} - {gpu_info['total_memory_gb']:.1f}GB")
        return gpu_info
    
    def optimize_batch_size(self, input_shape: Tuple[int, ...], 
                          min_batch_size: int = 1, 
                          max_batch_size: int = 256) -> int:
        """動態優化批次大小"""
        logger.info("開始批次大小優化...")
        
        if not self.gpu_info["available"]:
            logger.warning("GPU不可用，使用默認批次大小")
            return 16
        
        # 從最大值開始二分搜索
        left, right = min_batch_size, max_batch_size
        optimal_size = min_batch_size
        
        while left <= right:
            mid = (left + right) // 2
            
            try:
                # 測試批次大小
                if self._test_batch_size(input_shape, mid):
                    optimal_size = mid
                    left = mid + 1
                else:
                    right = mid - 1
                    
            except Exception as e:
                logger.warning(f"測試批次大小 {mid} 失敗: {e}")
                right = mid - 1
        
        # 留出安全餘量
        safe_batch_size = max(1, int(optimal_size * 0.8))
        
        logger.info(f"優化完成 - 建議批次大小: {safe_batch_size}")
        self.optimal_batch_size = safe_batch_size
        return safe_batch_size
    
    def _test_batch_size(self, input_shape: Tuple[int, ...], batch_size: int) -> bool:
        """測試特定批次大小是否可行"""
        try:
            # 清理GPU記憶體
            torch.cuda.empty_cache()
            gc.collect()
            
            # 創建測試數據
            test_input = torch.randn(batch_size, *input_shape[1:]).to(self.device)
            
            # 前向傳播測試
            self.model.eval()
            with torch.no_grad():
                _ = self.model(test_input)
            
            # 檢查記憶體使用率
            memory_used = torch.cuda.memory_allocated() / self.gpu_info["total_memory"]
            
            # 清理測試數據
            del test_input
            torch.cuda.empty_cache()
            
            return memory_used < self.target_gpu_utilization
            
        except Exception:
            return False
    
    def setup_mixed_precision(self) -> Tuple[torch.cuda.amp.GradScaler, bool]:
        """設置混合精度訓練"""
        if not self.gpu_info["available"] or self.gpu_info["compute_capability"] < 7:
            logger.warning("GPU不支持混合精度訓練")
            return None, False
        
        scaler = torch.cuda.amp.GradScaler()
        logger.info("✓ 混合精度訓練已啟用")
        return scaler, True
    
    def optimize_gradient_accumulation(self, target_effective_batch_size: int) -> int:
        """優化梯度累積步數"""
        if self.optimal_batch_size is None:
            logger.warning("請先運行批次大小優化")
            return 1
        
        gradient_accumulation = max(1, target_effective_batch_size // self.optimal_batch_size)
        self.optimal_gradient_accumulation = gradient_accumulation
        
        effective_batch_size = self.optimal_batch_size * gradient_accumulation
        logger.info(f"梯度累積優化: {gradient_accumulation}步, 有效批次大小: {effective_batch_size}")
        
        return gradient_accumulation
    
    def enable_torch_optimizations(self):
        """啟用PyTorch優化"""
        # 啟用cudnn benchmark
        torch.backends.cudnn.benchmark = True
        
        # 設置記憶體碎片整理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 設置線程數
        num_threads = min(8, psutil.cpu_count())
        torch.set_num_threads(num_threads)
        
        logger.info(f"✓ PyTorch優化已啟用 - 線程數: {num_threads}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """獲取優化摘要"""
        return {
            "gpu_info": self.gpu_info,
            "optimal_batch_size": self.optimal_batch_size,
            "gradient_accumulation": self.optimal_gradient_accumulation,
            "mixed_precision_available": self.gpu_info.get("compute_capability", 0) >= 7,
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self) -> Dict[str, str]:
        """獲取優化建議"""
        recommendations = {}
        
        if self.gpu_info["available"]:
            if self.gpu_info["total_memory_gb"] >= 12:
                recommendations["model_size"] = "可以使用大型模型配置"
            else:
                recommendations["model_size"] = "建議使用中型模型配置"
                
            if self.gpu_info["compute_capability"] >= 7:
                recommendations["precision"] = "建議使用混合精度訓練"
            else:
                recommendations["precision"] = "使用FP32精度"
        else:
            recommendations["fallback"] = "GPU不可用，將使用CPU訓練"
        
        return recommendations


def optimize_training_config(model: nn.Module, 
                           input_shape: Tuple[int, ...],
                           target_batch_size: int = 128) -> Dict[str, Any]:
    """優化訓練配置的便捷函數"""
    optimizer = PerformanceOptimizer(model)
    
    # 啟用基礎優化
    optimizer.enable_torch_optimizations()
    
    # 優化批次大小
    optimal_batch_size = optimizer.optimize_batch_size(input_shape)
    
    # 優化梯度累積
    gradient_accumulation = optimizer.optimize_gradient_accumulation(target_batch_size)
    
    # 設置混合精度
    scaler, use_amp = optimizer.setup_mixed_precision()
    
    config = {
        "batch_size": optimal_batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "effective_batch_size": optimal_batch_size * gradient_accumulation,
        "use_amp": use_amp,
        "scaler": scaler,
        "summary": optimizer.get_optimization_summary()
    }
    
    return config


if __name__ == "__main__":
    # 測試性能優化器
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(768, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = DummyModel().to(DEVICE)
    input_shape = (32, 768)  # (batch_size, features)
    
    config = optimize_training_config(model, input_shape)
    
    print("\n" + "="*50)
    print("🚀 性能優化結果")
    print("="*50)
    print(f"建議批次大小: {config['batch_size']}")
    print(f"梯度累積步數: {config['gradient_accumulation_steps']}")
    print(f"有效批次大小: {config['effective_batch_size']}")
    print(f"混合精度訓練: {config['use_amp']}")
    print("="*50)
