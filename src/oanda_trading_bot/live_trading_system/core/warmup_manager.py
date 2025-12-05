"""
WarmupManager - 系統預熱管理器
用於模型預加載、GPU記憶體預分配、推理預熱
"""
import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WarmupConfig:
    """預熱配置"""
    model_path: Optional[Path] = None
    warmup_iterations: int = 10
    preallocate_gpu_memory: bool = True
    gpu_memory_fraction: float = 0.8
    compile_model: bool = True  # PyTorch 2.0 torch.compile
    preload_scalers: bool = True


class WarmupManager:
    """
    系統預熱管理器
    
    功能：
    1. 模型預加載到GPU
    2. GPU記憶體預分配
    3. 推理預熱（JIT編譯）
    4. 連接池預熱
    """
    
    def __init__(self, config: Optional[WarmupConfig] = None):
        self.config = config or WarmupConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_warmed_up = False
        self.warmup_stats: Dict[str, Any] = {}
        
    def warmup_all(self) -> Dict[str, Any]:
        """執行完整預熱流程"""
        logger.info("=" * 50)
        logger.info("Starting System Warmup...")
        logger.info("=" * 50)
        
        start_time = time.time()
        results = {}
        
        # 1. GPU預熱
        if self.device.type == "cuda" and self.config.preallocate_gpu_memory:
            results["gpu"] = self._warmup_gpu()
        
        # 2. 模型預載
        if self.config.model_path:
            results["model"] = self._warmup_model()
        
        # 3. 推理預熱
        if self.model is not None:
            results["inference"] = self._warmup_inference()
        
        total_time = time.time() - start_time
        self.warmup_stats = {
            "total_warmup_time": total_time,
            "device": str(self.device),
            "results": results
        }
        
        self.is_warmed_up = True
        logger.info(f"System Warmup Complete in {total_time:.2f}s")
        
        return self.warmup_stats
    
    def _warmup_gpu(self) -> Dict[str, Any]:
        """GPU預熱"""
        logger.info("Warming up GPU...")
        start = time.time()
        
        try:
            # 清理快取
            torch.cuda.empty_cache()
            
            # 獲取GPU資訊
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory = gpu_props.total_memory
            target_allocation = int(total_memory * self.config.gpu_memory_fraction * 0.1)
            
            # 預分配記憶體
            dummy = torch.zeros(target_allocation // 4, device=self.device)
            del dummy
            torch.cuda.synchronize()
            
            # 預熱CUDA kernel
            for _ in range(5):
                a = torch.randn(1024, 1024, device=self.device)
                b = torch.randn(1024, 1024, device=self.device)
                c = torch.matmul(a, b)
                del a, b, c
            
            torch.cuda.synchronize()
            
            elapsed = time.time() - start
            logger.info(f"GPU Warmup Complete: {elapsed:.2f}s")
            
            return {
                "success": True,
                "gpu_name": gpu_props.name,
                "total_memory_gb": total_memory / 1e9,
                "warmup_time": elapsed
            }
        except Exception as e:
            logger.error(f"GPU Warmup Failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _warmup_model(self) -> Dict[str, Any]:
        """模型預載"""
        logger.info(f"Loading model from {self.config.model_path}...")
        start = time.time()
        
        try:
            from oanda_trading_bot.training_system.agent.sac_agent_wrapper import QuantumEnhancedSAC
            
            # 載入模型
            self.model = QuantumEnhancedSAC.load(
                str(self.config.model_path),
                device=self.device
            )
            
            # PyTorch 2.0 編譯優化
            if self.config.compile_model and hasattr(torch, 'compile'):
                logger.info("Compiling model with torch.compile...")
                try:
                    self.model.policy = torch.compile(self.model.policy)
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
            
            elapsed = time.time() - start
            logger.info(f"Model Load Complete: {elapsed:.2f}s")
            
            return {
                "success": True,
                "model_path": str(self.config.model_path),
                "load_time": elapsed,
                "compiled": self.config.compile_model
            }
        except Exception as e:
            logger.error(f"Model Load Failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _warmup_inference(self) -> Dict[str, Any]:
        """推理預熱"""
        logger.info("Warming up inference pipeline...")
        start = time.time()
        
        try:
            # 創建模擬輸入
            batch_size = 1
            seq_len = 128
            num_symbols = 5
            feature_dim = 9  # OHLCV + indicators
            
            dummy_obs = np.random.randn(
                batch_size, num_symbols, seq_len, feature_dim
            ).astype(np.float32)
            
            inference_times = []
            
            # 預熱推理
            for i in range(self.config.warmup_iterations):
                iter_start = time.time()
                with torch.no_grad():
                    _ = self.model.predict(dummy_obs, deterministic=True)
                inference_times.append(time.time() - iter_start)
            
            avg_inference_time = np.mean(inference_times[2:])  # 跳過前2次
            
            elapsed = time.time() - start
            logger.info(f"Inference Warmup Complete: {elapsed:.2f}s")
            logger.info(f"Average Inference Time: {avg_inference_time*1000:.2f}ms")
            
            return {
                "success": True,
                "warmup_iterations": self.config.warmup_iterations,
                "warmup_time": elapsed,
                "avg_inference_ms": avg_inference_time * 1000,
                "inference_times": inference_times
            }
        except Exception as e:
            logger.error(f"Inference Warmup Failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_model(self):
        """獲取預熱後的模型"""
        if not self.is_warmed_up:
            logger.warning("System not warmed up. Call warmup_all() first.")
        return self.model
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取預熱統計"""
        return self.warmup_stats


def create_warmup_manager(model_path: Optional[str] = None) -> WarmupManager:
    """工廠函數"""
    config = WarmupConfig(
        model_path=Path(model_path) if model_path else None,
        warmup_iterations=10,
        preallocate_gpu_memory=True,
        compile_model=True
    )
    return WarmupManager(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = create_warmup_manager()
    stats = manager.warmup_all()
    print(f"Warmup Stats: {stats}")
