"""
GPU優化配置 - 針對 RTX 5090 32GB
自動檢測GPU並調整訓練參數
"""
import logging
import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class GPUOptimizedConfig:
    """GPU優化配置"""
    # 基礎設定
    device: str = "cuda"
    use_amp: bool = True
    amp_dtype: str = "float16"  # or "bfloat16"
    
    # 批次大小 (根據GPU記憶體自動調整)
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    
    # 模型大小
    hidden_dim: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    
    # 記憶體優化
    gradient_checkpointing: bool = False
    pin_memory: bool = True
    num_workers: int = 4
    
    # 效能優化
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    compile_model: bool = True  # PyTorch 2.0+
    
    # 經驗回放緩衝區
    buffer_size_per_symbol: int = 10000


class GPUOptimizer:
    """
    GPU優化器
    根據硬體自動調整配置
    """
    
    # GPU記憶體配置對照表 (GB -> 配置)
    MEMORY_CONFIGS = {
        8: {"batch_size": 32, "hidden_dim": 512, "num_layers": 12, "buffer_per_sym": 5000},
        12: {"batch_size": 64, "hidden_dim": 768, "num_layers": 16, "buffer_per_sym": 7500},
        16: {"batch_size": 96, "hidden_dim": 1024, "num_layers": 18, "buffer_per_sym": 8000},
        24: {"batch_size": 128, "hidden_dim": 1024, "num_layers": 20, "buffer_per_sym": 10000},
        32: {"batch_size": 192, "hidden_dim": 1024, "num_layers": 24, "buffer_per_sym": 15000},
        48: {"batch_size": 256, "hidden_dim": 2048, "num_layers": 28, "buffer_per_sym": 20000},
    }
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.gpu_info: Dict[str, Any] = {}
        
        if self.cuda_available:
            self._detect_gpu()
    
    def _detect_gpu(self):
        """檢測GPU資訊"""
        props = torch.cuda.get_device_properties(0)
        
        self.gpu_info = {
            "name": props.name,
            "memory_gb": props.total_memory / 1e9,
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "is_high_end": props.multi_processor_count >= 50,
            "supports_bf16": torch.cuda.is_bf16_supported(),
        }
        
        logger.info(f"Detected GPU: {props.name}")
        logger.info(f"Memory: {self.gpu_info['memory_gb']:.1f} GB")
        logger.info(f"Compute: {self.gpu_info['compute_capability']}")
    
    def get_optimized_config(self) -> GPUOptimizedConfig:
        """獲取優化配置"""
        config = GPUOptimizedConfig()
        
        if not self.cuda_available:
            config.device = "cpu"
            config.use_amp = False
            config.batch_size = 32
            config.num_workers = 0
            return config
        
        memory_gb = self.gpu_info["memory_gb"]
        
        # 找到最接近的配置
        best_match = 8
        for mem_tier in sorted(self.MEMORY_CONFIGS.keys()):
            if memory_gb >= mem_tier:
                best_match = mem_tier
        
        tier_config = self.MEMORY_CONFIGS[best_match]
        
        config.batch_size = tier_config["batch_size"]
        config.hidden_dim = tier_config["hidden_dim"]
        config.num_layers = tier_config["num_layers"]
        config.buffer_size_per_symbol = tier_config["buffer_per_sym"]
        
        # BF16 優先 (如果支持)
        if self.gpu_info.get("supports_bf16", False):
            config.amp_dtype = "bfloat16"
        
        # 高端GPU啟用編譯
        if self.gpu_info.get("is_high_end", False):
            config.compile_model = True
            config.num_workers = 8
        
        # RTX 5090 特殊優化
        if "5090" in self.gpu_info.get("name", ""):
            config.batch_size = 256
            config.buffer_size_per_symbol = 20000
            config.gradient_checkpointing = False  # 記憶體充足
            logger.info("Applied RTX 5090 optimizations")
        
        return config
    
    def apply_optimizations(self):
        """應用全局優化"""
        if not self.cuda_available:
            return
        
        # cuDNN優化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        
        # 記憶體分配優化
        if hasattr(torch.cuda, 'memory'):
            torch.cuda.memory.set_per_process_memory_fraction(0.95)
        
        # TF32 (Ampere+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        logger.info("GPU optimizations applied")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """獲取記憶體統計"""
        if not self.cuda_available:
            return {}
        
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "total_gb": self.gpu_info.get("memory_gb", 0),
        }
    
    def estimate_max_batch_size(
        self,
        model_params: int,
        sequence_length: int = 128,
        overhead_factor: float = 1.3
    ) -> int:
        """估算最大批次大小"""
        if not self.cuda_available:
            return 32
        
        available_memory = self.gpu_info["memory_gb"] * 0.9  # 保留10%
        
        # 粗略估算每個樣本的記憶體需求 (MB)
        bytes_per_param = 4 if not self.gpu_info.get("supports_bf16") else 2
        model_memory_gb = model_params * bytes_per_param / 1e9
        
        # 活化值記憶體 (粗略估算)
        activation_per_sample_mb = sequence_length * 1024 * 4 / 1e6
        
        available_for_batch = (available_memory - model_memory_gb) / overhead_factor
        max_batch = int(available_for_batch * 1000 / activation_per_sample_mb)
        
        # 確保是8的倍數 (Tensor Core優化)
        max_batch = (max_batch // 8) * 8
        
        return max(32, min(max_batch, 512))


# 全局優化器實例
_gpu_optimizer: Optional[GPUOptimizer] = None


def get_gpu_optimizer() -> GPUOptimizer:
    """獲取全局GPU優化器"""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        _gpu_optimizer = GPUOptimizer()
        _gpu_optimizer.apply_optimizations()
    return _gpu_optimizer


def get_optimized_training_config() -> Dict[str, Any]:
    """獲取優化的訓練配置字典"""
    optimizer = get_gpu_optimizer()
    config = optimizer.get_optimized_config()
    
    return {
        "device": config.device,
        "use_amp": config.use_amp,
        "batch_size": config.batch_size,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "buffer_size_per_symbol": config.buffer_size_per_symbol,
        "pin_memory": config.pin_memory,
        "num_workers": config.num_workers,
        "compile_model": config.compile_model,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    optimizer = GPUOptimizer()
    config = optimizer.get_optimized_config()
    
    print("\n" + "=" * 50)
    print("GPU優化配置")
    print("=" * 50)
    print(f"Device: {config.device}")
    print(f"Use AMP: {config.use_amp} ({config.amp_dtype})")
    print(f"Batch Size: {config.batch_size}")
    print(f"Hidden Dim: {config.hidden_dim}")
    print(f"Num Layers: {config.num_layers}")
    print(f"Buffer Size: {config.buffer_size_per_symbol}")
    print(f"Compile Model: {config.compile_model}")
    
    print("\n記憶體統計:")
    stats = optimizer.get_memory_stats()
    for k, v in stats.items():
        print(f"  {k}: {v:.2f} GB")
