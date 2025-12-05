"""
CUDA/GPU 完整診斷與性能測試腳本
適用於 RTX 5090 32GB (Blackwell 架構)

測試項目：
1. CUDA 驅動與運行時版本
2. cuDNN 版本與狀態
3. PyTorch CUDA 支持
4. GPU 記憶體管理
5. Tensor Core 性能測試
6. 混合精度 (AMP) 測試
7. Transformer 推論基準測試
"""
import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# 設置路徑
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class CUDADiagnostics:
    """CUDA 診斷工具"""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.torch = None
        self.cuda_available = False
        
    def run_all_tests(self) -> Dict[str, Any]:
        """執行所有測試"""
        logger.info("=" * 70)
        logger.info("RTX 5090 CUDA/GPU 完整診斷測試")
        logger.info("=" * 70)
        
        self._test_python_version()
        self._test_pytorch_import()
        
        if self.cuda_available:
            self._test_cuda_details()
            self._test_cudnn()
            self._test_gpu_properties()
            self._test_memory_allocation()
            self._test_tensor_operations()
            self._test_mixed_precision()
            self._test_tensor_cores()
            self._test_transformer_benchmark()
            self._test_training_simulation()
        
        self._print_summary()
        self._save_report()
        
        return self.results
    
    def _test_python_version(self):
        """檢查 Python 版本"""
        version = sys.version
        logger.info(f"Python Version: {version}")
        self.results["python_version"] = version
    
    def _test_pytorch_import(self):
        """測試 PyTorch 導入"""
        logger.info("\n" + "=" * 50)
        logger.info("1. PyTorch CUDA 支持檢查")
        logger.info("=" * 50)
        
        try:
            import torch
            self.torch = torch
            
            self.results["pytorch_version"] = torch.__version__
            logger.info(f"✅ PyTorch Version: {torch.__version__}")
            
            self.cuda_available = torch.cuda.is_available()
            self.results["cuda_available"] = self.cuda_available
            
            if self.cuda_available:
                logger.info(f"✅ CUDA Available: True")
            else:
                logger.error("❌ CUDA Not Available!")
                logger.info("可能原因：")
                logger.info("  1. 未安裝 CUDA Toolkit")
                logger.info("  2. PyTorch 未編譯 CUDA 支持")
                logger.info("  3. 驅動程式版本不相容")
                
        except ImportError as e:
            logger.error(f"❌ PyTorch 導入失敗: {e}")
            self.results["pytorch_error"] = str(e)
    
    def _test_cuda_details(self):
        """檢查 CUDA 詳細信息"""
        logger.info("\n" + "=" * 50)
        logger.info("2. CUDA 版本信息")
        logger.info("=" * 50)
        
        torch = self.torch
        
        # CUDA 版本
        cuda_version = torch.version.cuda
        self.results["cuda_version"] = cuda_version
        logger.info(f"✅ CUDA Version (PyTorch built): {cuda_version}")
        
        # CUDA 架構
        cuda_arch = torch.cuda.get_arch_list()
        self.results["cuda_arch_list"] = cuda_arch
        logger.info(f"✅ Supported CUDA Architectures: {cuda_arch}")
        
        # 檢查 Blackwell 支持 (compute capability 10.0+)
        if any("sm_100" in arch or "sm_10" in arch for arch in cuda_arch):
            logger.info("✅ Blackwell 架構 (sm_100) 支持: 是")
        else:
            logger.warning("⚠️ 注意: PyTorch 可能尚未支持最新 Blackwell 架構")
            logger.info("   RTX 5090 可能以相容模式運行")
        
        # 設備數量
        device_count = torch.cuda.device_count()
        self.results["device_count"] = device_count
        logger.info(f"✅ GPU 設備數量: {device_count}")
    
    def _test_cudnn(self):
        """檢查 cuDNN"""
        logger.info("\n" + "=" * 50)
        logger.info("3. cuDNN 狀態")
        logger.info("=" * 50)
        
        torch = self.torch
        
        cudnn_available = torch.backends.cudnn.is_available()
        self.results["cudnn_available"] = cudnn_available
        
        if cudnn_available:
            cudnn_version = torch.backends.cudnn.version()
            self.results["cudnn_version"] = cudnn_version
            logger.info(f"✅ cuDNN Available: True")
            logger.info(f"✅ cuDNN Version: {cudnn_version}")
            
            # 啟用 cuDNN 優化
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            logger.info("✅ cuDNN Benchmark Mode: Enabled")
            logger.info("✅ cuDNN Auto-Tuner: Enabled")
        else:
            logger.error("❌ cuDNN Not Available!")
    
    def _test_gpu_properties(self):
        """檢查 GPU 屬性"""
        logger.info("\n" + "=" * 50)
        logger.info("4. GPU 硬體屬性")
        logger.info("=" * 50)
        
        torch = self.torch
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            
            gpu_info = {
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "major_compute": props.major,
                "minor_compute": props.minor,
                "multi_processor_count": props.multi_processor_count,
            }
            self.results[f"gpu_{i}"] = gpu_info
            
            logger.info(f"\n📊 GPU {i}: {props.name}")
            logger.info(f"   總記憶體: {props.total_memory / 1e9:.2f} GB")
            logger.info(f"   Compute Capability: {props.major}.{props.minor}")
            logger.info(f"   SM 數量: {props.multi_processor_count}")
            
            # RTX 5090 應該有約170 SM
            if props.multi_processor_count >= 150:
                logger.info(f"   ✅ 高端 GPU 確認 (SM >= 150)")
            
            # 確認記憶體 (RTX 5090 = 32GB)
            if props.total_memory >= 30e9:
                logger.info(f"   ✅ 大記憶體 GPU (>= 30GB)")
    
    def _test_memory_allocation(self):
        """測試記憶體分配"""
        logger.info("\n" + "=" * 50)
        logger.info("5. GPU 記憶體管理測試")
        logger.info("=" * 50)
        
        torch = self.torch
        
        # 清理
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 初始狀態
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"初始狀態 - 已分配: {allocated:.2f} GB, 已保留: {reserved:.2f} GB")
        
        # 分配大型張量 (測試 10GB)
        try:
            test_size_gb = 10
            elements = int(test_size_gb * 1e9 / 4)  # float32 = 4 bytes
            
            logger.info(f"嘗試分配 {test_size_gb} GB 張量...")
            start = time.time()
            large_tensor = torch.zeros(elements, dtype=torch.float32, device="cuda")
            elapsed = time.time() - start
            
            allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"✅ 分配成功! 耗時: {elapsed:.2f}s, 已分配: {allocated:.2f} GB")
            
            # 清理
            del large_tensor
            torch.cuda.empty_cache()
            
            self.results["memory_allocation_test"] = {
                "success": True,
                "size_gb": test_size_gb,
                "time_seconds": elapsed
            }
            
        except RuntimeError as e:
            logger.error(f"❌ 記憶體分配失敗: {e}")
            self.results["memory_allocation_test"] = {"success": False, "error": str(e)}
    
    def _test_tensor_operations(self):
        """測試張量運算"""
        logger.info("\n" + "=" * 50)
        logger.info("6. 張量運算性能測試")
        logger.info("=" * 50)
        
        torch = self.torch
        
        sizes = [(1024, 1024), (4096, 4096), (8192, 8192)]
        
        for size in sizes:
            a = torch.randn(size, device="cuda")
            b = torch.randn(size, device="cuda")
            
            # 預熱
            for _ in range(3):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # 計時
            start = time.time()
            iterations = 100
            for _ in range(iterations):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            avg_time_ms = elapsed / iterations * 1000
            
            # 計算 TFLOPS
            flops = 2 * size[0] * size[1] * size[1]  # 矩陣乘法 FLOPs
            tflops = flops / (avg_time_ms / 1000) / 1e12
            
            logger.info(f"✅ MatMul {size[0]}x{size[1]}: {avg_time_ms:.2f} ms, {tflops:.2f} TFLOPS")
            
            del a, b, c
        
        torch.cuda.empty_cache()
    
    def _test_mixed_precision(self):
        """測試混合精度 (AMP)"""
        logger.info("\n" + "=" * 50)
        logger.info("7. 混合精度 (AMP) 測試")
        logger.info("=" * 50)
        
        torch = self.torch
        
        # 檢查 BF16 支持 (Ampere+)
        bf16_supported = torch.cuda.is_bf16_supported()
        self.results["bf16_supported"] = bf16_supported
        logger.info(f"BFloat16 支持: {'✅ 是' if bf16_supported else '❌ 否'}")
        
        # AMP 測試
        try:
            from torch.cuda.amp import autocast, GradScaler
            
            scaler = GradScaler()
            
            # 創建測試模型
            model = torch.nn.Sequential(
                torch.nn.Linear(1024, 2048),
                torch.nn.ReLU(),
                torch.nn.Linear(2048, 1024)
            ).cuda()
            
            optimizer = torch.optim.Adam(model.parameters())
            x = torch.randn(32, 1024, device="cuda")
            
            # 預熱
            for _ in range(5):
                with autocast():
                    y = model(x)
                    loss = y.sum()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            torch.cuda.synchronize()
            
            # AMP基準測試
            start = time.time()
            for _ in range(100):
                with autocast():
                    y = model(x)
                    loss = y.sum()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            torch.cuda.synchronize()
            amp_time = time.time() - start
            
            # FP32基準測試
            model_fp32 = model.float()
            x_fp32 = x.float()
            optimizer_fp32 = torch.optim.Adam(model_fp32.parameters())
            
            start = time.time()
            for _ in range(100):
                y = model_fp32(x_fp32)
                loss = y.sum()
                loss.backward()
                optimizer_fp32.step()
                optimizer_fp32.zero_grad()
            torch.cuda.synchronize()
            fp32_time = time.time() - start
            
            speedup = fp32_time / amp_time
            
            logger.info(f"✅ AMP 測試成功")
            logger.info(f"   FP32 時間: {fp32_time*10:.2f} ms/iter")
            logger.info(f"   AMP 時間: {amp_time*10:.2f} ms/iter")
            logger.info(f"   加速比: {speedup:.2f}x")
            
            self.results["amp_test"] = {
                "success": True,
                "fp32_time_ms": fp32_time * 10,
                "amp_time_ms": amp_time * 10,
                "speedup": speedup
            }
            
            del model, model_fp32, x, x_fp32
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ AMP 測試失敗: {e}")
            self.results["amp_test"] = {"success": False, "error": str(e)}
    
    def _test_tensor_cores(self):
        """測試 Tensor Cores"""
        logger.info("\n" + "=" * 50)
        logger.info("8. Tensor Cores 性能測試")
        logger.info("=" * 50)
        
        torch = self.torch
        
        # Tensor Cores 最佳化需要特定尺寸 (8的倍數)
        m, n, k = 4096, 4096, 4096
        
        dtypes = [
            ("FP32", torch.float32),
            ("FP16", torch.float16),
            ("BF16", torch.bfloat16) if torch.cuda.is_bf16_supported() else None,
        ]
        
        results = {}
        
        for dtype_info in dtypes:
            if dtype_info is None:
                continue
            
            name, dtype = dtype_info
            
            try:
                a = torch.randn(m, k, dtype=dtype, device="cuda")
                b = torch.randn(k, n, dtype=dtype, device="cuda")
                
                # 預熱
                for _ in range(5):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                # 計時
                start = time.time()
                for _ in range(50):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                avg_time_ms = elapsed / 50 * 1000
                flops = 2 * m * n * k
                tflops = flops / (avg_time_ms / 1000) / 1e12
                
                results[name] = tflops
                logger.info(f"✅ {name}: {avg_time_ms:.2f} ms, {tflops:.2f} TFLOPS")
                
                del a, b, c
                
            except Exception as e:
                logger.error(f"❌ {name} 測試失敗: {e}")
        
        self.results["tensor_core_tflops"] = results
        
        # RTX 5090 FP16 理論峰值約 320 TFLOPS
        if "FP16" in results and results["FP16"] > 100:
            logger.info(f"✅ Tensor Cores 運作正常 (效率 > 30%)")
        
        torch.cuda.empty_cache()
    
    def _test_transformer_benchmark(self):
        """Transformer 推論基準測試"""
        logger.info("\n" + "=" * 50)
        logger.info("9. Transformer 推論基準測試")
        logger.info("=" * 50)
        
        torch = self.torch
        
        try:
            # 模擬訓練系統的 Transformer 配置
            batch_size = 32
            seq_len = 128
            hidden_dim = 1024
            num_layers = 12
            num_heads = 16
            
            # 創建 Transformer 編碼器
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True
            ).cuda()
            
            transformer = torch.nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            ).cuda()
            
            # 輸入
            x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")
            
            # 預熱
            with torch.no_grad():
                for _ in range(5):
                    _ = transformer(x)
            torch.cuda.synchronize()
            
            # FP32 基準
            start = time.time()
            with torch.no_grad():
                for _ in range(50):
                    _ = transformer(x)
            torch.cuda.synchronize()
            fp32_time = (time.time() - start) / 50 * 1000
            
            # AMP 基準
            from torch.cuda.amp import autocast
            start = time.time()
            with torch.no_grad():
                with autocast():
                    for _ in range(50):
                        _ = transformer(x)
            torch.cuda.synchronize()
            amp_time = (time.time() - start) / 50 * 1000
            
            logger.info(f"✅ Transformer 配置:")
            logger.info(f"   Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_dim}")
            logger.info(f"   Layers: {num_layers}, Heads: {num_heads}")
            logger.info(f"✅ FP32 推論: {fp32_time:.2f} ms/batch")
            logger.info(f"✅ AMP 推論: {amp_time:.2f} ms/batch")
            logger.info(f"   每秒處理: {1000/amp_time * batch_size:.0f} samples")
            
            self.results["transformer_benchmark"] = {
                "fp32_ms": fp32_time,
                "amp_ms": amp_time,
                "samples_per_second": 1000/amp_time * batch_size
            }
            
            del transformer, x
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ Transformer 測試失敗: {e}")
            self.results["transformer_benchmark"] = {"error": str(e)}
    
    def _test_training_simulation(self):
        """模擬訓練測試"""
        logger.info("\n" + "=" * 50)
        logger.info("10. 訓練模擬測試 (SAC Agent)")
        logger.info("=" * 50)
        
        torch = self.torch
        
        try:
            # 模擬 SAC 訓練中的批次處理
            batch_size = 64
            obs_dim = 128 * 5 * 9  # timesteps * symbols * features
            action_dim = 5 * 3     # symbols * actions
            hidden_dim = 1024
            
            # 模擬 Actor-Critic 網路
            actor = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, action_dim * 2)  # mean + log_std
            ).cuda()
            
            critic = torch.nn.Sequential(
                torch.nn.Linear(obs_dim + action_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1)
            ).cuda()
            
            # 優化器
            actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
            critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
            
            # 模擬訓練步驟
            scaler = torch.cuda.amp.GradScaler()
            
            # 預熱
            for _ in range(10):
                obs = torch.randn(batch_size, obs_dim, device="cuda")
                actions = torch.randn(batch_size, action_dim, device="cuda")
                
                with torch.cuda.amp.autocast():
                    # Actor 更新
                    actor_out = actor(obs)
                    actor_loss = actor_out[:, :action_dim].sum()
                    
                    # Critic 更新
                    critic_in = torch.cat([obs, actions], dim=-1)
                    q_value = critic(critic_in)
                    critic_loss = q_value.sum()
                
                scaler.scale(actor_loss).backward()
                scaler.step(actor_opt)
                scaler.update()
                actor_opt.zero_grad()
                
                scaler.scale(critic_loss).backward()
                scaler.step(critic_opt)
                scaler.update()
                critic_opt.zero_grad()
            
            torch.cuda.synchronize()
            
            # 基準測試
            start = time.time()
            num_steps = 1000
            
            for _ in range(num_steps):
                obs = torch.randn(batch_size, obs_dim, device="cuda")
                actions = torch.randn(batch_size, action_dim, device="cuda")
                
                with torch.cuda.amp.autocast():
                    actor_out = actor(obs)
                    actor_loss = actor_out[:, :action_dim].sum()
                    
                    critic_in = torch.cat([obs, actions], dim=-1)
                    q_value = critic(critic_in)
                    critic_loss = q_value.sum()
                
                scaler.scale(actor_loss).backward()
                scaler.step(actor_opt)
                scaler.update()
                actor_opt.zero_grad()
                
                scaler.scale(critic_loss).backward()
                scaler.step(critic_opt)
                scaler.update()
                critic_opt.zero_grad()
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            steps_per_second = num_steps / elapsed
            samples_per_second = steps_per_second * batch_size
            
            logger.info(f"✅ 訓練模擬成功!")
            logger.info(f"   步數/秒: {steps_per_second:.0f}")
            logger.info(f"   樣本/秒: {samples_per_second:.0f}")
            logger.info(f"   預估每小時訓練步數: {steps_per_second * 3600:.0f}")
            
            self.results["training_simulation"] = {
                "steps_per_second": steps_per_second,
                "samples_per_second": samples_per_second,
                "estimated_steps_per_hour": steps_per_second * 3600
            }
            
            del actor, critic
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ 訓練模擬失敗: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.results["training_simulation"] = {"error": str(e)}
    
    def _print_summary(self):
        """列印摘要"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 診斷摘要")
        logger.info("=" * 70)
        
        if self.cuda_available:
            logger.info("✅ CUDA 可用")
            logger.info(f"   PyTorch: {self.results.get('pytorch_version', 'N/A')}")
            logger.info(f"   CUDA: {self.results.get('cuda_version', 'N/A')}")
            logger.info(f"   cuDNN: {self.results.get('cudnn_version', 'N/A')}")
            
            if "gpu_0" in self.results:
                gpu = self.results["gpu_0"]
                logger.info(f"   GPU: {gpu['name']}")
                logger.info(f"   記憶體: {gpu['total_memory_gb']:.1f} GB")
            
            if "tensor_core_tflops" in self.results:
                tc = self.results["tensor_core_tflops"]
                logger.info(f"   FP16 TFLOPS: {tc.get('FP16', 'N/A')}")
            
            if "training_simulation" in self.results:
                ts = self.results["training_simulation"]
                if "steps_per_second" in ts:
                    logger.info(f"   訓練速度: {ts['steps_per_second']:.0f} steps/s")
            
            logger.info("\n🎉 您的 RTX 5090 已準備就緒進行訓練和推論!")
        else:
            logger.error("❌ CUDA 不可用，請檢查驅動和安裝")
    
    def _save_report(self):
        """保存報告"""
        import json
        
        report_path = PROJECT_ROOT / "reports" / f"cuda_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 轉換不可序列化的值
        def serialize(obj):
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize(v) for v in obj]
            else:
                return str(obj)
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(serialize(self.results), f, indent=2)
        
        logger.info(f"\n報告已保存: {report_path}")


def main():
    """主函數"""
    diagnostics = CUDADiagnostics()
    results = diagnostics.run_all_tests()
    
    return 0 if results.get("cuda_available", False) else 1


if __name__ == "__main__":
    sys.exit(main())
