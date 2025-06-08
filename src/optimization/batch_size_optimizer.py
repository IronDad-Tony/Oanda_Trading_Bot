# src/optimization/batch_size_optimizer.py
"""
批次大小自動優化工具
基於GPU能力測試結果自動優化所有配置文件中的批次大小

主要功能：
1. 檢測當前配置的批次大小
2. 基於GPU測試結果計算最佳批次大小
3. 自動更新所有相關配置文件
4. 提供安全的回退機制
"""

import json
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
import logging
from datetime import datetime

# 添加項目根目錄到Python路徑
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    # 嘗試相對導入
    from src.common.logger_setup import logger
    from src.optimization.performance_optimizer import PerformanceOptimizer
except ImportError:
    try:
        # 嘗試絕對導入
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from src.common.logger_setup import logger
        from src.optimization.performance_optimizer import PerformanceOptimizer
    except ImportError:
        # 創建基本logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)


class BatchSizeOptimizer:
    """批次大小優化器"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self.config_files = self._discover_config_files()
        self.backup_dir = self.project_root / "backups" / "config_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def _discover_config_files(self) -> Dict[str, Path]:
        """發現所有包含批次大小的配置文件"""
        config_files = {}
        
        # 主要配置文件
        main_config = self.project_root / "src" / "common" / "config.py"
        if main_config.exists():
            config_files["main_config"] = main_config
            
        # Transformer配置文件
        transformer_config = self.project_root / "configs" / "enhanced_transformer_config.json"
        if transformer_config.exists():
            config_files["transformer_config"] = transformer_config
            
        # 其他可能的配置文件
        for pattern in ["configs/*.json", "src/**/*config*.py"]:
            parts = pattern.split("*")
            base_dir = self.project_root / parts[0]
            if base_dir.exists():
                for file in base_dir.rglob(f"*{parts[1]}*{parts[2] if len(parts) > 2 else ''}"):
                    if file.name not in [f.name for f in config_files.values()]:
                        config_files[file.stem] = file
                        
        return config_files
    
    def analyze_current_batch_sizes(self) -> Dict[str, Any]:
        """分析當前的批次大小配置"""
        current_configs = {}
        
        for config_name, config_path in self.config_files.items():
            try:
                if config_path.suffix == ".py":
                    # Python配置文件
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    batch_sizes = self._extract_batch_sizes_from_python(content)
                    if batch_sizes:
                        current_configs[config_name] = {
                            "type": "python",
                            "path": config_path,
                            "batch_sizes": batch_sizes
                        }
                        
                elif config_path.suffix == ".json":
                    # JSON配置文件
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                    
                    batch_sizes = self._extract_batch_sizes_from_json(content)
                    if batch_sizes:
                        current_configs[config_name] = {
                            "type": "json",
                            "path": config_path,
                            "batch_sizes": batch_sizes,
                            "content": content
                        }
                        
            except Exception as e:
                logger.warning(f"無法分析配置文件 {config_path}: {e}")
                
        return current_configs
    
    def _extract_batch_sizes_from_python(self, content: str) -> Dict[str, int]:
        """從Python文件中提取批次大小"""
        batch_sizes = {}
        
        # 匹配各種批次大小變數
        patterns = [
            r'SAC_BATCH_SIZE\s*=\s*(\d+)',
            r'BATCH_SIZE\s*=\s*(\d+)',
            r'batch_size\s*=\s*(\d+)',
            r'TRANSFORMER_BATCH_SIZE\s*=\s*(\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                var_name = pattern.split('\\')[0]
                batch_sizes[var_name] = int(matches[0])
                
        return batch_sizes
    
    def _extract_batch_sizes_from_json(self, content: dict) -> Dict[str, int]:
        """從JSON配置中提取批次大小"""
        batch_sizes = {}
        
        def search_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if key == "batch_size" and isinstance(value, int):
                        batch_sizes[new_path] = value
                    elif isinstance(value, (dict, list)):
                        search_recursive(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_recursive(item, f"{path}[{i}]")
        
        search_recursive(content)
        return batch_sizes
    
    def calculate_optimal_batch_sizes(self, gpu_test_result: int = None) -> Dict[str, int]:
        """計算最佳批次大小"""
        if gpu_test_result is None:
            # 運行GPU測試
            try:
                optimizer = PerformanceOptimizer()
                result = optimizer.test_optimal_batch_size()
                optimal_batch_size = result["recommended_batch_size"]
            except Exception as e:
                logger.warning(f"GPU測試失敗，使用預設值: {e}")
                optimal_batch_size = 64
        else:
            optimal_batch_size = gpu_test_result
          # 為不同用途計算合適的批次大小
        optimal_configs = {
            "SAC_BATCH_SIZE": min(optimal_batch_size, 128),  # SAC不需要太大
            "transformer_batch_size": optimal_batch_size,     # Transformer可以用較大的
            "general_batch_size": optimal_batch_size // 2,    # 通用批次大小
            "evaluation_batch_size": optimal_batch_size // 4  # 評估用較小的
        }
        
        return optimal_configs
    
    def backup_configs(self) -> List[str]:
        """備份當前配置"""
        backup_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for config_name, config_path in self.config_files.items():
            try:
                # 只備份文件，不備份目錄
                if config_path.is_file():
                    backup_path = self.backup_dir / f"{config_name}_backup_{timestamp}{config_path.suffix}"
                    backup_path.write_text(config_path.read_text(encoding='utf-8'), encoding='utf-8')
                    backup_files.append(str(backup_path))  # 轉換為字符串
                    logger.info(f"已備份 {config_path} -> {backup_path}")
            except Exception as e:
                logger.error(f"備份失敗 {config_path}: {e}")
                
        return backup_files
    
    def _serialize_configs(self, configs: Dict[str, Any]) -> Dict[str, Any]:
        """序列化配置字典，處理Path對象"""
        serialized = {}
        for key, value in configs.items():
            serialized_value = value.copy()
            if "path" in serialized_value:
                serialized_value["path"] = str(serialized_value["path"])
            serialized[key] = serialized_value
        return serialized
    
    def update_config_files(self, optimal_configs: Dict[str, int]) -> Dict[str, bool]:
        """更新配置文件"""
        update_results = {}
        
        for config_name, config_info in self.analyze_current_batch_sizes().items():
            try:
                if config_info["type"] == "python":
                    success = self._update_python_config(config_info, optimal_configs)
                elif config_info["type"] == "json":
                    success = self._update_json_config(config_info, optimal_configs)
                else:
                    success = False
                    
                update_results[config_name] = success
                
            except Exception as e:
                logger.error(f"更新配置失敗 {config_name}: {e}")
                update_results[config_name] = False
                
        return update_results
    
    def _update_python_config(self, config_info: Dict, optimal_configs: Dict[str, int]) -> bool:
        """更新Python配置文件"""
        try:
            config_path = config_info["path"]
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 更新SAC_BATCH_SIZE
            if "SAC_BATCH_SIZE" in optimal_configs:
                pattern = r'SAC_BATCH_SIZE\s*=\s*\d+'
                replacement = f'SAC_BATCH_SIZE = {optimal_configs["SAC_BATCH_SIZE"]}'
                content = re.sub(pattern, replacement, content)
            
            # 更新其他批次大小變數
            for var_pattern in ['BATCH_SIZE', 'batch_size']:
                if var_pattern in content and "general_batch_size" in optimal_configs:
                    pattern = f'{var_pattern}\\s*=\\s*\\d+'
                    replacement = f'{var_pattern} = {optimal_configs["general_batch_size"]}'
                    content = re.sub(pattern, replacement, content)
            
            # 寫回文件
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            logger.info(f"已更新Python配置: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"更新Python配置失敗: {e}")
            return False
    
    def _update_json_config(self, config_info: Dict, optimal_configs: Dict[str, int]) -> bool:
        """更新JSON配置文件"""
        try:
            config_path = config_info["path"]
            content = config_info["content"].copy()
            
            # 更新訓練相關的批次大小
            if "training" in content and "batch_size" in content["training"]:
                content["training"]["batch_size"] = optimal_configs.get("transformer_batch_size", 32)
            
            # 更新評估相關的批次大小
            if "evaluation" in content and "batch_size" in content["evaluation"]:
                content["evaluation"]["batch_size"] = optimal_configs.get("evaluation_batch_size", 16)
            
            # 寫回文件
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
                
            logger.info(f"已更新JSON配置: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"更新JSON配置失敗: {e}")
            return False
    
    def optimize_batch_sizes(self, gpu_test_result: int = None, 
                           create_backup: bool = True) -> Dict[str, Any]:
        """執行批次大小優化"""
        logger.info("開始批次大小優化...")
        
        # 分析當前配置
        current_configs = self.analyze_current_batch_sizes()
        logger.info(f"發現 {len(current_configs)} 個配置文件")
        
        # 計算最佳批次大小
        optimal_configs = self.calculate_optimal_batch_sizes(gpu_test_result)
        logger.info(f"計算出最佳批次大小: {optimal_configs}")
        
        # 備份現有配置
        backup_files = []
        if create_backup:
            backup_files = self.backup_configs()
          # 更新配置文件
        update_results = self.update_config_files(optimal_configs)
        
        # 生成報告
        report = {
            "timestamp": datetime.now().isoformat(),
            "current_configs": self._serialize_configs(current_configs),
            "optimal_configs": optimal_configs,
            "update_results": update_results,
            "backup_files": backup_files,  # 已經是字符串列表
            "success_rate": sum(update_results.values()) / len(update_results) if update_results else 0
        }
        
        # 保存報告
        report_path = self.project_root / "reports" / "batch_size_optimization.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"優化完成，報告已保存至: {report_path}")
        return report
    
    def print_optimization_summary(self, report: Dict[str, Any]):
        """打印優化摘要"""
        print("\n" + "="*60)
        print("🚀 批次大小優化報告")
        print("="*60)
        
        print(f"\n📋 發現的配置文件:")
        for config_name, config_info in report["current_configs"].items():
            print(f"  • {config_name}: {config_info['path']}")
            for var_name, current_value in config_info["batch_sizes"].items():
                print(f"    - {var_name}: {current_value}")
        
        print(f"\n🎯 建議的最佳批次大小:")
        for config_name, optimal_value in report["optimal_configs"].items():
            print(f"  • {config_name}: {optimal_value}")
        
        print(f"\n✅ 更新結果:")
        success_count = sum(report["update_results"].values())
        total_count = len(report["update_results"])
        print(f"  成功更新: {success_count}/{total_count} 個配置文件")
        
        for config_name, success in report["update_results"].items():
            icon = "✅" if success else "❌"
            print(f"  {icon} {config_name}")
        
        if report["backup_files"]:
            print(f"\n💾 備份文件:")
            for backup_file in report["backup_files"]:
                print(f"  • {backup_file}")
        
        print("\n" + "="*60)


def main():
    """主函數：執行批次大小優化"""
    # 首先運行GPU測試
    print("🔍 正在測試GPU最佳批次大小...")
    try:
        from src.optimization.performance_optimizer import PerformanceOptimizer
        perf_optimizer = PerformanceOptimizer()
        gpu_result = perf_optimizer.test_optimal_batch_size()
        optimal_batch_size = gpu_result["recommended_batch_size"]
        print(f"✅ GPU測試完成，建議批次大小: {optimal_batch_size}")
    except Exception as e:
        print(f"⚠️ GPU測試失敗，使用預設值: {e}")
        optimal_batch_size = 64
    
    # 執行批次大小優化
    optimizer = BatchSizeOptimizer()
    report = optimizer.optimize_batch_sizes(gpu_test_result=optimal_batch_size)
    optimizer.print_optimization_summary(report)
    
    # 提示重啟
    if report["success_rate"] > 0:
        print("\n⚠️ 配置已更新，建議重啟訓練程序以應用新設定")
        return True
    else:
        print("\n❌ 優化失敗，請檢查錯誤日誌")
        return False


if __name__ == "__main__":
    main()
