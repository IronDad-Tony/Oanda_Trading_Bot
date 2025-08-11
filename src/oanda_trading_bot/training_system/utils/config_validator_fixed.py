# src/utils/config_validator_fixed.py
"""
配置統一性驗證和修復工具
檢查系統各模組間配置的一致性，並提供自動修復建議

主要功能：
1. 配置一致性檢查
2. 維度匹配驗證  
3. 自動修復建議
4. 配置版本管理
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
import numpy as np

try:
    from oanda_trading_bot.training_system.common.logger_setup import logger
    from oanda_trading_bot.training_system.common.config import *
except ImportError:
    logger = logging.getLogger(__name__)


@dataclass
class ConfigIssue:
    """配置問題描述"""
    level: str  # "error", "warning", "info"
    category: str  # "dimension", "path", "value", "compatibility"
    module: str
    issue: str
    current_value: Any
    expected_value: Any = None
    fix_suggestion: str = ""


@dataclass
class ValidationReport:
    """驗證報告"""
    timestamp: str
    total_checks: int
    passed_checks: int
    issues: List[ConfigIssue]
    critical_errors: int
    warnings: int
    overall_status: str  # "passed", "warnings", "failed"


class ConfigValidator:
    """配置驗證器"""
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.config_files = self._discover_config_files()
        self.issues = []
        
    def _discover_config_files(self) -> List[Path]:
        """發現所有配置相關文件"""
        config_patterns = [
            "src/common/config.py",
            "configs/*.json",
            "src/environment/*config*.py", 
            "src/models/*config*.py",
            "src/agent/*config*.py"
        ]
        
        files = []
        for pattern in config_patterns:
            if "*" in pattern:
                parts = pattern.split("*")
                base_dir = self.project_root / parts[0]
                if base_dir.exists():
                    for file in base_dir.rglob(f"*{parts[1]}*{parts[2] if len(parts) > 2 else ''}"):
                        files.append(file)
            else:
                file_path = self.project_root / pattern
                if file_path.exists():
                    files.append(file_path)
        
        return files
    
    def validate_all(self) -> ValidationReport:
        """執行全面配置驗證"""
        logger.info("開始配置一致性驗證...")
        
        self.issues = []
        checks = [
            self._check_dimension_consistency,
            self._check_path_validity,
            self._check_transformer_config,
            self._check_quantum_strategy_config,
            self._check_progressive_learning_config,
            self._check_device_compatibility,
            self._check_data_consistency,
            self._check_training_parameters
        ]
        
        total_checks = len(checks)
        passed_checks = 0
        
        for check in checks:
            try:
                if check():
                    passed_checks += 1
                logger.info(f"✓ 完成檢查: {check.__name__}")
            except Exception as e:
                self._add_issue("error", "system", "ConfigValidator", 
                              f"檢查 {check.__name__} 失敗: {str(e)}")
                logger.error(f"✗ 檢查失敗: {check.__name__}: {e}")
        
        # 分析問題嚴重性
        critical_errors = len([i for i in self.issues if i.level == "error"])
        warnings = len([i for i in self.issues if i.level == "warning"])
        
        if critical_errors > 0:
            overall_status = "failed"
        elif warnings > 0:
            overall_status = "warnings"
        else:
            overall_status = "passed"
        
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            total_checks=total_checks,
            passed_checks=passed_checks,
            issues=self.issues,
            critical_errors=critical_errors,
            warnings=warnings,
            overall_status=overall_status
        )
        
        self._save_report(report)
        self._print_summary(report)
        
        return report
    
    def _check_dimension_consistency(self) -> bool:
        """檢查維度配置一致性"""
        try:
            # 檢查 Transformer 配置
            transformer_config_path = self.project_root / "configs" / "enhanced_transformer_config.json"
            if transformer_config_path.exists():
                with open(transformer_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                model_dim = config.get('model', {}).get('model_dim', 768)
                num_layers = config.get('model', {}).get('num_layers', 16)
                num_heads = config.get('model', {}).get('num_heads', 24)
                
                # 檢查是否為合理值
                if model_dim % num_heads != 0:
                    self._add_issue("error", "dimension", "enhanced_transformer_config.json",
                                  f"model_dim ({model_dim}) 必須被 num_heads ({num_heads}) 整除",
                                  f"{model_dim} % {num_heads} = {model_dim % num_heads}",
                                  f"調整 model_dim 為 {(model_dim // num_heads) * num_heads} 或調整 num_heads")
                    return False
                
        except Exception as e:
            self._add_issue("error", "dimension", "ConfigValidator", f"維度檢查失敗: {str(e)}")
            return False
            
        return True
    
    def _check_path_validity(self) -> bool:
        """檢查路徑配置有效性"""
        try:
            required_dirs = ['data', 'logs', 'weights', 'configs']
            for dir_name in required_dirs:
                dir_path = self.project_root / dir_name
                if not dir_path.exists():
                    self._add_issue("warning", "path", "project_structure",
                                  f"目錄 {dir_name} 不存在", str(dir_path),
                                  f"創建目錄: mkdir -p {dir_path}")
                    
            # 檢查 .env 文件
            env_path = self.project_root / ".env"
            if not env_path.exists():
                self._add_issue("warning", "path", ".env",
                              "環境配置文件不存在", str(env_path),
                              "創建 .env 文件並設置 OANDA API 配置")
                              
        except Exception as e:
            self._add_issue("error", "path", "ConfigValidator", f"路徑檢查失敗: {str(e)}")
            return False
            
        return True
    
    def _check_transformer_config(self) -> bool:
        """檢查 Transformer 配置合理性"""
        try:
            config_path = self.project_root / "configs" / "enhanced_transformer_config.json"
            if not config_path.exists():
                self._add_issue("warning", "compatibility", "enhanced_transformer_config.json",
                              "增強 Transformer 配置文件不存在", None,
                              "使用默認配置或創建配置文件")
                return True
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 檢查訓練參數合理性
            training = config.get('training', {})
            batch_size = training.get('batch_size', 32)
            learning_rate = training.get('learning_rate', 0.0001)
            
            if batch_size <= 0 or batch_size > 256:
                self._add_issue("warning", "value", "enhanced_transformer_config.json",
                              f"batch_size ({batch_size}) 可能不合理", batch_size,
                              "建議使用 16-64 之間的值")
                              
            if learning_rate <= 0 or learning_rate > 0.01:
                self._add_issue("warning", "value", "enhanced_transformer_config.json",
                              f"learning_rate ({learning_rate}) 可能不合理", learning_rate,
                              "建議使用 0.0001-0.001 之間的值")
                              
        except Exception as e:
            self._add_issue("error", "compatibility", "enhanced_transformer_config.json", 
                          f"Transformer 配置檢查失敗: {str(e)}")
            return False
            
        return True
    
    def _check_quantum_strategy_config(self) -> bool:
        """檢查量子策略層配置"""
        try:
            # 檢查基礎配置是否存在
            try:
                from oanda_trading_bot.training_system.common.config import MAX_SYMBOLS_ALLOWED
                if MAX_SYMBOLS_ALLOWED <= 0 or MAX_SYMBOLS_ALLOWED > 50:
                    self._add_issue("warning", "value", "config.py",
                                  f"MAX_SYMBOLS_ALLOWED ({MAX_SYMBOLS_ALLOWED}) 可能不合理", 
                                  MAX_SYMBOLS_ALLOWED, "建議設置為 5-20 之間的值")
                else:
                    logger.info(f"✓ MAX_SYMBOLS_ALLOWED 配置正常: {MAX_SYMBOLS_ALLOWED}")
                    
            except ImportError:
                self._add_issue("error", "value", "config.py",
                              "無法導入 MAX_SYMBOLS_ALLOWED", None, "檢查 config.py 文件")
                return False
                              
        except Exception as e:
            self._add_issue("error", "compatibility", "quantum_strategy", 
                          f"量子策略配置檢查失敗: {str(e)}")
            return False
            
        return True
    
    def _check_progressive_learning_config(self) -> bool:
        """檢查漸進式學習配置"""
        try:
            config_path = self.project_root / "configs" / "enhanced_transformer_config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                progressive = config.get('progressive_learning', {})
                if not progressive.get('enabled', False):
                    self._add_issue("info", "value", "enhanced_transformer_config.json",
                                  "漸進式學習未啟用", False, "考慮啟用以提升學習效果")
                                  
        except Exception as e:
            self._add_issue("warning", "compatibility", "progressive_learning", 
                          f"漸進式學習配置檢查失敗: {str(e)}")
            
        return True
    
    def _check_device_compatibility(self) -> bool:
        """檢查設備兼容性"""
        try:
            # 檢查 CUDA 可用性
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                
                logger.info(f"檢測到 GPU: {gpu_name} (設備 {current_device}/{gpu_count})")
                
                # 檢查 GPU 記憶體
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                if gpu_memory_gb < 4:
                    self._add_issue("warning", "compatibility", "GPU",
                                  f"GPU 記憶體較小 ({gpu_memory_gb:.1f}GB)", gpu_memory_gb,
                                  "考慮減小 batch_size 或使用 gradient_checkpointing")
            else:
                self._add_issue("info", "compatibility", "GPU",
                              "未檢測到 CUDA GPU，將使用 CPU", "CPU",
                              "安裝 CUDA 版本的 PyTorch 以獲得更好性能")
                              
        except Exception as e:
            self._add_issue("warning", "compatibility", "device", 
                          f"設備兼容性檢查失敗: {str(e)}")
            
        return True
    
    def _check_data_consistency(self) -> bool:
        """檢查數據配置一致性"""
        try:
            # 檢查數據目錄
            data_dir = self.project_root / "data"
            if data_dir.exists():
                # 檢查 mmap 數據
                mmap_dirs = list(data_dir.glob("mmap*"))
                if mmap_dirs:
                    logger.info(f"發現 {len(mmap_dirs)} 個 mmap 數據目錄")
                else:
                    self._add_issue("info", "data", "data_directory",
                                  "未發現 mmap 數據目錄", None,
                                  "首次運行時會自動下載和創建數據")
                                  
        except Exception as e:
            self._add_issue("warning", "data", "data_consistency", 
                          f"數據一致性檢查失敗: {str(e)}")
            
        return True
    
    def _check_training_parameters(self) -> bool:
        """檢查訓練參數合理性"""
        try:
            # 檢查配置文件中的訓練參數
            config_path = self.project_root / "configs" / "enhanced_transformer_config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                training = config.get('training', {})
                total_timesteps = training.get('total_timesteps', 2000000)
                
                if total_timesteps < 100000:
                    self._add_issue("warning", "value", "training_parameters",
                                  f"總訓練步數較少 ({total_timesteps})", total_timesteps,
                                  "建議使用至少 500,000 步以獲得良好效果")
                elif total_timesteps > 10000000:
                    self._add_issue("warning", "value", "training_parameters",
                                  f"總訓練步數較多 ({total_timesteps})", total_timesteps,
                                  "考慮分階段訓練或使用早停機制")
                                  
        except Exception as e:
            self._add_issue("warning", "value", "training_parameters", 
                          f"訓練參數檢查失敗: {str(e)}")
            
        return True
    
    def _add_issue(self, level: str, category: str, module: str, issue: str, 
                   current_value: Any = None, fix_suggestion: str = ""):
        """添加問題到列表"""
        self.issues.append(ConfigIssue(
            level=level,
            category=category,
            module=module,
            issue=issue,
            current_value=current_value,
            fix_suggestion=fix_suggestion
        ))
    
    def _save_report(self, report: ValidationReport):
        """保存驗證報告"""
        reports_dir = self.project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"config_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 轉換為可序列化格式
        report_dict = asdict(report)
        report_dict['issues'] = [asdict(issue) for issue in report.issues]
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
        logger.info(f"驗證報告已保存至: {report_file}")
    
    def _print_summary(self, report: ValidationReport):
        """打印驗證摘要"""
        print("\n" + "="*60)
        print("🔍 配置驗證報告摘要")
        print("="*60)
        print(f"總檢查項目: {report.total_checks}")
        print(f"通過檢查: {report.passed_checks}")
        print(f"嚴重錯誤: {report.critical_errors}")
        print(f"警告: {report.warnings}")
        print(f"整體狀態: {report.overall_status.upper()}")
        
        if report.issues:
            print(f"\n📋 發現的問題:")
            for i, issue in enumerate(report.issues, 1):
                icon = "🚨" if issue.level == "error" else "⚠️" if issue.level == "warning" else "ℹ️"
                print(f"\n{i}. {icon} [{issue.level.upper()}] {issue.module}")
                print(f"   問題: {issue.issue}")
                if issue.current_value is not None:
                    print(f"   當前值: {issue.current_value}")
                if issue.fix_suggestion:
                    print(f"   建議: {issue.fix_suggestion}")
        
        print("\n" + "="*60)
    
    def auto_fix(self, report: ValidationReport) -> bool:
        """自動修復一些簡單問題"""
        logger.info("開始自動修復...")
        
        fixed_count = 0
        for issue in report.issues:
            if issue.level == "warning" and issue.category == "path":
                if "目錄" in issue.issue and "不存在" in issue.issue:
                    # 自動創建缺失目錄
                    try:
                        dir_path = Path(str(issue.current_value))
                        dir_path.mkdir(parents=True, exist_ok=True)
                        logger.info(f"✓ 已創建目錄: {dir_path}")
                        fixed_count += 1
                    except Exception as e:
                        logger.error(f"✗ 創建目錄失敗: {e}")
        
        logger.info(f"自動修復完成，共修復 {fixed_count} 個問題")
        return fixed_count > 0


def main():
    """主函數：執行配置驗證"""
    validator = ConfigValidator()
    report = validator.validate_all()
    
    if report.overall_status == "failed":
        print("\n❌ 發現嚴重配置問題，建議修復後再運行系統")
        return False
    elif report.overall_status == "warnings":
        print("\n⚠️ 發現一些配置警告，系統可以運行但建議優化")
        
        # 詢問是否自動修復
        try:
            user_input = input("\n是否嘗試自動修復？(y/n): ").lower().strip()
            if user_input in ['y', 'yes', '是']:
                validator.auto_fix(report)
        except KeyboardInterrupt:
            print("\n用戶取消操作")
            
        return True
    else:
        print("\n✅ 配置驗證通過，系統可以正常運行")
        return True


if __name__ == "__main__":
    main()
