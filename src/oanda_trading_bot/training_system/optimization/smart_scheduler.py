# src/optimization/smart_scheduler.py
"""
智能訓練調度器
基於系統性能和訓練進度動態調整學習參數

主要功能：
1. 自適應學習率調度
2. 動態批次大小調整
3. 早停機制優化
4. 訓練進度監控
5. 資源使用優化
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Callable
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

try:
    from oanda_trading_bot.training_system.common.logger_setup import logger
    from oanda_trading_bot.training_system.optimization.performance_optimizer import PerformanceOptimizer
except ImportError:
    logger = logging.getLogger(__name__)


class SmartTrainingScheduler:
    """智能訓練調度器"""
    
    def __init__(self, 
                 model: nn.Module,
                 initial_lr: float = 0.0001,
                 patience_factor: float = 1.5,
                 performance_threshold: float = 0.02):
        self.model = model
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.patience_factor = patience_factor
        self.performance_threshold = performance_threshold
        
        # 訓練歷史
        self.training_history = {
            "losses": [],
            "rewards": [],
            "learning_rates": [],
            "batch_sizes": [],
            "timestamps": [],
            "gpu_utilization": [],
            "training_speed": []
        }
        
        # 早停參數
        self.best_performance = float('-inf')
        self.patience_counter = 0
        self.base_patience = 50
        
        # 自適應參數
        self.lr_reduction_factor = 0.7
        self.lr_increase_factor = 1.1
        self.min_lr = 1e-6
        self.max_lr = 0.01
        
        logger.info("智能訓練調度器初始化完成")
    
    def update(self, 
               current_loss: float,
               current_reward: float,
               training_time: float,
               gpu_utilization: float = None) -> Dict[str, Any]:
        """更新調度器狀態並返回調整建議"""
        
        timestamp = datetime.now()
        
        # 記錄訓練歷史
        self.training_history["losses"].append(current_loss)
        self.training_history["rewards"].append(current_reward)
        self.training_history["learning_rates"].append(self.current_lr)
        self.training_history["timestamps"].append(timestamp)
        self.training_history["training_speed"].append(training_time)
        
        if gpu_utilization is not None:
            self.training_history["gpu_utilization"].append(gpu_utilization)
        
        # 計算性能指標
        performance_metrics = self._calculate_performance_metrics()
        
        # 學習率調整
        lr_adjustment = self._adjust_learning_rate(performance_metrics)
        
        # 早停檢查
        early_stop_decision = self._check_early_stopping(performance_metrics)
        
        # 生成調整建議
        recommendations = self._generate_recommendations(performance_metrics)
        
        return {
            "performance_metrics": performance_metrics,
            "lr_adjustment": lr_adjustment,
            "early_stop": early_stop_decision,
            "recommendations": recommendations,
            "should_save_checkpoint": self._should_save_checkpoint(performance_metrics)
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """計算性能指標"""
        if len(self.training_history["losses"]) < 2:
            return {"improvement_rate": 0.0, "stability": 0.0, "trend": 0.0}
        
        recent_window = min(10, len(self.training_history["losses"]))
        recent_losses = self.training_history["losses"][-recent_window:]
        recent_rewards = self.training_history["rewards"][-recent_window:]
        
        # 改善率計算
        if len(recent_losses) >= 5:
            early_avg = np.mean(recent_losses[:len(recent_losses)//2])
            late_avg = np.mean(recent_losses[len(recent_losses)//2:])
            improvement_rate = (early_avg - late_avg) / max(early_avg, 1e-8)
        else:
            improvement_rate = 0.0
        
        # 穩定性計算（損失方差）
        stability = 1.0 / (1.0 + np.var(recent_losses))
        
        # 趨勢計算
        if len(recent_rewards) >= 3:
            trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
        else:
            trend = 0.0
        
        # 訓練速度
        if len(self.training_history["training_speed"]) >= 5:
            avg_speed = np.mean(self.training_history["training_speed"][-5:])
            speed_stability = 1.0 / (1.0 + np.var(self.training_history["training_speed"][-5:]))
        else:
            avg_speed = 0.0
            speed_stability = 1.0
        
        return {
            "improvement_rate": improvement_rate,
            "stability": stability,
            "trend": trend,
            "avg_training_speed": avg_speed,
            "speed_stability": speed_stability,
            "current_performance": recent_rewards[-1] if recent_rewards else 0.0
        }
    
    def _adjust_learning_rate(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """調整學習率"""
        improvement_rate = metrics["improvement_rate"]
        stability = metrics["stability"]
        trend = metrics["trend"]
        
        old_lr = self.current_lr
        adjustment_factor = 1.0
        reason = "保持不變"
        
        # 學習率調整邏輯
        if improvement_rate < -0.05:  # 性能退化
            adjustment_factor = self.lr_reduction_factor
            reason = "性能退化，降低學習率"
        elif improvement_rate > 0.02 and stability > 0.8:  # 穩定改善
            adjustment_factor = self.lr_increase_factor
            reason = "穩定改善，適度提高學習率"
        elif improvement_rate < 0.005 and stability > 0.9:  # 收斂緩慢
            adjustment_factor = self.lr_reduction_factor
            reason = "收斂緩慢，降低學習率精細調整"
        elif trend < -0.1:  # 獎勵下降趨勢
            adjustment_factor = self.lr_reduction_factor * 0.8
            reason = "獎勵下降趨勢，大幅降低學習率"
        
        # 應用調整
        self.current_lr = np.clip(
            self.current_lr * adjustment_factor,
            self.min_lr,
            self.max_lr
        )
        
        return {
            "old_lr": old_lr,
            "new_lr": self.current_lr,
            "adjustment_factor": adjustment_factor,
            "reason": reason
        }
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """檢查是否應該早停"""
        current_performance = metrics["current_performance"]
        
        if current_performance > self.best_performance + self.performance_threshold:
            self.best_performance = current_performance
            self.patience_counter = 0
            should_stop = False
            reason = "性能提升，重置計數器"
        else:
            self.patience_counter += 1
            
            # 動態調整耐心值
            dynamic_patience = int(self.base_patience * self.patience_factor)
            
            if self.patience_counter >= dynamic_patience:
                should_stop = True
                reason = f"超過耐心值 {dynamic_patience}，建議早停"
            else:
                should_stop = False
                reason = f"等待中 ({self.patience_counter}/{dynamic_patience})"
        
        return {
            "should_stop": should_stop,
            "patience_counter": self.patience_counter,
            "best_performance": self.best_performance,
            "reason": reason
        }
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """生成訓練建議"""
        recommendations = []
        
        # 性能相關建議
        if metrics["improvement_rate"] < -0.1:
            recommendations.append("⚠️ 性能嚴重退化，考慮回退到上一個檢查點")
        elif metrics["improvement_rate"] < 0.001:
            recommendations.append("📈 收斂緩慢，可以考慮調整模型架構或增加正則化")
        
        # 穩定性相關建議
        if metrics["stability"] < 0.5:
            recommendations.append("🌊 訓練不穩定，建議降低學習率或增加批次大小")
        elif metrics["stability"] > 0.95:
            recommendations.append("✅ 訓練非常穩定，可以考慮適度提高學習率")
        
        # 趨勢相關建議
        if metrics["trend"] > 0.1:
            recommendations.append("🚀 獎勵上升趨勢良好，保持當前策略")
        elif metrics["trend"] < -0.1:
            recommendations.append("📉 獎勵下降趨勢，需要調整策略")
        
        # 訓練速度相關建議
        if metrics.get("speed_stability", 1.0) < 0.7:
            recommendations.append("⏱️ 訓練速度不穩定，檢查GPU使用率和記憶體")
        
        return recommendations
    
    def _should_save_checkpoint(self, metrics: Dict[str, float]) -> bool:
        """判斷是否應該保存檢查點"""
        return (
            metrics["current_performance"] > self.best_performance or
            metrics["improvement_rate"] > 0.05 or
            len(self.training_history["losses"]) % 100 == 0
        )
    
    def get_training_summary(self) -> Dict[str, Any]:
        """獲取訓練摘要"""
        if not self.training_history["losses"]:
            return {"status": "no_data"}
        
        total_steps = len(self.training_history["losses"])
        training_duration = (
            self.training_history["timestamps"][-1] - 
            self.training_history["timestamps"][0]
        ).total_seconds() / 3600  # 小時
        
        return {
            "total_steps": total_steps,
            "training_duration_hours": training_duration,
            "best_performance": self.best_performance,
            "current_lr": self.current_lr,
            "avg_loss": np.mean(self.training_history["losses"][-100:]),
            "avg_reward": np.mean(self.training_history["rewards"][-100:]),
            "improvement_trend": self._calculate_overall_trend(),
            "efficiency_score": self._calculate_efficiency_score()
        }
    
    def _calculate_overall_trend(self) -> str:
        """計算整體趨勢"""
        if len(self.training_history["rewards"]) < 10:
            return "insufficient_data"
        
        recent_rewards = self.training_history["rewards"][-50:]
        early_avg = np.mean(recent_rewards[:len(recent_rewards)//2])
        late_avg = np.mean(recent_rewards[len(recent_rewards)//2:])
        
        improvement = (late_avg - early_avg) / max(abs(early_avg), 1e-8)
        
        if improvement > 0.1:
            return "strong_improvement"
        elif improvement > 0.02:
            return "moderate_improvement"
        elif improvement > -0.02:
            return "stable"
        elif improvement > -0.1:
            return "slight_decline"
        else:
            return "significant_decline"
    
    def _calculate_efficiency_score(self) -> float:
        """計算訓練效率評分"""
        if len(self.training_history["losses"]) < 10:
            return 0.0
        
        # 考慮收斂速度、穩定性和資源利用率
        improvement_rate = self._calculate_performance_metrics()["improvement_rate"]
        stability = self._calculate_performance_metrics()["stability"]
        
        # GPU利用率
        if self.training_history["gpu_utilization"]:
            avg_gpu_util = np.mean(self.training_history["gpu_utilization"][-20:])
        else:
            avg_gpu_util = 0.5  # 假設中等利用率
        
        # 效率評分（0-1）
        efficiency = (
            0.4 * min(1.0, max(0.0, improvement_rate * 10)) +  # 改善率
            0.3 * stability +  # 穩定性
            0.3 * avg_gpu_util  # GPU利用率
        )
        
        return efficiency
    
    def save_training_history(self, filepath: str):
        """保存訓練歷史"""
        # 轉換datetime為字符串
        history_to_save = self.training_history.copy()
        history_to_save["timestamps"] = [
            ts.isoformat() for ts in history_to_save["timestamps"]
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"訓練歷史已保存至: {filepath}")


def create_smart_scheduler(model: nn.Module, 
                         config: Dict[str, Any] = None) -> SmartTrainingScheduler:
    """創建智能調度器的便捷函數"""
    default_config = {
        "initial_lr": 0.0001,
        "patience_factor": 1.5,
        "performance_threshold": 0.02
    }
    
    if config:
        default_config.update(config)
    
    return SmartTrainingScheduler(model, **default_config)


if __name__ == "__main__":
    # 測試智能調度器
    class DummyModel(nn.Module):
        def forward(self, x):
            return x
    
    model = DummyModel()
    scheduler = create_smart_scheduler(model)
    
    # 模擬訓練過程
    print("🧠 智能訓練調度器測試")
    print("="*50)
    
    for step in range(20):
        # 模擬訓練數據
        loss = 1.0 - step * 0.03 + np.random.normal(0, 0.05)
        reward = step * 0.1 + np.random.normal(0, 0.02)
        training_time = 0.5 + np.random.normal(0, 0.1)
        gpu_util = 0.8 + np.random.normal(0, 0.1)
        
        result = scheduler.update(loss, reward, training_time, gpu_util)
        
        if step % 5 == 0:
            print(f"\n步驟 {step+1}:")
            print(f"  當前學習率: {result['lr_adjustment']['new_lr']:.6f}")
            print(f"  性能改善率: {result['performance_metrics']['improvement_rate']:.4f}")
            print(f"  調整原因: {result['lr_adjustment']['reason']}")
            if result["recommendations"]:
                print(f"  建議: {result['recommendations'][0]}")
    
    summary = scheduler.get_training_summary()
    print(f"\n📊 訓練摘要:")
    print(f"  總步數: {summary['total_steps']}")
    print(f"  最佳性能: {summary['best_performance']:.4f}")
    print(f"  當前學習率: {summary['current_lr']:.6f}")
    print(f"  整體趨勢: {summary['improvement_trend']}")
    print(f"  效率評分: {summary['efficiency_score']:.3f}")
