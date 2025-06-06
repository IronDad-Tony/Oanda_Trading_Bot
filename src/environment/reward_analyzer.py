"""
獎勵分析和可視化工具
提供獎勵數據的統計分析和可視化功能
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import seaborn as sns
from datetime import datetime
import logging

from .reward_normalization_config import (
    VISUALIZATION_CONFIG, 
    get_component_weight,
    get_performance_based_adjustment
)

logger = logging.getLogger(__name__)

class RewardAnalyzer:
    """獎勵分析器"""
    
    def __init__(self, history_length: int = 1000):
        self.history_length = history_length
        self.reward_history = deque(maxlen=history_length)
        self.analysis_cache = {}
        
    def add_reward_data(self, reward_info: Dict[str, Any]):
        """添加獎勵數據"""
        self.reward_history.append(reward_info)
        # 清除分析緩存
        self.analysis_cache.clear()
    
    def analyze_reward_distribution(self) -> Dict[str, Any]:
        """分析獎勵分佈"""
        if not self.reward_history:
            return {}
        
        # 提取獎勵值
        total_rewards = [r['total_reward'] for r in self.reward_history]
        
        # 基本統計
        stats = {
            'count': len(total_rewards),
            'mean': np.mean(total_rewards),
            'std': np.std(total_rewards),
            'min': np.min(total_rewards),
            'max': np.max(total_rewards),
            'median': np.median(total_rewards),
            'q25': np.percentile(total_rewards, 25),
            'q75': np.percentile(total_rewards, 75),
            'skewness': self._calculate_skewness(total_rewards),
            'kurtosis': self._calculate_kurtosis(total_rewards),
        }
        
        # 範圍分析
        ranges = VISUALIZATION_CONFIG['reward_ranges']
        range_counts = {}
        for range_name, (min_val, max_val) in ranges.items():
            count = sum(1 for r in total_rewards if min_val <= r <= max_val)
            range_counts[range_name] = {
                'count': count,
                'percentage': count / len(total_rewards) * 100
            }
        
        stats['range_distribution'] = range_counts
        
        return stats
    
    def analyze_component_contributions(self) -> Dict[str, Any]:
        """分析組件貢獻度"""
        if not self.reward_history:
            return {}
        
        # 收集所有組件數據
        component_data = defaultdict(list)
        for reward_info in self.reward_history:
            components = reward_info.get('components', {})
            for comp_name, comp_value in components.items():
                component_data[comp_name].append(comp_value)
        
        # 分析每個組件
        component_analysis = {}
        for comp_name, values in component_data.items():
            if values:
                analysis = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'abs_mean': np.mean(np.abs(values)),
                    'contribution_ratio': np.mean(np.abs(values)) / np.sum([np.mean(np.abs(v)) for v in component_data.values()]),
                    'weight': get_component_weight(comp_name),
                    'frequency': len([v for v in values if abs(v) > 0.01]),
                    'frequency_ratio': len([v for v in values if abs(v) > 0.01]) / len(values)
                }
                component_analysis[comp_name] = analysis
        
        # 按貢獻度排序
        sorted_components = sorted(
            component_analysis.items(),
            key=lambda x: x[1]['abs_mean'],
            reverse=True
        )
        
        return {
            'component_details': dict(sorted_components),
            'top_contributors': sorted_components[:10],
            'total_components': len(component_analysis)
        }
    
    def analyze_stage_performance(self) -> Dict[str, Any]:
        """分析不同階段的表現"""
        if not self.reward_history:
            return {}
        
        stage_data = defaultdict(list)
        for reward_info in self.reward_history:
            stage = reward_info.get('stage', 1)
            stage_data[stage].append(reward_info['total_reward'])
        
        stage_analysis = {}
        for stage, rewards in stage_data.items():
            if rewards:
                analysis = {
                    'count': len(rewards),
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'min_reward': np.min(rewards),
                    'max_reward': np.max(rewards),
                    'positive_ratio': len([r for r in rewards if r > 0]) / len(rewards),
                    'performance_score': self._calculate_performance_score(rewards)
                }
                stage_analysis[stage] = analysis
        
        return stage_analysis
    
    def analyze_normalization_effectiveness(self) -> Dict[str, Any]:
        """分析標準化效果"""
        if not self.reward_history:
            return {}
        
        raw_rewards = []
        normalized_rewards = []
        
        for reward_info in self.reward_history:
            if 'raw_reward_info' in reward_info:
                raw_rewards.append(reward_info['raw_reward_info']['total_reward'])
                normalized_rewards.append(reward_info['total_reward'])
        
        if not raw_rewards:
            return {'error': '沒有原始獎勵數據進行比較'}
        
        analysis = {
            'raw_stats': {
                'mean': np.mean(raw_rewards),
                'std': np.std(raw_rewards),
                'min': np.min(raw_rewards),
                'max': np.max(raw_rewards),
                'range': np.max(raw_rewards) - np.min(raw_rewards)
            },
            'normalized_stats': {
                'mean': np.mean(normalized_rewards),
                'std': np.std(normalized_rewards),
                'min': np.min(normalized_rewards),
                'max': np.max(normalized_rewards),
                'range': np.max(normalized_rewards) - np.min(normalized_rewards)
            },
            'effectiveness_metrics': {
                'range_compression_ratio': (np.max(raw_rewards) - np.min(raw_rewards)) / (np.max(normalized_rewards) - np.min(normalized_rewards)),
                'variance_retention': np.var(normalized_rewards) / np.var(raw_rewards),
                'correlation': np.corrcoef(raw_rewards, normalized_rewards)[0, 1],
                'target_range_utilization': (np.max(normalized_rewards) - np.min(normalized_rewards)) / 200.0  # 200 = 100 - (-100)
            }
        }
        
        return analysis
    
    def generate_performance_report(self) -> str:
        """生成性能報告"""
        if not self.reward_history:
            return "沒有獎勵數據可分析"
        
        # 獲取分析結果
        distribution = self.analyze_reward_distribution()
        components = self.analyze_component_contributions()
        stages = self.analyze_stage_performance()
        normalization = self.analyze_normalization_effectiveness()
        
        # 生成報告
        report = []
        report.append("=" * 60)
        report.append("獎勵系統性能分析報告")
        report.append("=" * 60)
        report.append(f"分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"數據樣本數: {len(self.reward_history)}")
        report.append("")
        
        # 獎勵分佈分析
        report.append("1. 獎勵分佈統計")
        report.append("-" * 30)
        if distribution:
            report.append(f"平均獎勵: {distribution['mean']:.2f}")
            report.append(f"獎勵標準差: {distribution['std']:.2f}")
            report.append(f"獎勵範圍: [{distribution['min']:.2f}, {distribution['max']:.2f}]")
            report.append(f"中位數: {distribution['median']:.2f}")
            report.append(f"偏度: {distribution['skewness']:.3f}")
            report.append(f"峰度: {distribution['kurtosis']:.3f}")
            
            report.append("\n獎勵範圍分佈:")
            for range_name, data in distribution['range_distribution'].items():
                report.append(f"  {range_name}: {data['count']} ({data['percentage']:.1f}%)")
        report.append("")
        
        # 組件貢獻分析
        report.append("2. 組件貢獻度分析")
        report.append("-" * 30)
        
        if components and 'top_contributors' in components:
            report.append("前10大貢獻組件:")
            for i, (comp_name, analysis) in enumerate(components['top_contributors'], 1):
                report.append(f"  {i:2d}. {comp_name:<25} | 平均: {analysis['mean']:7.3f} | 貢獻: {analysis['contribution_ratio']*100:5.1f}%")
        report.append("")
        
        # 階段表現分析
        report.append("3. 階段表現分析")
        report.append("-" * 30)
        if stages:
            for stage, analysis in sorted(stages.items()):
                report.append(f"階段 {stage}:")
                report.append(f"  樣本數: {analysis['count']}")
                report.append(f"  平均獎勵: {analysis['mean_reward']:.2f}")
                report.append(f"  正獎勵比例: {analysis['positive_ratio']*100:.1f}%")
                report.append(f"  表現分數: {analysis['performance_score']:.3f}")
                report.append("")
        
        # 標準化效果分析
        if normalization and 'effectiveness_metrics' in normalization:
            report.append("4. 標準化效果分析")
            report.append("-" * 30)
            metrics = normalization['effectiveness_metrics']
            report.append(f"範圍壓縮比: {metrics['range_compression_ratio']:.2f}")
            report.append(f"方差保留率: {metrics['variance_retention']*100:.1f}%")
            report.append(f"相關係數: {metrics['correlation']:.3f}")
            report.append(f"目標範圍利用率: {metrics['target_range_utilization']*100:.1f}%")
            report.append("")
        
        # 總結和建議
        report.append("5. 總結與建議")
        report.append("-" * 30)
        suggestions = self._generate_suggestions(distribution, components, stages, normalization)
        for suggestion in suggestions:
            report.append(f"• {suggestion}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """計算偏度"""
        if len(data) < 3:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        
        skew = np.mean([(x - mean_val) ** 3 for x in data]) / (std_val ** 3)
        return skew
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """計算峰度"""
        if len(data) < 4:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        
        kurt = np.mean([(x - mean_val) ** 4 for x in data]) / (std_val ** 4) - 3
        return kurt
    
    def _calculate_performance_score(self, rewards: List[float]) -> float:
        """計算表現分數"""
        if not rewards:
            return 0.0
        
        # 綜合考慮平均獎勵、穩定性和正獎勵比例
        mean_reward = np.mean(rewards)
        stability = 1.0 / (1.0 + np.std(rewards))
        positive_ratio = len([r for r in rewards if r > 0]) / len(rewards)
        
        # 加權評分
        score = (mean_reward * 0.4 + stability * 0.3 + positive_ratio * 0.3)
        return max(0.0, min(1.0, (score + 100) / 200))  # 標準化到 [0, 1]
    
    def _generate_suggestions(self, 
                            distribution: Dict, 
                            components: Dict, 
                            stages: Dict, 
                            normalization: Dict) -> List[str]:
        """生成改進建議"""
        suggestions = []
        
        # 基於分佈的建議
        if distribution:
            if distribution['std'] > 30:
                suggestions.append("獎勵變動性較大，建議調整標準化策略以降低波動")
            
            if abs(distribution['skewness']) > 1.0:
                suggestions.append(f"獎勵分佈偏度較大 ({distribution['skewness']:.2f})，考慮調整組件權重")
            
            range_dist = distribution.get('range_distribution', {})
            if range_dist.get('terrible', {}).get('percentage', 0) > 20:
                suggestions.append("糟糕獎勵比例過高，需要優化獎勵計算邏輯")
        
        # 基於組件的建議
        if components and 'component_details' in components:
            top_component = next(iter(components['component_details'].items()), None)
            if top_component and top_component[1]['contribution_ratio'] > 0.5:
                suggestions.append(f"單一組件 '{top_component[0]}' 貢獻度過高，考慮平衡組件權重")
        
        # 基於階段的建議
        if stages:
            for stage, analysis in stages.items():
                if analysis['positive_ratio'] < 0.3:
                    suggestions.append(f"階段 {stage} 正獎勵比例過低，需要調整獎勵策略")
        
        # 基於標準化效果的建議
        if normalization and 'effectiveness_metrics' in normalization:
            metrics = normalization['effectiveness_metrics']
            if metrics['target_range_utilization'] < 0.5:
                suggestions.append("目標範圍利用率不足，可考慮調整標準化參數")
            
            if metrics['correlation'] < 0.8:
                suggestions.append("標準化後相關性較低，可能需要重新評估標準化方法")
        
        if not suggestions:
            suggestions.append("獎勵系統表現良好，繼續保持當前配置")
        
        return suggestions

class RewardVisualizer:
    """獎勵可視化器"""
    
    @staticmethod
    def plot_reward_history(reward_history: List[Dict], 
                          title: str = "獎勵歷史",
                          save_path: Optional[str] = None) -> plt.Figure:
        """繪製獎勵歷史圖"""
        if not reward_history:
            return None
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # 提取數據
        steps = [r['step'] for r in reward_history]
        total_rewards = [r['total_reward'] for r in reward_history]
        stages = [r.get('stage', 1) for r in reward_history]
        
        # 總獎勵趨勢
        ax1.plot(steps, total_rewards, alpha=0.7, color='blue')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_title(f"{title} - 總獎勵趨勢")
        ax1.set_ylabel("獎勵值")
        ax1.grid(True, alpha=0.3)
        
        # 移動平均
        if len(total_rewards) > 10:
            window = min(50, len(total_rewards) // 4)
            moving_avg = pd.Series(total_rewards).rolling(window=window).mean()
            ax1.plot(steps, moving_avg, color='red', linewidth=2, label=f'{window}步移動平均')
            ax1.legend()
        
        # 獎勵分佈直方圖
        ax2.hist(total_rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(x=np.mean(total_rewards), color='red', linestyle='--', label=f'平均值: {np.mean(total_rewards):.2f}')
        ax2.set_title("獎勵分佈")
        ax2.set_xlabel("獎勵值")
        ax2.set_ylabel("頻次")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 階段分佈
        stage_counts = pd.Series(stages).value_counts().sort_index()
        ax3.bar(stage_counts.index, stage_counts.values, alpha=0.7, color=['orange', 'purple', 'brown'][:len(stage_counts)])
        ax3.set_title("階段分佈")
        ax3.set_xlabel("階段")
        ax3.set_ylabel("步數")
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_component_analysis(component_analysis: Dict,
                              title: str = "組件貢獻度分析",
                              save_path: Optional[str] = None) -> plt.Figure:
        """繪製組件分析圖"""
        if not component_analysis or 'component_details' not in component_analysis:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        details = component_analysis['component_details']
        
        # 組件貢獻度餅圖（前10）
        top_10 = list(details.items())[:10]
        names = [name for name, _ in top_10]
        contributions = [data['abs_mean'] for _, data in top_10]
        
        ax1.pie(contributions, labels=names, autopct='%1.1f%%', startangle=90)
        ax1.set_title("前10組件貢獻度")
        
        # 組件平均值橫向條形圖
        means = [data['mean'] for _, data in top_10]
        y_pos = np.arange(len(names))
        
        colors = ['green' if m >= 0 else 'red' for m in means]
        ax2.barh(y_pos, means, color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names)
        ax2.set_xlabel("平均獎勵值")
        ax2.set_title("組件平均獎勵值")
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # 組件頻率分析
        frequencies = [data['frequency_ratio'] for _, data in top_10]
        ax3.bar(range(len(names)), frequencies, alpha=0.7, color='purple')
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=45, ha='right')
        ax3.set_ylabel("激活頻率")
        ax3.set_title("組件激活頻率")
        ax3.grid(True, alpha=0.3)
        
        # 組件權重 vs 實際貢獻散點圖
        weights = [data['weight'] for _, data in details.items()]
        actual_contributions = [data['abs_mean'] for _, data in details.items()]
        
        ax4.scatter(weights, actual_contributions, alpha=0.6, color='blue')
        ax4.set_xlabel("設定權重")
        ax4.set_ylabel("實際平均貢獻")
        ax4.set_title("權重 vs 實際貢獻")
        ax4.grid(True, alpha=0.3)
        
        # 添加對角線參考
        max_val = max(max(weights), max(actual_contributions))
        ax4.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='理想線')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
