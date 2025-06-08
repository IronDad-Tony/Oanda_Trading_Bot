"""
獎勵標準化器
將原始獎勵值標準化到 -100 到 +100 的範圍，提高可讀性和解釋性
"""

from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import numpy as np
import logging
from .dynamic_reweighting import VolatilityAnalyzer  # 新增導入

logger = logging.getLogger(__name__)

class RewardNormalizer:
    """
    獎勵標準化器
    
    功能：
    1. 將原始獎勵值映射到 -100 到 +100 範圍
    2. 支持動態範圍調整和歷史數據分析
    3. 保持獎勵信號的相對強度和方向性
    4. 提供多種標準化策略
    """
    
    def __init__(self, 
                 target_range: Tuple[float, float] = (-100.0, 100.0),
                 history_window: int = 1000,
                 adaptive_scaling: bool = True):
        """
        初始化獎勵標準化器
        
        Args:
            target_range: 目標範圍 (min, max)
            history_window: 歷史數據窗口大小
            adaptive_scaling: 是否使用自適應縮放
        """
        self.target_min, self.target_max = target_range
        self.target_range = self.target_max - self.target_min
        self.history_window = history_window
        self.adaptive_scaling = adaptive_scaling
        
        # 歷史獎勵數據
        self.reward_history = deque(maxlen=history_window)
        self.component_history = deque(maxlen=history_window)
        
        # 動態範圍統計
        self.current_min = None
        self.current_max = None
        self.running_mean = Decimal('0')
        self.running_std = Decimal('1')
        
        # 標準化策略配置
        self.normalization_config = {
            'method': 'percentile_clipping',  # 'minmax', 'zscore', 'percentile_clipping', 'robust'
            'percentile_range': (5, 95),      # 用於percentile_clipping
            'outlier_threshold': 3.0,         # 異常值閾值
            'smoothing_factor': 0.95,         # 指數平滑係數
            'min_samples': 50,                # 最小樣本數才開始標準化
        }
        
        # 組件權重（用於組件級別標準化）
        self.component_weights = {
            # 階段1組件權重
            'profit_reward': 2.0,
            'loss_penalty': 1.5,
            'trade_frequency': 1.0,
            'exploration': 1.2,
            'concept_mastery': 1.5,
            'hold_profit': 1.3,
            'learning_progress': 1.1,
            
            # 階段2組件權重
            'sortino_ratio': 2.5,
            'sharpe_ratio': 2.0,
            'drawdown_penalty': 2.0,
            'profit_run': 1.5,
            'quick_cut_loss': 1.3,
            'profit_loss_ratio': 1.8,
            'win_rate_penalty': 1.6,
            'trend_following': 1.2,
            
            # 階段3組件權重
            'information_ratio': 3.0,
            'kelly_criterion': 2.5,
            'omega_ratio': 2.3,
            'tail_ratio': 2.0,
            'regime_adaptation': 1.8,
            'volatility_timing': 1.5,
            'behavioral_finance': 1.3,
            'skewness': 1.2,
            'kurtosis': 1.5,
            'unconventional_strategy': 1.0,
        }
        
        logger.info(f"獎勵標準化器已初始化，目標範圍: {target_range}")
    
    def normalize_reward(self, 
                        reward_info: Dict[str, Any], 
                        method: Optional[str] = None) -> Dict[str, Any]:
        """
        標準化獎勵值
        
        Args:
            reward_info: 原始獎勵信息
            method: 標準化方法 ('total', 'component', 'hybrid')
        
        Returns:
            標準化後的獎勵信息
        """
        method = method or 'hybrid'
        
        # 記錄原始數據
        original_total = reward_info['total_reward']
        original_components = reward_info['components'].copy()
        
        self.reward_history.append(original_total)
        self.component_history.append(original_components)
        
        # 更新統計信息
        self._update_statistics()
        
        # 根據方法執行標準化
        if method == 'total':
            normalized_info = self._normalize_total_reward(reward_info)
        elif method == 'component':
            normalized_info = self._normalize_component_wise(reward_info)
        else:  # hybrid
            normalized_info = self._normalize_hybrid(reward_info)
        
        # 添加標準化元信息
        normalized_info['normalization'] = {
            'method': method,
            'original_total': original_total,
            'scale_factor': self._get_current_scale_factor(),
            'statistics': {
                'min': self.current_min,
                'max': self.current_max,
                'mean': float(self.running_mean),
                'std': float(self.running_std),
            }
        }
        
        return normalized_info
    
    def _normalize_total_reward(self, reward_info: Dict[str, Any]) -> Dict[str, Any]:
        """對總獎勵進行標準化"""
        result = reward_info.copy()
        
        if len(self.reward_history) < self.normalization_config['min_samples']:
            # 樣本不足，使用保守標準化
            normalized_total = self._simple_clip(
                reward_info['total_reward'], 
                -5.0, 5.0,  # 假設初始範圍
                self.target_min, self.target_max
            )
        else:
            normalized_total = self._apply_normalization_strategy(
                reward_info['total_reward'],
                list(self.reward_history)
            )
        
        result['total_reward'] = normalized_total
        result['normalized'] = True
        
        return result
    
    def _normalize_component_wise(self, reward_info: Dict[str, Any]) -> Dict[str, Any]:
        """對各組件分別標準化"""
        result = reward_info.copy()
        
        if len(self.component_history) < self.normalization_config['min_samples']:
            # 樣本不足，使用權重比例標準化
            normalized_components = self._weight_based_normalization(
                reward_info['components']
            )
        else:
            normalized_components = self._statistical_component_normalization(
                reward_info['components']
            )
        
        result['components'] = normalized_components
        result['total_reward'] = sum(normalized_components.values())
        result['normalized'] = True
        
        return result
    
    def _normalize_hybrid(self, reward_info: Dict[str, Any]) -> Dict[str, Any]:
        """混合標準化策略"""
        result = reward_info.copy()
        
        # 1. 組件級標準化
        if len(self.component_history) >= self.normalization_config['min_samples']:
            normalized_components = self._statistical_component_normalization(
                reward_info['components']
            )
        else:
            normalized_components = self._weight_based_normalization(
                reward_info['components']
            )
        
        # 2. 總體縮放調整
        component_sum = sum(normalized_components.values())
        
        if len(self.reward_history) >= self.normalization_config['min_samples']:
            # 使用歷史數據進行總體調整
            target_total = self._apply_normalization_strategy(
                reward_info['total_reward'],
                list(self.reward_history)
            )
            
            # 比例調整組件
            if abs(component_sum) > 1e-6:
                scale_factor = target_total / component_sum
                normalized_components = {
                    k: v * scale_factor for k, v in normalized_components.items()
                }
        
        result['components'] = normalized_components
        result['total_reward'] = sum(normalized_components.values())
        result['normalized'] = True
        
        return result
    
    def _apply_normalization_strategy(self, value: float, history: List[float]) -> float:
        """應用指定的標準化策略"""
        method = self.normalization_config['method']
        
        if method == 'minmax':
            return self._minmax_normalize(value, history)
        elif method == 'zscore':
            return self._zscore_normalize(value, history)
        elif method == 'percentile_clipping':
            return self._percentile_clip_normalize(value, history)
        elif method == 'robust':
            return self._robust_normalize(value, history)
        else:
            return self._minmax_normalize(value, history)
    
    def _minmax_normalize(self, value: float, history: List[float]) -> float:
        """Min-Max標準化"""
        hist_min = min(history)
        hist_max = max(history)
        
        if abs(hist_max - hist_min) < 1e-6:
            return 0.0
        
        normalized = (value - hist_min) / (hist_max - hist_min)
        return self.target_min + normalized * self.target_range
    
    def _zscore_normalize(self, value: float, history: List[float]) -> float:
        """Z-score標準化"""
        mean_val = np.mean(history)
        std_val = np.std(history)
        
        if std_val < 1e-6:
            return 0.0
        
        z_score = (value - mean_val) / std_val
        # 將z-score映射到目標範圍（假設±3σ為界）
        clipped_z = np.clip(z_score, -3, 3)
        normalized = clipped_z / 3.0  # [-1, 1]
        
        return self.target_min + (normalized + 1) / 2 * self.target_range
    
    def _percentile_clip_normalize(self, value: float, history: List[float]) -> float:
        """百分位數截斷標準化"""
        p_low, p_high = self.normalization_config['percentile_range']
        
        low_val = np.percentile(history, p_low)
        high_val = np.percentile(history, p_high)
        
        if abs(high_val - low_val) < 1e-6:
            return 0.0
        
        # 截斷並標準化
        clipped_value = np.clip(value, low_val, high_val)
        normalized = (clipped_value - low_val) / (high_val - low_val)
        
        return self.target_min + normalized * self.target_range
    
    def _robust_normalize(self, value: float, history: List[float]) -> float:
        """鲁棒標準化（基於中位數和MAD）"""
        median_val = np.median(history)
        mad = np.median([abs(x - median_val) for x in history])
        
        if mad < 1e-6:
            return 0.0
        
        # 鲁棒z-score
        robust_z = (value - median_val) / (1.4826 * mad)  # 1.4826 是常數
        clipped_z = np.clip(robust_z, -3, 3)
        normalized = clipped_z / 3.0  # [-1, 1]
        
        return self.target_min + (normalized + 1) / 2 * self.target_range
    
    def _weight_based_normalization(self, components: Dict[str, float]) -> Dict[str, float]:
        """基於權重的組件標準化"""
        normalized = {}
        
        for comp_name, comp_value in components.items():
            weight = self.component_weights.get(comp_name, 1.0)
            
            # 根據權重和值大小進行標準化
            if comp_value >= 0:
                # 正獎勵：權重越高，標準化後的值越大
                normalized_value = min(comp_value * weight * 10, self.target_max * 0.3)
            else:
                # 負獎勵：權重越高，懲罰越明顯
                normalized_value = max(comp_value * weight * 10, self.target_min * 0.3)
            
            normalized[comp_name] = normalized_value
        
        return normalized
    
    def _statistical_component_normalization(self, components: Dict[str, float]) -> Dict[str, float]:
        """基於統計信息的組件標準化"""
        normalized = {}
        
        # 收集各組件的歷史數據
        component_stats = self._get_component_statistics()
        
        for comp_name, comp_value in components.items():
            if comp_name in component_stats:
                stats = component_stats[comp_name]
                
                # 使用組件特定的統計信息進行標準化
                if stats['std'] > 1e-6:
                    z_score = (comp_value - stats['mean']) / stats['std']
                    clipped_z = np.clip(z_score, -2, 2)
                    
                    # 根據權重調整範圍
                    weight = self.component_weights.get(comp_name, 1.0)
                    max_range = min(self.target_max * 0.5, 50 * weight)
                    min_range = max(self.target_min * 0.5, -50 * weight)
                    
                    normalized_value = clipped_z / 2.0 * (max_range - min_range) / 2
                else:
                    normalized_value = 0.0
            else:
                # 新組件，使用權重標準化
                weight = self.component_weights.get(comp_name, 1.0)
                normalized_value = comp_value * weight * 5
            
            normalized[comp_name] = normalized_value
        
        return normalized
    
    def _get_component_statistics(self) -> Dict[str, Dict[str, float]]:
        """獲取各組件的統計信息"""
        stats = {}
        
        if not self.component_history:
            return stats
        
        # 收集所有組件名
        all_components = set()
        for comp_dict in self.component_history:
            all_components.update(comp_dict.keys())
        
        # 計算每個組件的統計信息
        for comp_name in all_components:
            values = [
                comp_dict.get(comp_name, 0.0) 
                for comp_dict in self.component_history
            ]
            
            if values:
                stats[comp_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len([v for v in values if abs(v) > 1e-6])
                }
        
        return stats
    
    def _simple_clip(self, value: float, old_min: float, old_max: float, 
                    new_min: float, new_max: float) -> float:
        """簡單線性縮放"""
        if abs(old_max - old_min) < 1e-6:
            return 0.0
        
        # 線性映射
        normalized = (value - old_min) / (old_max - old_min)
        return new_min + normalized * (new_max - new_min)
    
    def _update_statistics(self):
        """更新運行統計信息"""
        if not self.reward_history:
            return
        
        rewards = list(self.reward_history)
        self.current_min = min(rewards)
        self.current_max = max(rewards)
        
        # 指數移動平均
        current_mean = Decimal(str(np.mean(rewards)))
        current_std = Decimal(str(np.std(rewards)))
        
        if len(rewards) == 1:
            self.running_mean = current_mean
            self.running_std = max(current_std, Decimal('0.1'))
        else:
            alpha = Decimal(str(1 - self.normalization_config['smoothing_factor']))
            self.running_mean = (alpha * current_mean + 
                               (Decimal('1') - alpha) * self.running_mean)
            self.running_std = (alpha * current_std + 
                              (Decimal('1') - alpha) * self.running_std)
            self.running_std = max(self.running_std, Decimal('0.1'))
    
    def _get_current_scale_factor(self) -> float:
        """獲取當前縮放因子"""
        if self.current_min is None or self.current_max is None:
            return 1.0
        
        current_range = self.current_max - self.current_min
        if current_range < 1e-6:
            return 1.0
        
        return self.target_range / current_range
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """獲取標準化統計信息"""
        if not self.reward_history:
            return {}
        
        rewards = list(self.reward_history)
        component_stats = self._get_component_statistics()
        
        return {
            'total_rewards': {
                'count': len(rewards),
                'min': min(rewards),
                'max': max(rewards),
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'range': max(rewards) - min(rewards),
            },
            'component_statistics': component_stats,
            'normalization_config': self.normalization_config.copy(),
            'target_range': (self.target_min, self.target_max),
            'current_scale_factor': self._get_current_scale_factor(),
        }
    
    def update_config(self, **kwargs):
        """更新標準化配置"""
        for key, value in kwargs.items():
            if key in self.normalization_config:
                self.normalization_config[key] = value
                logger.info(f"標準化配置已更新：{key} = {value}")
    
    def reset_history(self):
        """重置歷史數據"""
        self.reward_history.clear()
        self.component_history.clear()
        self.current_min = None
        self.current_max = None
        self.running_mean = Decimal('0')
        self.running_std = Decimal('1')
        logger.info("獎勵標準化器歷史數據已重置")

class DynamicRewardNormalizer(RewardNormalizer):
    """
    動態獎勵標準化器（帶波動率感知）
    
    功能擴展：
    1. 集成實時波動率分析模組
    2. 實現基於市場波動的動態權重調整
    """
    
    def __init__(self, volatility_window=100):
        """
        初始化動態獎勵標準化器
        
        Args:
            volatility_window: 波動率分析窗口大小
        """
        super().__init__()
        self.volatility_analyzer = VolatilityAnalyzer(window=volatility_window)
        self.base_weights = self.component_weights.copy()  # 保存基礎權重用於恢復
        logger.info(f"動態獎勵標準化器已初始化，波動率窗口: {volatility_window}")
    
    def update_weights(self, market_state):
        """
        基於波動率分析動態調整權重
        
        Args:
            market_state: 包含市場狀態信息的字典
        """
        # 計算當前波動率
        volatility = self.volatility_analyzer.calculate(market_state)
        
        # 基於波動率調整權重
        if volatility > 0.015:  # 高波動
            self._apply_high_volatility_weights()
        elif volatility < 0.005:  # 低波動
            self._apply_low_volatility_weights()
        else:  # 中等波動
            self._reset_to_base_weights()
        
        logger.info(f"基於波動率 {volatility:.4f} 更新權重配置")
    
    def _apply_high_volatility_weights(self):
        """應用高波動環境權重配置"""
        # 增加風險控制組件權重
        self.component_weights['drawdown_penalty'] = self.base_weights['drawdown_penalty'] * 2.0
        self.component_weights['quick_cut_loss'] = self.base_weights['quick_cut_loss'] * 1.8
        self.component_weights['volatility_timing'] = self.base_weights['volatility_timing'] * 1.5
        
        # 減少風險較高組件權重
        self.component_weights['profit_run'] = self.base_weights['profit_run'] * 0.7
        self.component_weights['trend_following'] = self.base_weights['trend_following'] * 0.8
    
    def _apply_low_volatility_weights(self):
        """應用低波動環境權重配置"""
        # 增加趨勢跟隨組件權重
        self.component_weights['trend_following'] = self.base_weights['trend_following'] * 1.5
        self.component_weights['hold_profit'] = self.base_weights['hold_profit'] * 1.3
        
        # 減少風險控制組件權重
        self.component_weights['drawdown_penalty'] = self.base_weights['drawdown_penalty'] * 0.7
        self.component_weights['quick_cut_loss'] = self.base_weights['quick_cut_loss'] * 0.8
    
    def _reset_to_base_weights(self):
        """恢復基礎權重配置"""
        self.component_weights = self.base_weights.copy()
