"""
獎勵標準化配置文件
定義不同階段和情況下的標準化策略
"""

# 標準化策略配置
NORMALIZATION_STRATEGIES = {
    'conservative': {
        'method': 'percentile_clipping',
        'percentile_range': (10, 90),
        'outlier_threshold': 2.5,
        'smoothing_factor': 0.98,
        'min_samples': 100,
        'description': '保守策略：使用10-90百分位數截斷，適合初期訓練'
    },
    
    'aggressive': {
        'method': 'minmax',
        'percentile_range': (2, 98),
        'outlier_threshold': 3.5,
        'smoothing_factor': 0.90,
        'min_samples': 30,
        'description': '激進策略：使用min-max標準化，適合快速學習'
    },
    
    'balanced': {
        'method': 'percentile_clipping',
        'percentile_range': (5, 95),
        'outlier_threshold': 3.0,
        'smoothing_factor': 0.95,
        'min_samples': 50,
        'description': '平衡策略：5-95百分位數截斷，通用性最佳'
    },
    
    'robust': {
        'method': 'robust',
        'percentile_range': (5, 95),
        'outlier_threshold': 2.0,
        'smoothing_factor': 0.96,
        'min_samples': 75,
        'description': '鲁棒策略：基於中位數和MAD，抗異常值干擾'
    },
    
    'adaptive': {
        'method': 'zscore',
        'percentile_range': (3, 97),
        'outlier_threshold': 3.0,
        'smoothing_factor': 0.92,
        'min_samples': 40,
        'description': '自適應策略：基於Z-score，動態調整範圍'
    }
}

# 階段特定的標準化配置
STAGE_SPECIFIC_CONFIG = {
    1: {
        'strategy': 'aggressive',
        'target_range': (-100.0, 100.0),
        'component_weights_multiplier': 1.2,
        'description': '階段1：激進標準化，鼓勵探索'
    },
    
    2: {
        'strategy': 'balanced',
        'target_range': (-100.0, 100.0),
        'component_weights_multiplier': 1.0,
        'description': '階段2：平衡標準化，穩定學習'
    },
    
    3: {
        'strategy': 'robust',
        'target_range': (-100.0, 100.0),
        'component_weights_multiplier': 0.8,
        'description': '階段3：鲁棒標準化，精確調優'
    }
}

# 組件權重配置（按重要性排序）
COMPONENT_IMPORTANCE_WEIGHTS = {
    # 核心盈利指標（最高權重）
    'profit_reward': 3.0,
    'loss_penalty': 2.5,
    'sortino_ratio': 3.0,
    'sharpe_ratio': 2.8,
    'information_ratio': 3.2,
    
    # 風險管理指標（高權重）
    'drawdown_penalty': 2.5,
    'max_drawdown_duration_penalty': 2.3,
    'kurtosis': 2.0,
    'tail_ratio': 2.2,
    
    # 交易效率指標（中高權重）
    'profit_loss_ratio': 2.5,
    'win_rate_penalty': 2.0,
    'kelly_criterion': 2.8,
    'omega_ratio': 2.6,
    
    # 策略執行指標（中等權重）
    'profit_run': 2.0,
    'quick_cut_loss': 1.8,
    'hold_profit': 1.9,
    'trend_following': 1.7,
    
    # 學習進度指標（中等權重）
    'concept_mastery': 1.8,
    'learning_progress': 1.6,
    'exploration': 1.5,
    'mistake_recovery': 1.4,
    
    # 高級策略指標（中低權重）
    'regime_adaptation': 1.5,
    'volatility_timing': 1.4,
    'behavioral_finance': 1.3,
    'unconventional_strategy': 1.2,
    
    # 輔助指標（低權重）
    'trade_frequency': 1.0,
    'commission_efficiency': 0.8,
    'skewness': 1.1,
    'first_profit_milestone': 1.0,
    'trade_milestone': 0.9,
}

# 動態調整配置
DYNAMIC_ADJUSTMENT_CONFIG = {
    'performance_threshold': {
        'excellent': 0.8,    # 表現優秀時的閾值
        'good': 0.6,         # 表現良好時的閾值
        'poor': 0.3,         # 表現不佳時的閾值
    },
    
    'adjustment_factors': {
        'excellent': {
            'range_expansion': 1.2,      # 優秀表現時擴大範圍
            'sensitivity_increase': 1.1,  # 增加敏感度
        },
        'good': {
            'range_expansion': 1.0,      # 良好表現時保持範圍
            'sensitivity_increase': 1.0,  # 保持敏感度
        },
        'poor': {
            'range_expansion': 0.8,      # 不佳表現時縮小範圍
            'sensitivity_increase': 0.9,  # 降低敏感度
        }
    },
    
    'volatility_adjustment': {
        'high_volatility_threshold': 2.0,   # 高波動性閾值
        'low_volatility_threshold': 0.5,    # 低波動性閾值
        'high_vol_damping': 0.7,             # 高波動時的阻尼因子
        'low_vol_amplification': 1.3,       # 低波動時的放大因子
    }
}

# 可視化配置
VISUALIZATION_CONFIG = {
    'reward_ranges': {
        'excellent': (70, 100),      # 優秀獎勵範圍
        'good': (30, 70),            # 良好獎勵範圍
        'neutral': (-30, 30),        # 中性獎勵範圍
        'poor': (-70, -30),          # 不佳獎勵範圍
        'terrible': (-100, -70),     # 糟糕獎勵範圍
    },
    
    'color_mapping': {
        'excellent': '#00FF00',      # 綠色
        'good': '#90EE90',           # 淺綠色
        'neutral': '#FFFF00',        # 黃色
        'poor': '#FFA500',           # 橙色
        'terrible': '#FF0000',       # 紅色
    },
    
    'component_display_threshold': 0.1,  # 組件顯示閾值
    'history_display_length': 100,       # 歷史顯示長度
}

def get_strategy_config(strategy_name: str) -> dict:
    """獲取指定策略的配置"""
    return NORMALIZATION_STRATEGIES.get(strategy_name, NORMALIZATION_STRATEGIES['balanced'])

def get_stage_config(stage: int) -> dict:
    """獲取指定階段的配置"""
    return STAGE_SPECIFIC_CONFIG.get(stage, STAGE_SPECIFIC_CONFIG[2])

def get_component_weight(component_name: str) -> float:
    """獲取組件權重"""
    return COMPONENT_IMPORTANCE_WEIGHTS.get(component_name, 1.0)

def adjust_weights_for_stage(stage: int) -> dict:
    """根據階段調整權重"""
    stage_config = get_stage_config(stage)
    multiplier = stage_config.get('component_weights_multiplier', 1.0)
    
    return {
        name: weight * multiplier 
        for name, weight in COMPONENT_IMPORTANCE_WEIGHTS.items()
    }

def get_performance_based_adjustment(performance_score: float) -> dict:
    """根據表現分數獲取調整參數"""
    thresholds = DYNAMIC_ADJUSTMENT_CONFIG['performance_threshold']
    adjustments = DYNAMIC_ADJUSTMENT_CONFIG['adjustment_factors']
    
    if performance_score >= thresholds['excellent']:
        return adjustments['excellent']
    elif performance_score >= thresholds['good']:
        return adjustments['good']
    else:
        return adjustments['poor']
