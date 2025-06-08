"""
動態權重調整模組
實現實時波動率感知和權重動態調整算法
"""

import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

class VolatilityAnalyzer:
    """實時波動率分析器"""
    
    def __init__(self, window: int = 100):
        """
        初始化波動率分析器
        
        Args:
            window: 分析窗口大小
        """
        self.window = window
        self.price_history = deque(maxlen=window)
        self.return_history = deque(maxlen=window-1)
        logger.info(f"波動率分析器已初始化，窗口大小: {window}")
    
    def update(self, market_state: dict):
        """
        更新市場狀態
        
        Args:
            market_state: 包含當前價格信息的市場狀態字典
        """
        current_price = market_state.get('price')
        if current_price is None:
            logger.warning("市場狀態缺少價格信息，無法更新波動率")
            return
        
        # 添加新價格並計算回報率
        if len(self.price_history) > 0:
            last_price = self.price_history[-1]
            returns = (current_price - last_price) / last_price
            self.return_history.append(returns)
        
        self.price_history.append(current_price)
    
    def calculate(self, market_state: dict) -> float:
        """
        計算當前波動率
        
        Args:
            market_state: 包含當前價格信息的市場狀態字典
            
        Returns:
            計算得到的波動率值
        """
        self.update(market_state)
        
        if len(self.return_history) < 2:
            return 0.0  # 數據不足時返回0
        
        # 計算標準差作為波動率
        volatility = np.std(list(self.return_history))
        return float(volatility)
    
    def get_volatility_level(self, market_state: dict) -> str:
        """
        獲取波動率級別分類
        
        Args:
            market_state: 市場狀態
            
        Returns:
            波動率級別: 'low', 'medium', 'high'
        """
        volatility = self.calculate(market_state)
        
        if volatility < 0.005:
            return 'low'
        elif volatility < 0.015:
            return 'medium'
        else:
            return 'high'

class DynamicReweightingAlgorithm:
    """動態權重調整算法"""
    
    def __init__(self, base_weights: dict):
        """
        初始化權重調整算法
        
        Args:
            base_weights: 基礎權重配置
        """
        self.base_weights = base_weights
        self.volatility_level = 'medium'
        logger.info("動態權重調整算法已初始化")
    
    def adjust_weights(self, volatility_level: str) -> dict:
        """
        根據波動率級別調整權重
        
        Args:
            volatility_level: 波動率級別 ('low', 'medium', 'high')
            
        Returns:
            調整後的權重配置
        """
        self.volatility_level = volatility_level
        adjusted_weights = self.base_weights.copy()
        
        # 根據波動率級別應用不同調整策略
        if volatility_level == 'low':
            # 低波動環境：增加趨勢跟隨和持有獲利權重
            adjusted_weights['trend_following'] *= 1.5
            adjusted_weights['hold_profit'] *= 1.3
        elif volatility_level == 'high':
            # 高波動環境：增加風險控制和快速止損權重
            adjusted_weights['drawdown_penalty'] *= 2.0
            adjusted_weights['quick_cut_loss'] *= 1.8
            adjusted_weights['volatility_timing'] *= 1.5
        else:  # medium
            # 中等波動環境：平衡配置
            pass
        
        # 確保權重總和不變
        total_base = sum(self.base_weights.values())
        total_adjusted = sum(adjusted_weights.values())
        if total_adjusted > 0:
            scale_factor = total_base / total_adjusted
            for key in adjusted_weights:
                adjusted_weights[key] *= scale_factor
        
        return adjusted_weights