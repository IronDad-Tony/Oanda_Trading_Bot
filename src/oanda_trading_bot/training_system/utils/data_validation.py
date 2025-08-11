# src/utils/data_validation.py
"""
數據驗證工具
用於驗證交易數據的完整性和有效性
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """數據驗證器類"""
    
    def __init__(self):
        """初始化數據驗證器"""
        self.validation_rules = {
            'price': {'min': 0.0, 'max': 1000000.0},
            'volume': {'min': 0.0, 'max': float('inf')},
            'spread': {'min': 0.0, 'max': 1000.0},
        }
    
    def validate_price_data(self, data: np.ndarray) -> Dict[str, Any]:
        """
        驗證價格數據
        
        Args:
            data: 價格數據數組
            
        Returns:
            驗證結果字典
        """
        results = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        if data is None or len(data) == 0:
            results['valid'] = False
            results['issues'].append('數據為空')
            return results
        
        # 檢查NaN值
        nan_count = np.isnan(data).sum()
        if nan_count > 0:
            results['issues'].append(f'包含 {nan_count} 個NaN值')
            if nan_count / len(data) > 0.1:  # 超過10%的NaN值
                results['valid'] = False
        
        # 檢查無限值
        inf_count = np.isinf(data).sum()
        if inf_count > 0:
            results['issues'].append(f'包含 {inf_count} 個無限值')
            results['valid'] = False
        
        # 檢查價格範圍
        valid_data = data[~(np.isnan(data) | np.isinf(data))]
        if len(valid_data) > 0:
            min_price = valid_data.min()
            max_price = valid_data.max()
            
            if min_price < self.validation_rules['price']['min']:
                results['issues'].append(f'最小價格 {min_price} 低於閾值')
                results['valid'] = False
                
            if max_price > self.validation_rules['price']['max']:
                results['issues'].append(f'最大價格 {max_price} 高於閾值')
                results['valid'] = False
            
            results['stats'] = {
                'min': float(min_price),
                'max': float(max_price),
                'mean': float(valid_data.mean()),
                'std': float(valid_data.std()),
                'valid_count': len(valid_data),
                'total_count': len(data)
            }
        
        return results
    
    def validate_dataset(self, dataset: Any) -> Dict[str, Any]:
        """
        驗證數據集
        
        Args:
            dataset: 數據集對象
            
        Returns:
            驗證結果字典
        """
        results = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        try:
            # 檢查數據集長度
            if hasattr(dataset, '__len__'):
                dataset_length = len(dataset)
                results['stats']['length'] = dataset_length
                
                if dataset_length == 0:
                    results['valid'] = False
                    results['issues'].append('數據集為空')
                    return results
            
            # 檢查數據集樣本
            if hasattr(dataset, '__getitem__'):
                try:
                    sample = dataset[0]
                    results['stats']['sample_type'] = type(sample).__name__
                    
                    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                        # 檢查特徵和標籤
                        features, labels = sample[0], sample[1]
                        
                        if hasattr(features, 'shape'):
                            results['stats']['feature_shape'] = features.shape
                        if hasattr(labels, 'shape'):
                            results['stats']['label_shape'] = labels.shape
                            
                except Exception as e:
                    results['issues'].append(f'無法獲取數據集樣本: {str(e)}')
                    results['valid'] = False
            
        except Exception as e:
            results['valid'] = False
            results['issues'].append(f'數據集驗證錯誤: {str(e)}')
        
        return results
    
    def check_data_consistency(self, data1: np.ndarray, data2: np.ndarray) -> bool:
        """
        檢查兩個數據集的一致性
        
        Args:
            data1: 第一個數據集
            data2: 第二個數據集
            
        Returns:
            是否一致
        """
        if data1.shape != data2.shape:
            return False
        
        # 檢查非NaN值的一致性
        valid_mask1 = ~(np.isnan(data1) | np.isinf(data1))
        valid_mask2 = ~(np.isnan(data2) | np.isinf(data2))
        
        if not np.array_equal(valid_mask1, valid_mask2):
            return False
        
        # 檢查有效數據的相關性
        valid_data1 = data1[valid_mask1]
        valid_data2 = data2[valid_mask2]
        
        if len(valid_data1) > 1:
            correlation = np.corrcoef(valid_data1, valid_data2)[0, 1]
            return not np.isnan(correlation) and correlation > 0.5
        
        return True
