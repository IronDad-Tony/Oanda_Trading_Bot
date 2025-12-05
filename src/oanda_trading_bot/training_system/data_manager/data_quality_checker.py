"""
DataQualityChecker - 數據品質檢查器
用於驗證和清洗訓練數據
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QualityIssue(Enum):
    """數據品質問題類型"""
    MISSING_VALUES = "missing_values"
    DUPLICATE_TIMESTAMPS = "duplicate_timestamps"
    OUT_OF_ORDER = "out_of_order"
    PRICE_ANOMALY = "price_anomaly"
    VOLUME_ANOMALY = "volume_anomaly"
    GAP_TOO_LARGE = "gap_too_large"
    NEGATIVE_VALUES = "negative_values"
    ZERO_SPREAD = "zero_spread"
    INVERTED_BIDASK = "inverted_bidask"


@dataclass
class QualityReport:
    """數據品質報告"""
    symbol: str
    total_rows: int
    issues: Dict[QualityIssue, int] = field(default_factory=dict)
    issue_details: List[Dict[str, Any]] = field(default_factory=list)
    is_valid: bool = True
    coverage_percent: float = 100.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def add_issue(self, issue_type: QualityIssue, count: int = 1, details: Optional[Dict] = None):
        self.issues[issue_type] = self.issues.get(issue_type, 0) + count
        if details:
            details["issue_type"] = issue_type.value
            self.issue_details.append(details)
        if issue_type in [QualityIssue.MISSING_VALUES, QualityIssue.PRICE_ANOMALY]:
            self.is_valid = False
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "total_rows": self.total_rows,
            "is_valid": self.is_valid,
            "coverage_percent": self.coverage_percent,
            "issues": {k.value: v for k, v in self.issues.items()},
            "issue_count": len(self.issue_details),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


@dataclass
class QualityConfig:
    """品質檢查配置"""
    # 價格異常閾值
    max_price_change_percent: float = 5.0  # 單根K線最大價格變化
    max_gap_seconds: int = 300  # 最大允許的時間間隙 (秒)
    
    # 異常值檢測
    z_score_threshold: float = 4.0  # Z-score閾值
    
    # 最小數據要求
    min_rows_required: int = 100
    min_coverage_percent: float = 95.0
    
    # 自動修復
    auto_fill_missing: bool = True
    auto_remove_duplicates: bool = True
    auto_fix_order: bool = True


class DataQualityChecker:
    """
    數據品質檢查器
    
    功能：
    1. 缺失值檢測
    2. 重複時間戳檢測
    3. 時間順序檢測
    4. 價格異常檢測
    5. 成交量異常檢測
    6. Bid/Ask價格合理性
    7. 自動修復
    """
    
    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()
    
    def check_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        expected_interval_seconds: int = 5
    ) -> QualityReport:
        """
        檢查DataFrame數據品質
        
        Args:
            df: 數據DataFrame
            symbol: 交易品種
            expected_interval_seconds: 預期的時間間隔
            
        Returns:
            品質報告
        """
        report = QualityReport(
            symbol=symbol,
            total_rows=len(df)
        )
        
        if len(df) < self.config.min_rows_required:
            report.is_valid = False
            logger.warning(f"{symbol}: Insufficient data ({len(df)} rows)")
            return report
        
        # 確保有時間列
        time_col = self._find_time_column(df)
        if time_col is None:
            report.is_valid = False
            return report
        
        report.start_time = pd.to_datetime(df[time_col].iloc[0])
        report.end_time = pd.to_datetime(df[time_col].iloc[-1])
        
        # 執行各項檢查
        self._check_missing_values(df, report)
        self._check_duplicates(df, time_col, report)
        self._check_order(df, time_col, report)
        self._check_gaps(df, time_col, expected_interval_seconds, report)
        self._check_price_anomalies(df, report)
        self._check_volume_anomalies(df, report)
        self._check_bidask_validity(df, report)
        self._check_negative_values(df, report)
        
        # 計算覆蓋率
        report.coverage_percent = self._calculate_coverage(
            df, time_col, expected_interval_seconds
        )
        
        if report.coverage_percent < self.config.min_coverage_percent:
            report.is_valid = False
        
        logger.info(f"{symbol}: Quality check complete. Valid={report.is_valid}, Issues={len(report.issues)}")
        
        return report
    
    def _find_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """查找時間列"""
        candidates = ["time", "timestamp", "datetime", "date", "candle_time"]
        for col in candidates:
            if col in df.columns:
                return col
        
        # 檢查是否是索引
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index.name or "index"
        
        return None
    
    def _check_missing_values(self, df: pd.DataFrame, report: QualityReport):
        """檢查缺失值"""
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            report.add_issue(
                QualityIssue.MISSING_VALUES,
                total_missing,
                {"columns": missing_counts[missing_counts > 0].to_dict()}
            )
    
    def _check_duplicates(self, df: pd.DataFrame, time_col: str, report: QualityReport):
        """檢查重複時間戳"""
        duplicates = df.duplicated(subset=[time_col], keep=False)
        dup_count = duplicates.sum()
        
        if dup_count > 0:
            report.add_issue(
                QualityIssue.DUPLICATE_TIMESTAMPS,
                dup_count,
                {"first_duplicate": str(df[duplicates][time_col].iloc[0])}
            )
    
    def _check_order(self, df: pd.DataFrame, time_col: str, report: QualityReport):
        """檢查時間順序"""
        times = pd.to_datetime(df[time_col])
        is_sorted = times.is_monotonic_increasing
        
        if not is_sorted:
            out_of_order = (times.diff() < timedelta(0)).sum()
            report.add_issue(
                QualityIssue.OUT_OF_ORDER,
                out_of_order
            )
    
    def _check_gaps(
        self,
        df: pd.DataFrame,
        time_col: str,
        expected_interval: int,
        report: QualityReport
    ):
        """檢查時間間隙"""
        times = pd.to_datetime(df[time_col])
        gaps = times.diff().dt.total_seconds()
        
        # 找出過大的間隙 (超過預期的20倍)
        large_gaps = gaps > self.config.max_gap_seconds
        gap_count = large_gaps.sum()
        
        if gap_count > 0:
            report.add_issue(
                QualityIssue.GAP_TOO_LARGE,
                gap_count,
                {"max_gap_seconds": float(gaps.max())}
            )
    
    def _check_price_anomalies(self, df: pd.DataFrame, report: QualityReport):
        """檢查價格異常"""
        price_cols = [c for c in df.columns if "close" in c.lower() or "price" in c.lower()]
        
        for col in price_cols:
            if col not in df.columns:
                continue
            
            prices = df[col].astype(float)
            
            # 計算價格變化百分比
            returns = prices.pct_change().abs() * 100
            
            # 找出超過閾值的變化
            anomalies = returns > self.config.max_price_change_percent
            anomaly_count = anomalies.sum()
            
            if anomaly_count > 0:
                report.add_issue(
                    QualityIssue.PRICE_ANOMALY,
                    anomaly_count,
                    {"column": col, "max_change_percent": float(returns.max())}
                )
            
            # Z-score異常檢測
            z_scores = (prices - prices.mean()) / prices.std()
            z_anomalies = (z_scores.abs() > self.config.z_score_threshold).sum()
            
            if z_anomalies > 0:
                report.add_issue(
                    QualityIssue.PRICE_ANOMALY,
                    z_anomalies,
                    {"column": col, "type": "z_score_outlier"}
                )
    
    def _check_volume_anomalies(self, df: pd.DataFrame, report: QualityReport):
        """檢查成交量異常"""
        if "volume" not in df.columns:
            return
        
        volumes = df["volume"].astype(float)
        
        # 檢查零成交量比例
        zero_volume = (volumes == 0).sum()
        zero_ratio = zero_volume / len(volumes)
        
        if zero_ratio > 0.5:
            report.add_issue(
                QualityIssue.VOLUME_ANOMALY,
                zero_volume,
                {"zero_volume_ratio": zero_ratio}
            )
        
        # Z-score異常
        if volumes.std() > 0:
            z_scores = (volumes - volumes.mean()) / volumes.std()
            anomalies = (z_scores.abs() > self.config.z_score_threshold).sum()
            
            if anomalies > 0:
                report.add_issue(
                    QualityIssue.VOLUME_ANOMALY,
                    anomalies,
                    {"type": "z_score_outlier"}
                )
    
    def _check_bidask_validity(self, df: pd.DataFrame, report: QualityReport):
        """檢查Bid/Ask價格合理性"""
        bid_cols = [c for c in df.columns if "bid" in c.lower()]
        ask_cols = [c for c in df.columns if "ask" in c.lower()]
        
        for bid_col in bid_cols:
            # 找對應的ask列
            ask_col = bid_col.replace("bid", "ask")
            if ask_col not in df.columns:
                continue
            
            bid = df[bid_col].astype(float)
            ask = df[ask_col].astype(float)
            
            # 檢查反轉 (bid > ask)
            inverted = (bid > ask).sum()
            if inverted > 0:
                report.add_issue(
                    QualityIssue.INVERTED_BIDASK,
                    inverted,
                    {"columns": f"{bid_col}/{ask_col}"}
                )
            
            # 檢查零點差
            zero_spread = (bid == ask).sum()
            if zero_spread > len(df) * 0.01:  # 超過1%
                report.add_issue(
                    QualityIssue.ZERO_SPREAD,
                    zero_spread
                )
    
    def _check_negative_values(self, df: pd.DataFrame, report: QualityReport):
        """檢查負值"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        price_volume_cols = [c for c in numeric_cols if any(
            x in c.lower() for x in ["price", "bid", "ask", "open", "high", "low", "close", "volume"]
        )]
        
        for col in price_volume_cols:
            negatives = (df[col] < 0).sum()
            if negatives > 0:
                report.add_issue(
                    QualityIssue.NEGATIVE_VALUES,
                    negatives,
                    {"column": col}
                )
    
    def _calculate_coverage(
        self,
        df: pd.DataFrame,
        time_col: str,
        expected_interval: int
    ) -> float:
        """計算數據覆蓋率"""
        times = pd.to_datetime(df[time_col])
        
        if len(times) < 2:
            return 0.0
        
        total_duration = (times.iloc[-1] - times.iloc[0]).total_seconds()
        expected_rows = total_duration / expected_interval
        
        if expected_rows == 0:
            return 100.0
        
        coverage = (len(df) / expected_rows) * 100
        return min(coverage, 100.0)
    
    def fix_issues(
        self,
        df: pd.DataFrame,
        report: QualityReport
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        自動修復問題
        
        Returns:
            (修復後的DataFrame, 修復操作列表)
        """
        fixes = []
        df_fixed = df.copy()
        
        time_col = self._find_time_column(df_fixed)
        
        # 修復重複
        if self.config.auto_remove_duplicates and QualityIssue.DUPLICATE_TIMESTAMPS in report.issues:
            before = len(df_fixed)
            df_fixed = df_fixed.drop_duplicates(subset=[time_col], keep="first")
            fixes.append(f"Removed {before - len(df_fixed)} duplicate rows")
        
        # 修復順序
        if self.config.auto_fix_order and QualityIssue.OUT_OF_ORDER in report.issues:
            df_fixed = df_fixed.sort_values(time_col)
            fixes.append("Sorted by timestamp")
        
        # 填充缺失值
        if self.config.auto_fill_missing and QualityIssue.MISSING_VALUES in report.issues:
            numeric_cols = df_fixed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_fixed[col].isnull().any():
                    df_fixed[col] = df_fixed[col].fillna(method="ffill").fillna(method="bfill")
            fixes.append("Filled missing values with forward/backward fill")
        
        logger.info(f"Applied {len(fixes)} fixes")
        
        return df_fixed, fixes


def validate_training_data(
    data_dict: Dict[str, pd.DataFrame],
    expected_interval_seconds: int = 5
) -> Dict[str, QualityReport]:
    """
    驗證多個品種的訓練數據
    
    Args:
        data_dict: {symbol: DataFrame}
        expected_interval_seconds: 預期間隔
        
    Returns:
        {symbol: QualityReport}
    """
    checker = DataQualityChecker()
    reports = {}
    
    for symbol, df in data_dict.items():
        reports[symbol] = checker.check_dataframe(
            df, symbol, expected_interval_seconds
        )
    
    # 總結
    valid_count = sum(1 for r in reports.values() if r.is_valid)
    logger.info(f"Data validation complete: {valid_count}/{len(reports)} symbols valid")
    
    return reports


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 測試
    df = pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=100, freq="5s"),
        "bid_close": np.random.randn(100) * 0.01 + 1.08,
        "ask_close": np.random.randn(100) * 0.01 + 1.0801,
        "volume": np.random.randint(100, 1000, 100)
    })
    
    checker = DataQualityChecker()
    report = checker.check_dataframe(df, "EUR_USD", 5)
    
    print(f"Valid: {report.is_valid}")
    print(f"Coverage: {report.coverage_percent:.1f}%")
    print(f"Issues: {report.issues}")
