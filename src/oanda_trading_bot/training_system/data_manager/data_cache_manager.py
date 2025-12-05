"""
DataCacheManager - 智能數據緩存管理器
實現增量下載、自動清理、緩存驗證

最佳實踐策略：
1. 保留最近N天的數據作為緩存
2. 增量下載缺失的數據
3. 訓練開始時清理不完整/損壞的數據
4. 支持數據版本控制
"""
import logging
import os
import shutil
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import json
import sqlite3

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """緩存配置"""
    # 數據保留策略
    max_cache_days: int = 30          # 最多保留30天的數據
    max_cache_size_gb: float = 10.0   # 最大緩存大小 (GB)
    
    # 清理策略
    auto_clean_on_startup: bool = True    # 啟動時自動清理
    clean_incomplete_data: bool = True     # 清理不完整的數據
    clean_corrupted_data: bool = True      # 清理損壞的數據
    
    # 下載策略
    incremental_download: bool = True      # 增量下載
    validate_before_use: bool = True       # 使用前驗證
    
    # 路徑配置
    cache_dir: Optional[Path] = None


@dataclass
class DataSegment:
    """數據段信息"""
    symbol: str
    granularity: str
    start_time: datetime
    end_time: datetime
    row_count: int
    file_path: Optional[str] = None
    checksum: Optional[str] = None
    is_complete: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_validated: Optional[datetime] = None


class DataCacheManager:
    """
    智能數據緩存管理器
    
    功能：
    1. 數據段追蹤與管理
    2. 增量下載支持
    3. 自動清理過期數據
    4. 數據完整性驗證
    5. 緩存統計
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # 設置緩存目錄
        if self.config.cache_dir is None:
            from oanda_trading_bot.training_system.common.config import DATABASE_DIR
            self.config.cache_dir = DATABASE_DIR
        
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 緩存元數據數據庫
        self._meta_db_path = self.cache_dir / "cache_metadata.db"
        self._init_metadata_db()
        
        logger.info(f"DataCacheManager initialized. Cache dir: {self.cache_dir}")
    
    def _init_metadata_db(self):
        """初始化元數據數據庫"""
        with sqlite3.connect(self._meta_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    granularity TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    row_count INTEGER,
                    file_path TEXT,
                    checksum TEXT,
                    is_complete INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_validated TEXT,
                    UNIQUE(symbol, granularity, start_time, end_time)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_granularity 
                ON data_segments(symbol, granularity)
            """)
            conn.commit()
    
    def get_cached_segments(
        self, 
        symbol: str, 
        granularity: str
    ) -> List[DataSegment]:
        """獲取已緩存的數據段"""
        segments = []
        
        with sqlite3.connect(self._meta_db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM data_segments 
                WHERE symbol = ? AND granularity = ?
                ORDER BY start_time
            """, (symbol, granularity))
            
            for row in cursor:
                segments.append(DataSegment(
                    symbol=row['symbol'],
                    granularity=row['granularity'],
                    start_time=datetime.fromisoformat(row['start_time']),
                    end_time=datetime.fromisoformat(row['end_time']),
                    row_count=row['row_count'] or 0,
                    file_path=row['file_path'],
                    checksum=row['checksum'],
                    is_complete=bool(row['is_complete']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_validated=datetime.fromisoformat(row['last_validated']) if row['last_validated'] else None
                ))
        
        return segments
    
    def find_missing_ranges(
        self,
        symbol: str,
        granularity: str,
        requested_start: datetime,
        requested_end: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """
        找出需要下載的時間範圍
        
        Returns:
            缺失的時間範圍列表 [(start, end), ...]
        """
        cached = self.get_cached_segments(symbol, granularity)
        
        if not cached:
            return [(requested_start, requested_end)]
        
        # 合併已覆蓋的範圍
        covered = []
        for seg in cached:
            if seg.is_complete:
                covered.append((seg.start_time, seg.end_time))
        
        if not covered:
            return [(requested_start, requested_end)]
        
        # 排序並合併重疊範圍
        covered.sort(key=lambda x: x[0])
        merged = [covered[0]]
        
        for start, end in covered[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        # 找出缺失範圍
        missing = []
        current = requested_start
        
        for start, end in merged:
            if current < start:
                missing.append((current, min(start, requested_end)))
            current = max(current, end)
        
        if current < requested_end:
            missing.append((current, requested_end))
        
        return missing
    
    def register_segment(self, segment: DataSegment):
        """註冊數據段"""
        with sqlite3.connect(self._meta_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO data_segments 
                (symbol, granularity, start_time, end_time, row_count, 
                 file_path, checksum, is_complete, created_at, last_validated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                segment.symbol,
                segment.granularity,
                segment.start_time.isoformat(),
                segment.end_time.isoformat(),
                segment.row_count,
                segment.file_path,
                segment.checksum,
                1 if segment.is_complete else 0,
                segment.created_at.isoformat(),
                segment.last_validated.isoformat() if segment.last_validated else None
            ))
            conn.commit()
        
        logger.debug(f"Registered segment: {segment.symbol} {segment.start_time} - {segment.end_time}")
    
    def cleanup_startup(self) -> Dict[str, int]:
        """
        啟動時清理
        
        Returns:
            清理統計 {'incomplete': N, 'expired': N, 'corrupted': N}
        """
        logger.info("Starting cache cleanup...")
        stats = {'incomplete': 0, 'expired': 0, 'corrupted': 0}
        
        if self.config.clean_incomplete_data:
            stats['incomplete'] = self._clean_incomplete()
        
        if self.config.max_cache_days > 0:
            stats['expired'] = self._clean_expired()
        
        if self.config.clean_corrupted_data:
            stats['corrupted'] = self._clean_corrupted()
        
        self._enforce_size_limit()
        
        logger.info(f"Cleanup complete: {stats}")
        return stats
    
    def _clean_incomplete(self) -> int:
        """清理不完整的數據段"""
        count = 0
        
        with sqlite3.connect(self._meta_db_path) as conn:
            cursor = conn.execute("""
                SELECT id, file_path FROM data_segments WHERE is_complete = 0
            """)
            
            for row in cursor:
                if row[1] and Path(row[1]).exists():
                    try:
                        Path(row[1]).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete {row[1]}: {e}")
                count += 1
            
            conn.execute("DELETE FROM data_segments WHERE is_complete = 0")
            conn.commit()
        
        if count > 0:
            logger.info(f"Cleaned {count} incomplete segments")
        
        return count
    
    def _clean_expired(self) -> int:
        """清理過期數據"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.max_cache_days)
        count = 0
        
        with sqlite3.connect(self._meta_db_path) as conn:
            cursor = conn.execute("""
                SELECT id, file_path FROM data_segments 
                WHERE end_time < ?
            """, (cutoff.isoformat(),))
            
            for row in cursor:
                if row[1] and Path(row[1]).exists():
                    try:
                        Path(row[1]).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete {row[1]}: {e}")
                count += 1
            
            conn.execute("DELETE FROM data_segments WHERE end_time < ?", (cutoff.isoformat(),))
            conn.commit()
        
        if count > 0:
            logger.info(f"Cleaned {count} expired segments (older than {self.config.max_cache_days} days)")
        
        return count
    
    def _clean_corrupted(self) -> int:
        """清理損壞的數據"""
        count = 0
        
        with sqlite3.connect(self._meta_db_path) as conn:
            cursor = conn.execute("SELECT id, file_path, checksum FROM data_segments")
            to_delete = []
            
            for row in cursor:
                if row[1] and Path(row[1]).exists():
                    if row[2]:
                        # 驗證校驗和
                        actual_checksum = self._calculate_checksum(Path(row[1]))
                        if actual_checksum != row[2]:
                            to_delete.append(row[0])
                            try:
                                Path(row[1]).unlink()
                            except Exception:
                                pass
                elif row[1]:
                    # 文件不存在
                    to_delete.append(row[0])
            
            for seg_id in to_delete:
                conn.execute("DELETE FROM data_segments WHERE id = ?", (seg_id,))
                count += 1
            
            conn.commit()
        
        if count > 0:
            logger.info(f"Cleaned {count} corrupted segments")
        
        return count
    
    def _enforce_size_limit(self):
        """強制執行大小限制"""
        total_size = self._get_total_cache_size()
        max_bytes = self.config.max_cache_size_gb * 1e9
        
        if total_size <= max_bytes:
            return
        
        logger.info(f"Cache size {total_size/1e9:.2f}GB exceeds limit {self.config.max_cache_size_gb}GB")
        
        # 按時間刪除最舊的數據
        with sqlite3.connect(self._meta_db_path) as conn:
            cursor = conn.execute("""
                SELECT id, file_path FROM data_segments 
                ORDER BY created_at ASC
            """)
            
            for row in cursor:
                if self._get_total_cache_size() <= max_bytes * 0.8:
                    break
                
                if row[1] and Path(row[1]).exists():
                    try:
                        Path(row[1]).unlink()
                    except Exception:
                        pass
                
                conn.execute("DELETE FROM data_segments WHERE id = ?", (row[0],))
            
            conn.commit()
    
    def _get_total_cache_size(self) -> int:
        """獲取緩存總大小 (bytes)"""
        total = 0
        
        # SQLite數據庫大小
        for db_file in self.cache_dir.glob("*.db"):
            total += db_file.stat().st_size
        
        # 其他數據文件
        for data_file in self.cache_dir.rglob("*"):
            if data_file.is_file():
                total += data_file.stat().st_size
        
        return total
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """計算文件校驗和"""
        hasher = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """獲取緩存統計"""
        with sqlite3.connect(self._meta_db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_segments,
                    SUM(CASE WHEN is_complete = 1 THEN 1 ELSE 0 END) as complete_segments,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    MIN(start_time) as earliest_data,
                    MAX(end_time) as latest_data,
                    SUM(row_count) as total_rows
                FROM data_segments
            """)
            row = cursor.fetchone()
        
        return {
            'total_segments': row[0] or 0,
            'complete_segments': row[1] or 0,
            'unique_symbols': row[2] or 0,
            'earliest_data': row[3],
            'latest_data': row[4],
            'total_rows': row[5] or 0,
            'cache_size_gb': self._get_total_cache_size() / 1e9,
            'cache_dir': str(self.cache_dir)
        }
    
    def clear_all(self, confirm: bool = False) -> bool:
        """
        清空所有緩存
        
        Args:
            confirm: 必須為True才會執行
        """
        if not confirm:
            logger.warning("clear_all requires confirm=True")
            return False
        
        logger.warning("Clearing all cache data...")
        
        # 清空數據庫
        with sqlite3.connect(self._meta_db_path) as conn:
            conn.execute("DELETE FROM data_segments")
            conn.commit()
        
        # 不刪除數據庫文件本身，只清理數據文件
        for item in self.cache_dir.iterdir():
            if item.is_dir() and item.name.startswith("mmap_"):
                shutil.rmtree(item, ignore_errors=True)
        
        logger.info("Cache cleared")
        return True


# 全局實例
_cache_manager: Optional[DataCacheManager] = None


def get_cache_manager() -> DataCacheManager:
    """獲取全局緩存管理器"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = DataCacheManager()
    return _cache_manager


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 測試
    manager = DataCacheManager()
    
    # 清理
    stats = manager.cleanup_startup()
    print(f"Cleanup stats: {stats}")
    
    # 統計
    cache_stats = manager.get_cache_stats()
    print(f"Cache stats: {json.dumps(cache_stats, indent=2, default=str)}")
    
    # 查找缺失範圍
    missing = manager.find_missing_ranges(
        symbol="EUR_USD",
        granularity="S5",
        requested_start=datetime.now(timezone.utc) - timedelta(days=7),
        requested_end=datetime.now(timezone.utc)
    )
    print(f"Missing ranges: {missing}")
