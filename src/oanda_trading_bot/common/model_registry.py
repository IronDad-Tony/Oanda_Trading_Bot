"""
Model Registry - 模型版本控制和管理系統
"""
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import shutil
import logging
import subprocess

logger = logging.getLogger(__name__)


class ModelMetadata:
    """模型元數據"""
    
    def __init__(self, **kwargs):
        self.model_id: str = kwargs.get('model_id', str(uuid.uuid4()))
        self.model_name: str = kwargs.get('model_name', '')
        self.version: str = kwargs.get('version', '1.0.0')
        self.created_at: str = kwargs.get('created_at', datetime.now().isoformat())
        self.created_by: str = kwargs.get('created_by', 'system')
        self.git_commit: Optional[str] = kwargs.get('git_commit')
        
        # 訓練配置
        self.training_config: Dict = kwargs.get('training_config', {})
        
        # 模型架構
        self.model_architecture: Dict = kwargs.get('model_architecture', {})
        
        # 訓練指標
        self.training_metrics: Dict = kwargs.get('training_metrics', {})
        
        # 驗證結果
        self.validation_results: Dict = kwargs.get('validation_results', {})
        
        # 部署狀態
        self.deployment_status: str = kwargs.get('deployment_status', 'development')
        
        # 標籤
        self.tags: List[str] = kwargs.get('tags', [])
        
        # 備註
        self.notes: str = kwargs.get('notes', '')
        
        # 訓練數據範圍
        self.data_start_date: Optional[str] = kwargs.get('data_start_date')
        self.data_end_date: Optional[str] = kwargs.get('data_end_date')
    
    def to_dict(self) -> Dict:
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'version': self.version,
            'created_at': self.created_at,
            'created_by': self.created_by,
            'git_commit': self.git_commit,
            'training_config': self.training_config,
            'model_architecture': self.model_architecture,
            'training_metrics': self.training_metrics,
            'validation_results': self.validation_results,
            'deployment_status': self.deployment_status,
            'tags': self.tags,
            'notes': self.notes,
            'data_start_date': self.data_start_date,
            'data_end_date': self.data_end_date,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


class ModelRegistry:
    """模型倉庫管理器"""
    
    def __init__(self, registry_dir: Optional[Path] = None):
        if registry_dir is None:
            # 使用項目根目錄下的model_registry
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[3]
            registry_dir = project_root / 'model_registry'
        
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # 索引文件
        self.index_file = self.registry_dir / 'index.json'
        self._load_index()
        
        logger.info(f"ModelRegistry initialized at {self.registry_dir}")
    
    def _load_index(self):
        """加載索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self.index = {'models': []}
        else:
            self.index = {'models': []}
            self._save_index()
    
    def _save_index(self):
        """保存索引"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def save_model(
        self,
        model_path: Path,
        metadata: ModelMetadata,
        overwrite: bool = False
    ) -> str:
        """
        保存模型到倉庫
        
        Args:
            model_path: 模型文件路徑
            metadata: 模型元數據
            overwrite: 是否覆蓋已存在的模型
        
        Returns:
            model_id
        """
        model_id = metadata.model_id
        model_dir = self.registry_dir / model_id
        
        if model_dir.exists() and not overwrite:
            logger.warning(f"Model {model_id} already exists, using new ID")
            metadata.model_id = str(uuid.uuid4())
            model_id = metadata.model_id
            model_dir = self.registry_dir / model_id
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 複製模型文件
        model_file = model_dir / 'model.zip'
        try:
            shutil.copy(model_path, model_file)
            logger.info(f"Model file copied to {model_file}")
        except Exception as e:
            logger.error(f"Failed to copy model file: {e}")
            raise
        
        # 保存元數據
        metadata_file = model_dir / 'metadata.json'
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
        
        # 更新索引
        self.index['models'] = [
            m for m in self.index['models']
            if m['model_id'] != model_id
        ]
        
        self.index['models'].append({
            'model_id': model_id,
            'model_name': metadata.model_name,
            'version': metadata.version,
            'created_at': metadata.created_at,
            'deployment_status': metadata.deployment_status,
            'tags': metadata.tags,
        })
        self._save_index()
        
        logger.info(f"Model {model_id} saved to registry")
        return model_id
    
    def load_model(self, model_id: str) -> Optional[Path]:
        """
        加載模型文件路徑
        
        Args:
            model_id: 模型ID
        
        Returns:
            模型文件路徑，如果不存在則返回None
        """
        model_dir = self.registry_dir / model_id
        model_file = model_dir / 'model.zip'
        
        if not model_file.exists():
            logger.error(f"Model file not found: {model_file}")
            return None
        
        return model_file
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """
        獲取模型元數據
        
        Args:
            model_id: 模型ID
        
        Returns:
            模型元數據，如果不存在則返回None
        """
        model_dir = self.registry_dir / model_id
        metadata_file = model_dir / 'metadata.json'
        
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file}")
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ModelMetadata.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None
    
    def list_models(
        self,
        filters: Optional[Dict] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        列出模型
        
        Args:
            filters: 過濾條件
            limit: 返回數量限制
        
        Returns:
            模型列表
        """
        models = self.index.get('models', [])
        
        # 應用過濾器
        if filters:
            if 'status' in filters:
                models = [
                    m for m in models
                    if m.get('deployment_status') == filters['status']
                ]
            if 'tag' in filters:
                models = [
                    m for m in models
                    if filters['tag'] in m.get('tags', [])
                ]
        
        # 按創建時間倒序排序
        models = sorted(
            models,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )
        
        return models[:limit]
    
    def deploy_model(self, model_id: str, environment: str = 'production'):
        """
        部署模型
        
        Args:
            model_id: 模型ID
            environment: 環境 (development/staging/production)
        """
        metadata = self.get_model(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")
        
        # 更新部署狀態
        metadata.deployment_status = environment
        
        # 保存更新後的元數據
        model_dir = self.registry_dir / model_id
        metadata_file = model_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 更新索引
        for m in self.index['models']:
            if m['model_id'] == model_id:
                m['deployment_status'] = environment
                break
        self._save_index()
        
        logger.info(f"Model {model_id} deployed to {environment}")
    
    def compare_models(self, model_ids: List[str]) -> Dict:
        """
        對比多個模型
        
        Args:
            model_ids: 模型ID列表
        
        Returns:
            對比結果
        """
        comparison = {
            'models': [],
            'metrics_comparison': {},
        }
        
        for model_id in model_ids:
            metadata = self.get_model(model_id)
            if metadata:
                comparison['models'].append(metadata.to_dict())
        
        # 提取共同的指標進行對比
        if comparison['models']:
            all_metrics = set()
            for model in comparison['models']:
                all_metrics.update(model.get('training_metrics', {}).keys())
            
            for metric in all_metrics:
                values = {}
                for model in comparison['models']:
                    model_id = model['model_id']
                    value = model.get('training_metrics', {}).get(metric)
                    if value is not None:
                        values[model_id] = value
                comparison['metrics_comparison'][metric] = values
        
        return comparison
    
    def delete_model(self, model_id: str):
        """
        刪除模型
        
        Args:
            model_id: 模型ID
        """
        model_dir = self.registry_dir / model_id
        
        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            return
        
        # 刪除目錄
        shutil.rmtree(model_dir)
        
        # 更新索引
        self.index['models'] = [
            m for m in self.index['models']
            if m['model_id'] != model_id
        ]
        self._save_index()
        
        logger.info(f"Model {model_id} deleted")
    
    @staticmethod
    def get_git_commit() -> Optional[str]:
        """獲取當前Git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            return result.stdout.strip()[:8]  # 取前8個字符
        except Exception:
            return None
