"""
HyperparameterOptimizer - 超參數優化器
使用Optuna進行自動化超參數搜索
"""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.trial import Trial
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Run: pip install optuna")


@dataclass
class HPOConfig:
    """超參數優化配置"""
    study_name: str = "trading_bot_hpo"
    n_trials: int = 100
    timeout_hours: Optional[float] = 24.0
    n_jobs: int = 1  # 並行數
    direction: str = "maximize"  # maximize or minimize
    
    # 搜索空間
    search_space: Dict[str, Dict] = field(default_factory=dict)
    
    # 儲存路徑
    storage_path: Optional[Path] = None


@dataclass
class HPOResult:
    """優化結果"""
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    study_name: str
    optimization_time_seconds: float
    all_trials: List[Dict] = field(default_factory=list)


class HyperparameterOptimizer:
    """
    超參數優化器
    
    功能：
    1. 定義搜索空間
    2. 使用TPE採樣
    3. 早停 (Median Pruning)
    4. 結果持久化
    """
    
    # 預設搜索空間
    DEFAULT_SEARCH_SPACE = {
        # SAC Agent參數
        "learning_rate": {
            "type": "loguniform",
            "low": 1e-5,
            "high": 1e-2
        },
        "batch_size": {
            "type": "categorical",
            "choices": [32, 64, 128, 256]
        },
        "gamma": {
            "type": "uniform",
            "low": 0.95,
            "high": 0.999
        },
        "tau": {
            "type": "loguniform",
            "low": 0.001,
            "high": 0.05
        },
        "buffer_size_factor": {
            "type": "int",
            "low": 1000,
            "high": 10000
        },
        
        # Transformer參數
        "hidden_dim": {
            "type": "categorical",
            "choices": [256, 512, 1024]
        },
        "num_layers": {
            "type": "int",
            "low": 4,
            "high": 24
        },
        "num_heads": {
            "type": "categorical",
            "choices": [4, 8, 16]
        },
        "dropout_rate": {
            "type": "uniform",
            "low": 0.0,
            "high": 0.3
        },
        
        # 獎勵系統參數
        "reward_scaling": {
            "type": "loguniform",
            "low": 0.1,
            "high": 10.0
        },
    }
    
    def __init__(self, config: Optional[HPOConfig] = None):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")
        
        self.config = config or HPOConfig()
        self.study: Optional[optuna.Study] = None
        self._objective_fn: Optional[Callable] = None
        
        # 設置搜索空間
        if not self.config.search_space:
            self.config.search_space = self.DEFAULT_SEARCH_SPACE
    
    def create_study(self) -> optuna.Study:
        """創建Optuna study"""
        storage = None
        if self.config.storage_path:
            storage = f"sqlite:///{self.config.storage_path}"
        
        self.study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5
            ),
            storage=storage,
            load_if_exists=True
        )
        
        logger.info(f"Created study: {self.config.study_name}")
        return self.study
    
    def suggest_params(self, trial: Trial) -> Dict[str, Any]:
        """根據搜索空間建議參數"""
        params = {}
        
        for name, space in self.config.search_space.items():
            param_type = space["type"]
            
            if param_type == "uniform":
                params[name] = trial.suggest_float(
                    name, space["low"], space["high"]
                )
            elif param_type == "loguniform":
                params[name] = trial.suggest_float(
                    name, space["low"], space["high"], log=True
                )
            elif param_type == "int":
                params[name] = trial.suggest_int(
                    name, space["low"], space["high"]
                )
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(
                    name, space["choices"]
                )
        
        return params
    
    def set_objective(self, objective_fn: Callable[[Dict[str, Any]], float]):
        """
        設置目標函數
        
        Args:
            objective_fn: 接收參數字典，返回評估分數的函數
        """
        self._objective_fn = objective_fn
    
    def _wrapped_objective(self, trial: Trial) -> float:
        """包裝的目標函數"""
        if self._objective_fn is None:
            raise ValueError("Objective function not set. Call set_objective first.")
        
        params = self.suggest_params(trial)
        
        try:
            score = self._objective_fn(params)
            
            # 記錄中間結果
            for name, value in params.items():
                trial.set_user_attr(name, value)
            
            return score
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            raise optuna.TrialPruned()
    
    def optimize(self) -> HPOResult:
        """執行優化"""
        if self.study is None:
            self.create_study()
        
        start_time = datetime.now()
        
        timeout = None
        if self.config.timeout_hours:
            timeout = self.config.timeout_hours * 3600
        
        logger.info(f"Starting optimization: {self.config.n_trials} trials")
        
        self.study.optimize(
            self._wrapped_objective,
            n_trials=self.config.n_trials,
            timeout=timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # 收集所有試驗結果
        all_trials = []
        for t in self.study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                all_trials.append({
                    "number": t.number,
                    "value": t.value,
                    "params": t.params
                })
        
        result = HPOResult(
            best_params=self.study.best_params,
            best_value=self.study.best_value,
            n_trials=len(self.study.trials),
            study_name=self.config.study_name,
            optimization_time_seconds=elapsed,
            all_trials=all_trials
        )
        
        logger.info(f"Optimization complete. Best value: {result.best_value}")
        logger.info(f"Best params: {result.best_params}")
        
        return result
    
    def save_result(self, result: HPOResult, path: Path):
        """保存優化結果"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info(f"Results saved to {path}")
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """獲取最佳參數"""
        if self.study is None:
            return None
        return self.study.best_params
    
    def get_param_importances(self) -> Dict[str, float]:
        """獲取參數重要性"""
        if self.study is None or len(self.study.trials) < 10:
            return {}
        
        try:
            importances = optuna.importance.get_param_importances(self.study)
            return dict(importances)
        except Exception as e:
            logger.warning(f"Could not compute importances: {e}")
            return {}


def create_trading_objective(
    env_factory: Callable,
    model_factory: Callable,
    eval_episodes: int = 10,
    timesteps_per_trial: int = 10000
) -> Callable[[Dict[str, Any]], float]:
    """
    創建交易系統的目標函數
    
    Args:
        env_factory: 環境工廠函數
        model_factory: 模型工廠函數
        eval_episodes: 評估回合數
        timesteps_per_trial: 每次試驗的訓練步數
        
    Returns:
        目標函數
    """
    def objective(params: Dict[str, Any]) -> float:
        # 創建環境
        env = env_factory()
        
        # 創建模型 (使用建議的參數)
        model = model_factory(params)
        
        # 訓練
        model.learn(total_timesteps=timesteps_per_trial)
        
        # 評估
        total_reward = 0
        for _ in range(eval_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
        
        avg_reward = total_reward / eval_episodes
        
        # 清理
        env.close()
        
        return avg_reward
    
    return objective


# 快速啟動函數
def quick_hpo(
    objective_fn: Callable[[Dict[str, Any]], float],
    n_trials: int = 50,
    search_space: Optional[Dict] = None
) -> HPOResult:
    """
    快速超參數優化
    
    Args:
        objective_fn: 目標函數
        n_trials: 試驗次數
        search_space: 搜索空間（可選）
        
    Returns:
        優化結果
    """
    config = HPOConfig(
        n_trials=n_trials,
        search_space=search_space or {}
    )
    
    optimizer = HyperparameterOptimizer(config)
    optimizer.set_objective(objective_fn)
    
    return optimizer.optimize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed. Run: pip install optuna")
    else:
        # 測試
        def dummy_objective(params: Dict[str, Any]) -> float:
            """虛擬目標函數"""
            lr = params.get("learning_rate", 0.001)
            batch = params.get("batch_size", 64)
            return -((lr - 0.001) ** 2 + (batch - 128) ** 2 / 10000)
        
        config = HPOConfig(
            study_name="test_hpo",
            n_trials=10,
            search_space={
                "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
                "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]}
            }
        )
        
        optimizer = HyperparameterOptimizer(config)
        optimizer.set_objective(dummy_objective)
        result = optimizer.optimize()
        
        print(f"\nBest params: {result.best_params}")
        print(f"Best value: {result.best_value}")
