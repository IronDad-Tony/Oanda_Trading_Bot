# src/utils/config_synchronizer.py
"""
配置同步工具 - 確保所有模組使用統一配置
自動同步 config.py 與 enhanced_transformer_config.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# 確保能導入配置
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.common.config import *
except ImportError:
    print("❌ 無法導入 config.py，請檢查路徑")
    sys.exit(1)


def sync_transformer_config():
    """同步 Transformer 配置到 JSON 文件"""
    
    config_path = Path(__file__).parent.parent.parent / "configs" / "enhanced_transformer_config.json"
    
    # 從 config.py 構建配置
    unified_config = {
        "model": {
            "type": "EnhancedUniversalTradingTransformer",
            "model_dim": TRANSFORMER_MODEL_DIM,
            "num_layers": TRANSFORMER_NUM_LAYERS,
            "num_heads": TRANSFORMER_NUM_HEADS,
            "ffn_dim": TRANSFORMER_FFN_DIM,
            "dropout_rate": TRANSFORMER_DROPOUT_RATE,
            "timesteps": TIMESTEPS,
            "output_dim_per_symbol": TRANSFORMER_OUTPUT_DIM_PER_SYMBOL,
            "use_multi_scale": ENHANCED_TRANSFORMER_USE_MULTI_SCALE,
            "use_cross_time_fusion": ENHANCED_TRANSFORMER_USE_CROSS_TIME_FUSION,
            "multi_scale_kernels": ENHANCED_TRANSFORMER_MULTI_SCALE_KERNELS,
            "time_scales": ENHANCED_TRANSFORMER_TIME_SCALES
        },
        "training": {
            "total_timesteps": TRAINER_DEFAULT_TOTAL_TIMESTEPS,
            "learning_rate": 0.0001,
            "batch_size": 32,
            "buffer_size": 200000,
            "gradient_steps": 2,
            "train_freq": 4,
            "target_update_interval": 1000,
            "save_freq": TRAINER_SAVE_FREQ_STEPS,
            "eval_freq": TRAINER_EVAL_FREQ_STEPS,
            "n_eval_episodes": TRAINER_N_EVAL_EPISODES
        },
        "progressive_learning": {
            "enabled": True,
            "stage_advancement_episodes": 50,
            "reward_threshold_basic": -0.1,
            "reward_threshold_intermediate": 0.05,
            "reward_threshold_advanced": 0.15
        },
        "meta_learning": {
            "enabled": True,
            "adaptation_rate": 0.01,
            "memory_size": 1000,
            "update_frequency": 100
        },
        "quantum_strategies": {
            "enabled": True,
            "num_strategies": 20,
            "strategy_update_frequency": 500
        },
        "early_stopping": {
            "patience": EARLY_STOPPING_PATIENCE,
            "min_delta_percent": EARLY_STOPPING_MIN_DELTA_PERCENT,
            "min_evals": EARLY_STOPPING_MIN_EVALS
        },
        "device": {
            "device": str(DEVICE),
            "use_amp": USE_AMP
        },
        "data": {
            "max_symbols": MAX_SYMBOLS_ALLOWED,
            "default_symbols": DEFAULT_SYMBOLS,
            "granularity": GRANULARITY
        }
    }
    
    # 保存配置
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(unified_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 配置已同步至: {config_path}")
    
    # 驗證維度一致性
    model_dim = unified_config['model']['model_dim']
    num_heads = unified_config['model']['num_heads']
    
    if model_dim % num_heads != 0:
        print(f"❌ 維度不匹配: model_dim({model_dim}) % num_heads({num_heads}) = {model_dim % num_heads}")
        return False
    else:
        print(f"✅ 維度檢查通過: model_dim({model_dim}) / num_heads({num_heads}) = {model_dim // num_heads}")
        return True


def validate_config_consistency():
    """驗證配置一致性"""
    
    print("🔍 檢查配置一致性...")
    
    issues = []
    
    # 檢查基本維度
    if TRANSFORMER_MODEL_DIM % TRANSFORMER_NUM_HEADS != 0:
        issues.append(f"TRANSFORMER_MODEL_DIM({TRANSFORMER_MODEL_DIM}) 必須被 TRANSFORMER_NUM_HEADS({TRANSFORMER_NUM_HEADS}) 整除")
    
    # 檢查 FFN 維度
    expected_ffn = TRANSFORMER_MODEL_DIM * 4
    if TRANSFORMER_FFN_DIM != expected_ffn:
        issues.append(f"TRANSFORMER_FFN_DIM({TRANSFORMER_FFN_DIM}) 應為 MODEL_DIM * 4 = {expected_ffn}")
    
    # 檢查設備配置
    if not hasattr(sys.modules.get('torch', None), 'cuda'):
        issues.append("PyTorch 未正確安裝或導入")
    
    # 檢查路徑
    required_dirs = ['data', 'logs', 'weights', 'configs']
    base_path = Path(__file__).parent.parent.parent
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if not dir_path.exists():
            issues.append(f"目錄不存在: {dir_path}")
    
    if issues:
        print("❌ 發現問題:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        return False
    else:
        print("✅ 所有配置檢查通過")
        return True


def auto_create_directories():
    """自動創建必要目錄"""
    
    base_path = Path(__file__).parent.parent.parent
    required_dirs = [
        'data',
        'logs', 
        'weights',
        'configs',
        'reports',
        'data/database',
        'data/mmap_s5_universal',
        'logs/tensorboard'
    ]
    
    created_dirs = []
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(dir_path))
    
    if created_dirs:
        print(f"✅ 已創建目錄: {len(created_dirs)} 個")
        for dir_path in created_dirs:
            print(f"  - {dir_path}")
    else:
        print("ℹ️ 所有必要目錄已存在")


def main():
    """主函數"""
    print("🔧 OANDA 交易機器人配置同步工具")
    print("="*50)
    
    # 1. 創建必要目錄
    print("\n1. 創建必要目錄...")
    auto_create_directories()
    
    # 2. 驗證配置一致性
    print("\n2. 驗證配置一致性...")
    config_valid = validate_config_consistency()
    
    # 3. 同步配置文件
    print("\n3. 同步配置文件...")
    sync_success = sync_transformer_config()
    
    # 4. 最終狀態
    print("\n" + "="*50)
    if config_valid and sync_success:
        print("🎉 配置同步完成！系統可以運行")
        print("\n接下來可以執行:")
        print("  1. python 啟動完整監控系統.bat")
        print("  2. 或直接運行 Streamlit: streamlit run streamlit_app_complete.py")
        return True
    else:
        print("❌ 配置同步失敗，請檢查錯誤訊息")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
