#!/usr/bin/env python3
# train_universal_model.py
"""
通用交易模型訓練腳本
使用增強版訓練器進行完整的模型訓練
"""

import sys
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.trainer.enhanced_trainer import EnhancedUniversalTrainer, create_training_time_range
from src.common.logger_setup import logger


def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='通用交易模型訓練')
    
    # 交易symbols
    parser.add_argument('--symbols', nargs='+', 
                       default=['EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD'],
                       help='要交易的貨幣對列表')
    
    # 時間範圍
    parser.add_argument('--days-back', type=int, default=30,
                       help='從現在往前多少天的數據用於訓練')
    
    # 訓練參數
    parser.add_argument('--total-timesteps', type=int, default=50000,
                       help='總訓練步數')
    
    parser.add_argument('--save-freq', type=int, default=2500,
                       help='模型保存頻率')
    
    parser.add_argument('--eval-freq', type=int, default=5000,
                       help='模型評估頻率')
    
    # 模型相關
    parser.add_argument('--model-name', type=str, default='sac_universal_trader',
                       help='模型名稱前綴')
    
    parser.add_argument('--load-model', type=str, default=None,
                       help='加載已有模型路徑（用於斷點續練）')
    
    # 其他參數
    parser.add_argument('--granularity', type=str, default='S5',
                       choices=['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'H1'],
                       help='數據粒度')
    
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='初始資本')
    
    parser.add_argument('--max-episode-steps', type=int, default=None,
                       help='每個episode的最大步數')
    
    return parser.parse_args()


def main():
    """主函數"""
    args = parse_arguments()
    
    logger.info("=" * 80)
    logger.info("通用交易模型訓練開始")
    logger.info("=" * 80)
    
    # 顯示配置
    logger.info(f"訓練配置:")
    logger.info(f"  交易symbols: {args.symbols}")
    logger.info(f"  數據天數: {args.days_back} 天")
    logger.info(f"  數據粒度: {args.granularity}")
    logger.info(f"  總訓練步數: {args.total_timesteps:,}")
    logger.info(f"  保存頻率: {args.save_freq:,} 步")
    logger.info(f"  評估頻率: {args.eval_freq:,} 步")
    logger.info(f"  初始資本: ${args.initial_capital:,.2f}")
    logger.info(f"  模型名稱: {args.model_name}")
    if args.load_model:
        logger.info(f"  加載模型: {args.load_model}")
    
    try:
        # 創建時間範圍
        start_time, end_time = create_training_time_range(days_back=args.days_back)
        logger.info(f"  訓練時間範圍: {start_time} 到 {end_time}")
        
        # 創建訓練器
        trainer = EnhancedUniversalTrainer(
            trading_symbols=args.symbols,
            start_time=start_time,
            end_time=end_time,
            granularity=args.granularity,
            initial_capital=args.initial_capital,
            max_episode_steps=args.max_episode_steps,
            total_timesteps=args.total_timesteps,
            save_freq=args.save_freq,
            eval_freq=args.eval_freq,
            model_name_prefix=args.model_name
        )
        
        # 執行完整訓練流程
        success = trainer.run_full_training_pipeline(load_model_path=args.load_model)
        
        if success:
            logger.info("=" * 80)
            logger.info("🎉 訓練成功完成！")
            logger.info("=" * 80)
            return 0
        else:
            logger.error("=" * 80)
            logger.error("❌ 訓練失敗")
            logger.error("=" * 80)
            return 1
            
    except KeyboardInterrupt:
        logger.info("=" * 80)
        logger.info("⏹️  訓練被用戶中斷")
        logger.info("=" * 80)
        return 1
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"💥 訓練過程中發生錯誤: {e}")
        logger.error("=" * 80)
        logger.exception("詳細錯誤信息:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)