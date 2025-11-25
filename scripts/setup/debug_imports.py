import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    import oanda_trading_bot
    print("Import oanda_trading_bot: SUCCESS")
    from oanda_trading_bot.training_system.trainer.universal_trainer import UniversalTrainer
    print("Import UniversalTrainer: SUCCESS")
except Exception as e:
    print(f"Import FAILED: {e}")
    import traceback
    traceback.print_exc()
