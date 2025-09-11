import sys
import traceback

def main():
    try:
        import importlib
        m = importlib.import_module('oanda_trading_bot.live_trading_system.api.server')
        print('SERVER_IMPORT_OK', hasattr(m, 'app'))
    except Exception as e:
        print('SERVER_IMPORT_FAIL')
        traceback.print_exc()

if __name__ == '__main__':
    main()

