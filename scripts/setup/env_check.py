import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')
print('OANDA_API_KEY present:', bool(os.getenv('OANDA_API_KEY')))
print('OANDA_ACCOUNT_ID present:', bool(os.getenv('OANDA_ACCOUNT_ID')))
print('OANDA_ENVIRONMENT:', os.getenv('OANDA_ENVIRONMENT'))

