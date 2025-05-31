import requests
import os
from dotenv import load_dotenv

def verify_oanda_api():
    """Verify OANDA API connectivity"""
    load_dotenv()
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    access_token = os.getenv("OANDA_ACCESS_TOKEN")
    url = f"https://api-fxpractice.oanda.com/v3/accounts/{account_id}/instruments"
    
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        response = requests.get(url, headers=headers)
        print(f"HTTP Status: {response.status_code}")
        print(f"Response (first 100 chars): {response.text[:100]}")
    except Exception as e:
        print(f"Request Exception: {str(e)}")

if __name__ == "__main__":
    verify_oanda_api()