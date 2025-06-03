# Currency Conversion Inconsistency Fix

## 🔍 Problem Identified
The trading system has two different currency conversion implementations:

1. **trading_env.py** - `_get_exchange_rate_to_account_currency()`: Simple conversion without OANDA markup
2. **currency_manager.py** - `CurrencyDependencyManager`: Advanced conversion with 0.5% markup

## 🎯 Recommended Solution

### Option 1: Use CurrencyDependencyManager (RECOMMENDED)
Replace the trading_env.py currency conversion with the sophisticated currency manager:

```python
# In trading_env.py __init__()
from src.data_manager.currency_manager import CurrencyDependencyManager
self.currency_manager = CurrencyDependencyManager(ACCOUNT_CURRENCY, apply_oanda_markup=True)

# Replace _get_exchange_rate_to_account_currency() method
def _get_exchange_rate_to_account_currency(self, from_currency: str, current_prices_map: Dict[str, Tuple[Decimal, Decimal]]) -> Decimal:
    return self.currency_manager.convert_to_account_currency(from_currency, current_prices_map, is_credit=True)
```

### Option 2: Add Markup to Existing Implementation
Add 0.5% markup to the existing trading_env.py implementation:

```python
def _get_exchange_rate_to_account_currency(self, from_currency: str, current_prices_map: Dict[str, Tuple[Decimal, Decimal]]) -> Decimal:
    # ... existing logic ...
    if rate and rate > 0:
        # Apply OANDA 0.5% markup for currency conversion
        return rate * Decimal('0.995')  # Slightly unfavorable rate
    # ... rest of method ...
```

## ✅ Benefits of Fix
- **Consistency**: Single source of truth for currency conversion
- **OANDA Compliance**: Proper 0.5% markup application
- **Accuracy**: More realistic P&L calculations
- **Robustness**: Better fallback mechanisms
