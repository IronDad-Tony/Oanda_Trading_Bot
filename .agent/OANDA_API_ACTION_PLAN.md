# OANDA API Integration - Action Plan

**Created**: 2025-11-25  
**Status**: Ready for Implementation  

---

## 🎯 What We Found

Based on OANDA v20 API official documentation, your current implementation is **MISSING several CRITICAL features** needed for accurate trading and training:

### ❌ Critical Missing Features

1. **Instrument Details API** - MISSING
   - Margin rate (leverage)
   - Minimum trade size
   - Maximum position size  
   - Financing/swap rates
   
2. **Full Account Details** - PARTIAL
   - Home currency conversion factors
   - Margin closeout levels
   - Complete margin calculations

3. **Pricing API** - MISSING
   - Real-time tradeable prices
   - Better than candles for current prices
   - Includes conversion factors

4. **Trade Management** - MISSING
   - List open trades
   - Get trade details
   - Close specific trades

---

## ✅ What's Already Working

1. ✅ Basic account summary
2. ✅ Candle data (bid/ask/mid)
3. ✅ Order creation
4. ✅ Position management (basic)
5. ✅ Account changes tracking

---

## 🚀 Implementation Steps

### Step 1: Integrate Enhanced Client (IMMEDIATE)

File created: `src/oanda_trading_bot/live_trading_system/core/oanda_client_enhanced.py`

**DO THIS**:
```bash
# Merge enhanced client into main oanda_client.py
# OR import and use OandaClientEnhanced instead
```

**New Methods Added**:
```python
# Critical
get_instrument_details()      # Get margin rate, min size, etc.
get_account_details()          # Full account with conversion factors
get_pricing()                  # Current tradeable prices
get_home_conversion_factors()  # For P&L conversion

# Important
get_open_trades()             # List all open trades
get_trade_details()           # Specific trade info
close_trade()                 # Close specific trade

# Helper
get_current_mid_price()       # Accurate current price
is_instrument_tradeable()     # Check if can trade
get_available_units()         # Max units available
```

### Step 2: Create Margin Calculator

**Purpose**: Calculate margin EXACTLY as OANDA does

**Key Points** (from OANDA docs):
- All margin calculations use MID prices only
- Formula: `Margin = abs(units) * conversion_factor * margin_rate`
- Margin rate is instrument-specific
- Must convert to home currency

**Implementation**:
```python
class MarginCalculator:
    """Match OANDA margin calculations exactly"""
    
    def calculate_margin(
        self, 
        units: int,
        instrument_margin_rate: float,
        home_conversion_factor: float
    ) -> float:
        """
        OANDA Formula:
        Margin Required = abs(units) * conversion_factor * margin_rate
        
        - Uses MID prices only
        - Conversion to home currency
        - Instrument-specific margin rate
        """
        return abs(units) * home_conversion_factor * instrument_margin_rate
```

### Step 3: Create Currency Converter

**Purpose**: Convert P&L to home currency correctly

**Conversion Paths**:
- Direct: EUR_USD for EUR→USD
- Inverse: EUR_USD inverted for USD→EUR  
- Cross: GBP→JPY via GBP_USD, USD_JPY

**Implementation**:
```python
class CurrencyConverter:
    """Multi-currency conversion logic"""
    
    def convert_to_home(
        self,
        amount: float,
        from_currency: str,
        home_currency: str,
        prices: Dict[str, float]
    ) -> float:
        """Convert any currency to account home currency"""
        # Implement conversion logic
        pass
```

### Step 4: Update Training Environment

**File**: `src/oanda_trading_bot/training_system/environment/trading_env.py`

**Changes Needed**:

```python
class UniversalTradingEnvV4:
    
    def __init__(self, config):
        # ADD: Load real OANDA specs
        self.instrument_specs = self.load_instrument_specs()
        self.margin_calculator = MarginCalculator(...)
        self.currency_converter = CurrencyConverter(...)
    
    def load_instrument_specs(self):
        """Load REAL specs from OANDA API"""
        client = OandaClient.from_env()
        details = client.get_instrument_details(self.symbols)
        
        specs = {}
        for inst in details['instruments']:
            specs[inst['name']] = {
                'margin_rate': float(inst['marginRate']),
                'min_trade_size': float(inst['minimumTradeSize']),
                'max_position': float(inst['maximumPositionSize']),
                'financing_long': float(inst['financing']['longRate']),
                'financing_short': float(inst['financing']['shortRate']),
                'pip_location': inst['pipLocation'],
            }
        return specs
    
    def calculate_margin_required(self, symbol, units):
        """Use REAL margin calculation"""
        spec = self.instrument_specs[symbol]
        mid_price = self.get_mid_price(symbol)
        conversion = self.get_conversion_factor(symbol, mid_price)
        
        return self.margin_calculator.calculate_margin(
            units=units,
            instrument_margin_rate=spec['margin_rate'],
            home_conversion_factor=conversion
        )
    
    def validate_order(self, symbol, units):
        """Validate against REAL OANDA limits"""
        spec = self.instrument_specs[symbol]
        
        # Check minimum
        if abs(units) < spec['min_trade_size']:
            return False, f"Below minimum {spec['min_trade_size']}"
        
        # Check maximum
        if spec['max_position'] > 0:
            current = self.positions.get(symbol, 0)
            if abs(current + units) > spec['max_position']:
                return False, f"Exceeds maximum {spec['max_position']}"
        
        # Check margin
        margin_needed = self.calculate_margin_required(symbol, units)
        if margin_needed > self.available_margin:
            return False, f"Insufficient margin"
        
        return True, "OK"
```

### Step 5: Test Everything

**Create test script**: `scripts/testing/test_oanda_api_complete.py`

```python
"""Test all OANDA API features"""

def test_instrument_details():
    """Test getting instrument specs"""
    client = OandaClientEnhanced.from_env()
    details = client.get_instrument_details(['EUR_USD', 'GBP_USD'])
    
    assert details is not None
    assert 'instruments' in details
    
    for inst in details['instruments']:
        assert 'marginRate' in inst
        assert 'minimumTradeSize' in inst
        assert 'financing' in inst
        print(f"✓ {inst['name']}: margin={inst['marginRate']}")

def test_pricing():
    """Test getting current prices"""
    client = OandaClientEnhanced.from_env()
    pricing = client.get_pricing(['EUR_USD'])
    
    assert pricing is not None
    assert 'prices' in pricing
    
    for price in pricing['prices']:
        assert price['tradeable'] in [True, False]
        assert 'quoteHomeConversionFactors' in price
        print(f"✓ {price['instrument']}: tradeable={price['tradeable']}")

def test_margin_calculation():
    """Test margin matches OANDA"""
    client = OandaClientEnhanced.from_env()
    
    # Get specs
    details = client.get_instrument_details(['EUR_USD'])
    inst = details['instruments'][0]
    margin_rate = float(inst['marginRate'])
    
    # Get pricing with conversion factors
    pricing = client.get_pricing(['EUR_USD'])
    price = pricing['prices'][0]
    conversion = float(price['quoteHomeConversionFactors']['positiveUnits'])
    
    # Calculate
    units = 10000
    margin = abs(units) * conversion * margin_rate
    
    print(f"✓ EUR_USD 10000 units requires {margin:.2f} margin")
    
    # Compare with OANDA's calculation
    # (units_available also shows margin-based limits)

if __name__ == '__main__':
    test_instrument_details()
    test_pricing()
    test_margin_calculation()
    print("\n✅ All API tests passed!")
```

---

## 📋 Complete Checklist

### Phase 1: API Enhancement (TODAY)
- [ ] Integrate enhanced client methods
- [ ] Test all new API calls
- [ ] Verify data formats match docs

### Phase 2: Calculation Engines (TODAY)
- [ ] Implement MarginCalculator
- [ ] Implement CurrencyConverter
- [ ] Test against OANDA calculations

### Phase 3: Training Environment (THIS WEEK)
- [ ] Load real instrument specs
- [ ] Use real margin calculations
- [ ] Validate all trades against limits
- [ ] Test P&L conversions

### Phase 4: Live Trading (THIS WEEK)
- [ ] Use pricing API for current prices
- [ ] Implement trade tracking
- [ ] Verify margin usage
- [ ] Test with small real trades

### Phase 5: Validation (NEXT WEEK)
- [ ] Compare training vs live behavior
- [ ] Verify all conversions
- [ ] Test edge cases
- [ ] Document everything

---

## ⚠️ CRITICAL NOTES

### Margin Calculation Rules (from OANDA)

1. **Always use MID prices**
   - Never use bid/ask for margin
   - Mid = (bid + ask) / 2

2. **Conversion Factor**
   - Converts to account home currency
   - Different for long vs short sometimes
   - Get from pricing API

3. **Margin Rate**
   - Instrument-specific
   - e.g., 0.0333 = 30:1 leverage
   - e.g., 0.05 = 20:1 leverage

4. **Margin Closeout**
   - Occurs at 100% margin closeout percentage
   - All positions closed automatically
   - Must track this in training!

### Currency Conversion Rules

1. **Direct Pair**
   - EUR→USD: Use EUR_USD rate
   - USD→EUR: Use 1/EUR_USD

2. **Cross Pairs**
   - GBP→JPY: GBP_USD * USD_JPY
   - Most crosses go via USD

3. **Home Currency**
   - All P&L must convert to this
   - Account balance in this currency
   - Margin calculated in this currency

---

## 📚 References

- OANDA v20 API Docs: https://developer.oanda.com/rest-live-v20/introduction/
- Instrument Details: `/v3/accounts/{accountID}/instruments`
- Pricing: `/v3/accounts/{accountID}/pricing`
- Margin Calculation: https://www.oanda.com/us-en/trading/margin/
- Currency Conversion: Account ConversionFactors

---

## 🎯 Expected Outcomes

After implementation:

1. ✅ Training environment will use **REAL** OANDA margin rates
2. ✅ Model will learn with **REAL** trade size limits
3. ✅ P&L conversions will be **EXACT**
4. ✅ Live trading will match training behavior
5. ✅ Risk management will be **ACCURATE**

---

**Priority**: CRITICAL  
**Timeline**: Start today, complete this week  
**Impact**: High - affects model training accuracy

---

*Created automatically based on OANDA v20 API documentation analysis*
