# OANDA API Complete Implementation Guide

**Date**: 2025-11-25  
**Based on**: OANDA v20 API Documentation  

---

## 🎯 Required API Features for Complete Trading System

### Current Status vs Required Features

| Feature | Status | Priority | Notes |
|---------|--------|----------|-------|
| **Price Data** | ✅ Partial | HIGH | Missing mid prices, need volume |
| **Instrument Details** | ❌ Missing | CRITICAL | Need margin rate, min trade size |
| **Account Info** | ✅ Good | HIGH | Has summary, need full details |
| **Positions** | ✅ Basic | HIGH | Has open positions |
| **Trades** | ❌ Missing | HIGH | Need list trades, trade details |
| **Conversion Factors** | ❌ Missing | CRITICAL | For home currency conversion |
| **Margin Calculation** | ❌ Missing | CRITICAL | For proper risk management |
| **Pricing Stream** | ❌ Missing | MEDIUM | For real-time prices |

---

## 📊 Missing Critical Features

### 1. Instrument Details (CRITICAL)

**What's Missing**:
- Margin rate (for leverage calculation)
- Minimum trade size
- Maximum position size
- Display precision
- Financing rates (swap)

**API Endpoint**: `GET /v3/accounts/{accountID}/instruments`

**Why Critical**:
- Training environment needs correct margin requirements
- Model needs to know minimum trade sizes
- Risk management requires leverage information

### 2. Conversion Factors (CRITICAL)

**What's Missing**:
- Home currency conversion rates
- Cross-currency conversion paths

**API Endpoint**: Account details include `ConversionFactor` and `HomeConversionFactors`

**Why Critical**:
- All P&L must convert to account home currency
- Margin calculations use mid-prices in home currency
- Training environment must replicate exact conversion logic

### 3. Pricing API (Important)

**What's Missing**:
- Real-time streaming prices
- Latest tradeable prices
- Price status (tradeable/non-tradeable)

**API Endpoint**: `GET /v3/accounts/{accountID}/pricing`

**Why Important**:
- More accurate than candles for current prices
- Includes tradeable status
- Lower latency than candles

---

## 🔧 Implementation Plan

### Phase 1: Add Critical Missing Features (TODAY)

#### 1.1 Get Instrument Details
```python
def get_instrument_details(self, instruments: List[str]) -> Optional[Dict]:
    """Get detailed specs for instruments including margin rate, min trade size"""
    params = {"instruments": ",".join(instruments)}
    endpoint = accounts.AccountInstruments(self.account_id, params=params)
    return self._request(endpoint)
```

**Returns**:
```json
{
  "instruments": [
    {
      "name": "EUR_USD",
      "type": "CURRENCY",
      "displayName": "EUR/USD",
      "pipLocation": -4,
      "displayPrecision": 5,
      "tradeUnitsPrecision": 0,
      "minimumTradeSize": "1",
      "maximumTrailingStopDistance": "1.00000",
      "minimumTrailingStopDistance": "0.00050",
      "maximumPositionSize": "0",
      "maximumOrderUnits": "100000000",
      "marginRate": "0.0333",  # 30:1 leverage
      "commission": {
        "commission": "0",
        "unitsTraded": "0",
        "minimumCommission": "0"
      },
      "financing": {
        "longRate": "-0.0196",
        "shortRate": "0.0124"
      }
    }
  ]
}
```

#### 1.2 Get Full Account Details
```python
def get_account_details(self) -> Optional[Dict]:
    """Get complete account details including conversion factors"""
    endpoint = accounts.AccountDetails(self.account_id)
    return self._request(endpoint)
```

**Returns**:
```json
{
  "account": {
    "id": "...",
    "currency": "USD",
    "balance": "10000.0000",
    "pl": "0.0000",
    "resettablePL": "0.0000",
    "financing": "-5.1234",
    "commission": "0.0000",
    "dividendAdjustment": "0",
    "guaranteedExecutionFees": "0.0000",
    "marginRate": "0.0333",
    "marginCallEnterTime": null,
    "marginCallExtensionCount": 0,
    "lastMarginCallExtensionTime": null,
    "openTradeCount": 0,
    "openPositionCount": 0,
    "pendingOrderCount": 0,
    "hedgingEnabled": false,
    "unrealizedPL": "0.0000",
    "NAV": "10000.0000",
    "marginUsed": "0.0000",
    "marginAvailable": "10000.0000",
    "positionValue": "0.0000",
    "marginCloseoutUnrealizedPL": "0.0000",
    "marginCloseoutNAV": "10000.0000",
    "marginCloseoutMarginUsed": "0.0000",
    "marginCloseoutPercent": "0.00000",
    "marginCloseoutPositionValue": "0.0000",
    "withdrawalLimit": "10000.0000",
    "marginCallMarginUsed": "0.0000",
    "marginCallPercent": "0.00000",
    "lastTransactionID": "1234"
  }
}
```

#### 1.3 Get Conversion Factors
** Already included in account details above**

Key fields:
- `currency`: Home currency
- `marginRate`: Overall account margin rate

#### 1.4 Get Current Pricing
```python
def get_pricing(self, instruments: List[str]) -> Optional[Dict]:
    """Get current tradeable prices for instruments"""
    params = {"instruments": ",".join(instruments)}
    endpoint = pricing.PricingInfo(self.account_id, params=params)
    return self._request(endpoint)
```

**Returns**:
```json
{
  "prices": [
    {
      "type": "PRICE",
      "instrument": "EUR_USD",
      "time": "2025-11-25T05:59:30.123456789Z",
      "status": "tradeable",
      "tradeable": true,
      "bids": [
        {"price": "1.15150", "liquidity": 10000000}
      ],
      "asks": [
        {"price": "1.15159", "liquidity": 10000000}
      ],
      "closeoutBid": "1.15150",
      "closeoutAsk": "1.15159",
      "quoteHomeConversionFactors": {
        "positiveUnits": "1.00000000",
        "negativeUnits": "1.00000000"
      },
      "unitsAvailable": {
        "default": {
          "long": "299700.00",
          "short": "299700.00"
        },
        "openOnly": {
          "long": "299700.00",
          "short": "299700.00"
        },
        "reduceFirst": {
          "long": "299700.00",
          "short": "299700.00"
        },
        "reduceOnly": {
          "long": "0.00",
          "short": "0.00"
        }
      }
    }
  ]
}
```

#### 1.5 List Trades
```python
def get_open_trades(self) -> Optional[Dict]:
    """Get all open trades"""
    endpoint = trades.OpenTrades(self.account_id)
    return self._request(endpoint)

def get_trade_details(self, trade_id: str) -> Optional[Dict]:
    """Get details of a specific trade"""
    endpoint = trades.TradeDetails(self.account_id, trade_id)
    return self._request(endpoint)
```

### Phase 2: Enhance Training Environment

#### 2.1 Margin Calculation Engine
```python
class MarginCalculator:
    """Calculate margin requirements exactly as OANDA does"""
    
    def __init__(self, account_currency: str, margin_rate: float):
        self.account_currency = account_currency
        self.margin_rate = margin_rate
    
    def calculate_margin_required(
        self, 
        instrument: str,
        units: int,
        conversion_factor: float,
        instrument_margin_rate: float
    ) -> float:
        """
        Calculate margin required for a position
        
        Formula (from OANDA docs):
        Margin = abs(units) * conversion_factor * margin_rate
        
        - Uses MID prices only
        - Conversion factor converts to account home currency
        - Margin rate is instrument-specific or account-default
        """
        return abs(units) * conversion_factor * instrument_margin_rate
```

#### 2.2 Currency Conversion Engine
```python
class CurrencyConverter:
    """Handle multi-step currency conversions"""
    
    def __init__(self, home_currency: str):
        self.home_currency = home_currency
        self.conversion_cache = {}
    
    def get_conversion_path(self, from_currency: str, to_currency: str) -> List[str]:
        """
        Determine conversion path from one currency to another
        
        Direct: EUR -> USD (EUR_USD)
        Inverse: USD -> EUR (EUR_USD inverted)
        Cross: GBP -> JPY (GBP_USD, USD_JPY)
        """
        if from_currency == to_currency:
            return []
        
        # Try direct pair
        direct = f"{from_currency}_{to_currency}"
        inverse = f"{to_currency}_{from_currency}"
        
        # Try via USD (most common)
        via_usd = [
            f"{from_currency}_USD",
            f"USD_{to_currency}"
        ]
        
        return [direct, inverse] + via_usd
    
    def convert(
        self, 
        amount: float,
        from_currency: str,
        to_currency: str,
        prices: Dict[str, float]
    ) -> float:
        """Convert amount from one currency to another"""
        # Implement conversion logic using prices
        pass
```

#### 2.3 Enhanced Trading Environment
```python
class EnhancedTradingEnv:
    """Trading environment with accurate OANDA simulation"""
    
    def __init__(self):
        self.margin_calculator = None
        self.currency_converter = None
        self.instrument_specs = {}
        
    def load_instrument_specs(self, client: OandaClient):
        """Load all instrument specifications from OANDA"""
        instruments = self.config['symbols']
        specs = client.get_instrument_details(instruments)
        
        for inst in specs['instruments']:
            self.instrument_specs[inst['name']] = {
                'margin_rate': float(inst['marginRate']),
                'min_trade_size': float(inst['minimumTradeSize']),
                'max_position': float(inst['maximumPositionSize']),
                'pip_location': inst['pipLocation'],
                'precision': inst['displayPrecision'],
                'financing_long': float(inst['financing']['longRate']),
                'financing_short': float(inst['financing']['shortRate'])
            }
    
    def calculate_position_margin(self, symbol: str, units: int) -> float:
        """Calculate exact margin as OANDA would"""
        spec = self.instrument_specs[symbol]
        
        # Get current mid price
        mid_price = self.get_mid_price(symbol)
        
        # Get conversion factor to home currency
        conversion = self.get_home_conversion_factor(symbol, mid_price)
        
        # Calculate margin
        margin = self.margin_calculator.calculate_margin_required(
            instrument=symbol,
            units=units,
            conversion_factor=conversion,
            instrument_margin_rate=spec['margin_rate']
        )
        
        return margin
    
    def validate_trade(self, symbol: str, units: int) -> Tuple[bool, str]:
        """Validate trade meets OANDA requirements"""
        spec = self.instrument_specs[symbol]
        
        # Check minimum size
        if abs(units) < spec['min_trade_size']:
            return False, f"Units {units} below minimum {spec['min_trade_size']}"
        
        # Check maximum position
        if spec['max_position'] > 0:
            current_pos = self.positions.get(symbol, 0)
            new_pos = current_pos + units
            if abs(new_pos) > spec['max_position']:
                return False, f"Position would exceed maximum {spec['max_position']}"
        
        # Check margin available
        required_margin = self.calculate_position_margin(symbol, units)
        if required_margin > self.account_state['marginAvailable']:
            return False, f"Insufficient margin: need {required_margin}, have {self.account_state['marginAvailable']}"
        
        return True, "OK"
```

---

## 📋 Complete Implementation Checklist

### API Client Enhancements
- [ ] Add `get_instrument_details()` 
- [ ] Add `get_account_details()`
- [ ] Add `get_pricing()`
- [ ] Add `get_open_trades()`
- [ ] Add `get_trade_details()`
- [ ] Add `list_transactions()`
- [ ] Add pricing stream support (optional)

### Training Environment
- [ ] Implement `MarginCalculator`
- [ ] Implement `CurrencyConverter`
- [ ] Load instrument specs on init
- [ ] Validate all trades against specs
- [ ] Calculate margin exactly as OANDA
- [ ] Implement proper financing/swap
- [ ] Add conversion factor tracking

### Testing & Validation
- [ ] Verify margin calculations match OANDA
- [ ] Test currency conversions
- [ ] Validate min/max trade sizes
- [ ] Confirm leverage calculations
- [ ] Test cross-currency P&L

---

## 🎯 Priority Order

1. **CRITICAL** (Do Today):
   - Get instrument details API
   - Get account details API  
   - Implement margin calculator
   - Implement currency converter

2. **HIGH** (This Week):
   - Get trades API
   - Get pricing API
   - Enhanced environment validation
   - Complete testing

3. **MEDIUM** (Next Week):
   - Pricing streams
   - Transaction history
   - Advanced features

---

*Based on OANDA v20 API Official Documentation*
