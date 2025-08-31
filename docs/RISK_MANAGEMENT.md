Risk Management Controls

Overview
- The project supports consistent risk controls across Training and Live trading.
- Two sizing modes are available:
  1) ATR-based equal-risk sizing (preferred)
  2) Pip-based sizing (fallback)

Training System
- Parameters (in `training_system/common/config.py`):
  - `MAX_ACCOUNT_RISK_PERCENTAGE`: per-trade risk as a fraction of equity (e.g., 0.01 = 1%)
  - `ATR_PERIOD`: ATR lookback used by the environment
  - `STOP_LOSS_ATR_MULTIPLIER`: stop distance = ATR Ã— multiplier
  - `MAX_POSITION_SIZE_PERCENTAGE_OF_EQUITY`: cap on position value
- Behavior:
  - For each symbol, the environment computes ATR.
  - Target units are capped by both risk (R) and max-position-size; the minimum is used.
  - Commission and conversions follow OANDA credit/debit rules in account currency.

Live Trading System
- Configuration (`configs/live/live_config.json`):
  - `risk_management.use_atr_sizing`: true to enable ATR-based sizing.
  - `risk_management.atr_period`: ATR lookback (e.g., 14)
  - `risk_management.stop_loss_atr_multiplier`: SL distance multiplier (e.g., 2.0)
  - `risk_management.max_risk_per_trade_percent`: per-trade risk as % of equity (e.g., 0.1)
  - `risk_management.stop_loss_pips`, `take_profit_pips`: fallback and TP settings
- Behavior:
  - When ATR sizing is enabled and ATR is available, units = (equity Ã— risk%) / (ATR Ã— multiplier Ã— quoteâ†’account rate).
  - Stop loss is set at `price Â± ATR Ã— multiplier`.
  - If ATR is not available, the system falls back to pip-based sizing with pip SL.
  - Take profit can use ATR × `take_profit_atr_multiplier` when ATR sizing is enabled; otherwise, it falls back to `take_profit_pips`.

Notes and Recommendations
- ATR sizing is robust across volatility regimes; use pip fallback only when ATR cannot be computed (e.g., insufficient candles).
- Consider adding trailing stop or volatility-adaptive multipliers for advanced risk control.
- Ensure the quoteâ†’account conversion is available to compute correct risk in account currency.

