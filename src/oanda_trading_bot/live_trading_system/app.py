"""
Streamlit UI for Live Trading System (reworked).

This UI manages configuration (symbols, model, risk), controls
start/stop with cold start, and shows account/positions/trades.
"""
import os
import threading
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd

from oanda_trading_bot.live_trading_system.main import initialize_system, trading_loop


# -------------- Helpers --------------
def _scan_models(project_root: str, lookback_default: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for d in ["trained_models", "weights"]:
        p = os.path.join(project_root, d)
        if not os.path.isdir(p):
            continue
        for fn in os.listdir(p):
            if not fn.lower().endswith((".zip", ".pt", ".pth")):
                continue
            full = os.path.join(p, fn)
            max_symbols = 9999
            try:
                import re
                m = re.search(r"symbols(\d+)", fn, re.IGNORECASE)
                if m:
                    max_symbols = int(m.group(1))
            except Exception:
                pass
            items.append({
                "name": fn,
                "path": full,
                "max_symbols": max_symbols,
                "lookback_window": lookback_default,
                "type": "SAC" if "sac" in fn.lower() else "Torch",
            })
    return sorted(items, key=lambda x: (x.get("max_symbols", 0), x.get("name", "")))


def _account_summary(components: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return components["client"].get_account_summary() or {}
    except Exception:
        return {}


def _open_positions(components: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        data = components["client"].get_open_positions() or {}
        out: List[Dict[str, Any]] = []
        for pos in data.get("positions", []):
            ins = pos.get("instrument")
            long_units = int(pos.get("long", {}).get("units", "0"))
            short_units = int(pos.get("short", {}).get("units", "0"))
            if long_units:
                out.append({
                    "instrument": ins,
                    "side": "BUY",
                    "units": long_units,
                    "avg_price": float(pos.get("long", {}).get("averagePrice", 0) or 0),
                    "unrealized_pl": float(pos.get("long", {}).get("unrealizedPL", 0) or 0),
                })
            if short_units:
                out.append({
                    "instrument": ins,
                    "side": "SELL",
                    "units": short_units,
                    "avg_price": float(pos.get("short", {}).get("averagePrice", 0) or 0),
                    "unrealized_pl": float(pos.get("short", {}).get("unrealizedPL", 0) or 0),
                })
        return out
    except Exception:
        return []


def _trade_history(components: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
    try:
        return components["db_manager"].get_trade_history(limit=limit) or []
    except Exception:
        return []


def _apply_config(components: Dict[str, Any], symbols: List[str], model_path: Optional[str], target_capital: Optional[float], risk_params: Dict[str, Any]) -> Optional[str]:
    # Validate symbols against training set
    cfg = components["config"] or {}
    training = set(cfg.get("trading_instruments", []))
    if not set(symbols).issubset(training):
        return "選取的 symbols 不在訓練清單內"

    # Validate target capital against equity
    if target_capital is not None:
        acc = _account_summary(components)
        equity = float((((acc.get('account') or {}).get('NAV')) or ((acc.get('account') or {}).get('equity')) or 0))
        if target_capital > equity:
            return f"目標資金 {target_capital:.2f} 超過當前權益 {equity:.2f}"

    # Apply
    ss = components["system_state"]
    pred = components["prediction_service"]
    risk = components["risk_manager"]
    tl = components["trading_logic"]

    ss.set_selected_instruments(symbols)
    if model_path:
        pred.load_model(model_path, device=None)
        ss.set_current_model(model_path)

    update = dict(risk_params or {})
    if target_capital is not None:
        update["max_total_exposure_usd"] = float(target_capital)
    risk.update_params(update)

    # Prepare buffers for new selection
    lookback = int(cfg.get("model_lookback_window", 128))
    for sym in symbols:
        if sym not in tl.data_buffers:
            from collections import deque
            tl.data_buffers[sym] = deque(maxlen=lookback)
    return None


def _start(components: Dict[str, Any], warmup_timeout: int = 120):
    ss = components["system_state"]
    tl = components["trading_logic"]
    instruments = ss.get_selected_instruments()
    tl.warmup_buffers(instruments, max_wait_seconds=warmup_timeout, sleep_seconds=1)
    ss.start()
    if "_thread" not in components or components.get("_thread") is None:
        t = threading.Thread(target=trading_loop, args=(components,), daemon=True)
        components["_thread"] = t
        t.start()


def _stop(components: Dict[str, Any]):
    ss = components["system_state"]
    ss.stop()
    t = components.get("_thread")
    if t and t.is_alive():
        t.join(timeout=10)
        components["_thread"] = None


# -------------- UI --------------
st.set_page_config(page_title="Live Trading Dashboard", layout="wide")

if "components" not in st.session_state:
    st.session_state.components = initialize_system()
if "model_cache" not in st.session_state:
    st.session_state.model_cache = None

comps = st.session_state.components
if not comps:
    st.error("系統初始化失敗，請檢查 .env 與 configs。")
    st.stop()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
cfg = comps["config"] or {}
lookback_default = int(cfg.get("model_lookback_window", 128))
training_symbols: List[str] = cfg.get("trading_instruments", [])

with st.sidebar:
    st.header("系統設定")
    running = getattr(comps["system_state"], 'is_running', False)
    # Symbols
    selected_symbols = st.multiselect("選擇交易 symbols", options=training_symbols, default=training_symbols[:min(3, len(training_symbols))])

    # Models filtered by capacity
    if st.session_state.model_cache is None:
        st.session_state.model_cache = _scan_models(project_root, lookback_default)
    models = [m for m in (st.session_state.model_cache or []) if m.get("max_symbols", 0) >= len(selected_symbols)]
    model_display = [f"{m['name']} (max {m['max_symbols']})" for m in models]
    model_choice = st.selectbox("選擇模型", options=["(不更換)"] + model_display, index=0)
    model_path = None
    if model_choice and model_choice != "(不更換)":
        model_path = models[model_display.index(model_choice)]["path"]

    # Target capital (account ccy)
    acc = _account_summary(comps)
    equity = float((((acc.get('account') or {}).get('NAV')) or ((acc.get('account') or {}).get('equity')) or 0))
    st.caption(f"當前權益約: {equity:.2f}")
    target_capital = st.number_input("目標資金 (帳戶幣別)", min_value=0.0, value=min(1000.0, equity) if equity>0 else 0.0, step=100.0, format="%.2f")

    st.divider()
    st.subheader("風險控制")
    rm_cfg = (cfg.get("risk_management") or {}) if isinstance(cfg, dict) else {}
    max_risk_per_trade = st.number_input("每筆風險%", min_value=0.0, max_value=100.0, value=float(rm_cfg.get('max_risk_per_trade_percent', 1.0)), step=0.1, format="%.2f")
    use_atr = st.checkbox("使用 ATR 風險", value=bool(rm_cfg.get('use_atr_sizing', False)))
    atr_period = st.number_input("ATR 期間", min_value=1, value=int(rm_cfg.get('atr_period', 14)))
    sl_atr = st.number_input("SL ATR 倍數", min_value=0.0, value=float(rm_cfg.get('stop_loss_atr_multiplier', 2.0)), step=0.1)
    tp_atr = st.number_input("TP ATR 倍數", min_value=0.0, value=float(rm_cfg.get('take_profit_atr_multiplier', 3.0)), step=0.1)
    sl_pips = st.number_input("止損(pips)", min_value=0.0, value=float(rm_cfg.get('stop_loss_pips', 10)))
    tp_pips = st.number_input("停利(pips)", min_value=0.0, value=float(rm_cfg.get('take_profit_pips', 10)))

    if st.button("套用設定", disabled=running):
        err = _apply_config(
            comps,
            selected_symbols,
            model_path,
            target_capital if target_capital > 0 else None,
            {
                "max_risk_per_trade_percent": max_risk_per_trade,
                "use_atr_sizing": use_atr,
                "atr_period": atr_period,
                "stop_loss_atr_multiplier": sl_atr,
                "take_profit_atr_multiplier": tp_atr,
                "stop_loss_pips": sl_pips,
                "take_profit_pips": tp_pips,
            }
        )
        if err:
            st.error(err)
        else:
            st.success("設定已套用。")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start", type="primary", disabled=running):
            _start(comps, warmup_timeout=120)
            st.experimental_rerun()
    with col_b:
        if st.button("Stop", disabled=not running):
            _stop(comps)
            st.experimental_rerun()

    if st.button("平倉所有持倉", disabled=running):
        try:
            comps["order_manager"].close_all_positions()
            st.success("已送出全部平倉。")
        except Exception as e:
            st.error(f"平倉失敗: {e}")


st.title("Live Trading Dashboard")
status = "RUNNING" if getattr(comps["system_state"], 'is_running', False) else "STOPPED"
st.caption(f"狀態: {status}")

# Metrics
acc = _account_summary(comps)
acct = acc.get('account') or {}
col1, col2, col3, col4 = st.columns(4)
col1.metric("Equity", f"{float(acct.get('NAV') or acct.get('equity') or 0):.2f}")
col2.metric("P/L", f"{float(acct.get('pl') or 0):.2f}")
col3.metric("Margin Used", f"{float(acct.get('marginUsed') or 0):.2f}")
col4.metric("Open Positions", f"{int(acct.get('openPositionCount') or 0)}")

st.divider()

# Positions
st.subheader("Open Positions")
pos = _open_positions(comps)
if pos:
    st.dataframe(pd.DataFrame(pos), use_container_width=True)
else:
    st.info("目前沒有持倉。")

st.divider()

# Trades
st.subheader("Trade History (Recent)")
tr = _trade_history(comps, limit=200)
if tr:
    st.dataframe(pd.DataFrame(tr), use_container_width=True)
else:
    st.info("尚無交易紀錄。")

