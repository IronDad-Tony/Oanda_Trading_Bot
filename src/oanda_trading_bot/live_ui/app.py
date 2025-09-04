"""
New Live Trading UI (Streamlit)

Modern control/dashboard for live trading with:
- Training symbols selection
- Dynamic model capacity filtering
- Target capital (<= account equity)
- Cold-start warmup before trading
- Full account/positions/trades/performance panels
- Comprehensive risk controls and live updates
- Emergency Stop to flatten and halt

This UI replaces all previous UI scripts (Next.js and legacy Streamlit).
Launch via start_live_trading_ui.bat.
"""
from __future__ import annotations

import os
import json
import threading
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd

from oanda_trading_bot.live_trading_system.main import initialize_system, trading_loop


# --------------------- Utilities ---------------------
def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def get_training_symbols(cfg: Dict[str, Any]) -> List[str]:
    """Union of symbols from dataset metadata; fallback to config or defaults."""
    root = _project_root()
    mmap_dir = os.path.join(root, "data", "mmap_s5_universal")
    symbols: set[str] = set()
    try:
        if os.path.isdir(mmap_dir):
            for name in os.listdir(mmap_dir):
                meta_path = os.path.join(mmap_dir, name, "dataset_metadata.json")
                if os.path.isfile(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            for s in data.get("symbols", []) or []:
                                if isinstance(s, str):
                                    symbols.add(s)
                    except Exception:
                        pass
    except Exception:
        pass

    if symbols:
        return sorted(symbols)

    # Fallbacks
    cfg_syms = (cfg or {}).get("trading_instruments") or []
    if cfg_syms:
        return list(cfg_syms)

    try:
        from oanda_trading_bot.training_system.common.config import DEFAULT_SYMBOLS
        return list(DEFAULT_SYMBOLS)
    except Exception:
        return ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "XAU_USD"]


def scan_models(lookback_default: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    root = _project_root()
    for d in ("trained_models", "weights"):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        for fn in os.listdir(p):
            if not fn.lower().endswith((".zip", ".pt", ".pth")):
                continue
            full = os.path.join(p, fn)
            max_symbols = 999999
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


def account_summary(components: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return components["client"].get_account_summary() or {}
    except Exception:
        return {}


def open_positions(components: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        data = components["client"].get_open_positions() or {}
        out: List[Dict[str, Any]] = []
        for pos in data.get("positions", []) or []:
            ins = pos.get("instrument")
            long_units = int(pos.get("long", {}).get("units", "0") or 0)
            short_units = int(pos.get("short", {}).get("units", "0") or 0)
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


def trade_history(components: Dict[str, Any], limit: int = 200) -> List[Dict[str, Any]]:
    try:
        return components["db_manager"].get_trade_history(limit=limit) or []
    except Exception:
        return []


def compute_performance(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {
            "realized_pnl": 0.0,
            "win_rate": 0.0,
            "count": 0,
            "equity_curve": pd.DataFrame([]),
        }
    df = pd.DataFrame(trades).copy()
    # Robust conversions
    if "pnl" not in df.columns:
        df["pnl"] = 0.0
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)

    # Avoid ambiguous truth-value on Series; construct a proper Series fallback
    if "close_timestamp" in df.columns:
        close_col = df["close_timestamp"]
    else:
        close_col = pd.Series([None] * len(df), index=df.index)
    df["close_timestamp"] = pd.to_datetime(close_col, errors="coerce")

    # Sort by available time columns
    sort_by = [c for c in ["close_timestamp", "open_timestamp"] if c in df.columns]
    if sort_by:
        df = df.sort_values(sort_by, na_position="last")

    realized_pnl = float(df["pnl"].sum())
    wins = int((df["pnl"] > 0).sum())
    count = int(len(df))
    win_rate = (wins / count) * 100.0 if count > 0 else 0.0

    curve = df.dropna(subset=["close_timestamp"]) if "close_timestamp" in df.columns else pd.DataFrame([])
    if isinstance(curve, pd.DataFrame) and not curve.empty:
        curve = curve.copy()
        curve["cum_pnl"] = curve["pnl"].cumsum()
        curve = curve[["close_timestamp", "cum_pnl"]].rename(columns={"close_timestamp": "time"})
    else:
        curve = pd.DataFrame([])
    return {
        "realized_pnl": realized_pnl,
        "win_rate": win_rate,
        "count": count,
        "equity_curve": curve,
    }


def apply_config(
    components: Dict[str, Any],
    symbols: List[str],
    model_path: Optional[str],
    target_capital: Optional[float],
    risk_params: Dict[str, Any],
) -> Optional[str]:
    cfg = components.get("config") or {}

    # Validate symbols vs training set (or dataset union)
    allowed_syms = set(get_training_symbols(cfg))
    if not set(symbols).issubset(allowed_syms):
        return "選擇的 symbols 不在訓練系統可用清單內"

    # Validate target capital <= equity
    if target_capital is not None:
        acc = account_summary(components)
        acct = acc.get("account") or {}
        equity = float((acct.get("NAV") or acct.get("equity") or 0))
        if target_capital > equity:
            return f"目標操作金額 {target_capital:.2f} 超過目前帳戶餘額 {equity:.2f}"

    # Apply
    ss = components["system_state"]
    pred = components["prediction_service"]
    risk = components["risk_manager"]
    tl = components["trading_logic"]

    ss.set_selected_instruments(symbols)
    if model_path:
        try:
            pred.load_model(model_path, device=None)
            ss.set_current_model(model_path)
        except Exception as e:
            return f"載入模型失敗: {e}"

    update = dict(risk_params or {})
    if target_capital is not None:
        update["max_total_exposure_usd"] = float(target_capital)
    try:
        risk.update_params(update)
    except Exception as e:
        return f"更新風險參數失敗: {e}"

    # Ensure buffers exist for selected instruments
    lookback = int(cfg.get("model_lookback_window", 128))
    from collections import deque
    for sym in symbols:
        if sym not in tl.data_buffers:
            tl.data_buffers[sym] = deque(maxlen=lookback)
    return None


def start_trading(components: Dict[str, Any], warmup_timeout: int = 120):
    ss = components["system_state"]
    tl = components["trading_logic"]
    instruments = ss.get_selected_instruments()
    tl.warmup_buffers(instruments, max_wait_seconds=warmup_timeout, sleep_seconds=1)
    ss.start()
    if components.get("_thread") is None:
        t = threading.Thread(target=trading_loop, args=(components,), daemon=True)
        components["_thread"] = t
        t.start()


def stop_trading(components: Dict[str, Any]):
    ss = components["system_state"]
    ss.stop()
    t = components.get("_thread")
    if t and t.is_alive():
        t.join(timeout=10)
        components["_thread"] = None


# --------------------- UI ---------------------
st.set_page_config(page_title="Live Trading Control", layout="wide")

# Light modern styles
st.markdown(
    """
    <style>
      .metric {text-align:center}
      .stButton>button { width: 100%; height: 2.6em; }
      .risk-box { padding: 0.75rem; border: 1px solid #e0e0e0; border-radius: 8px; background: #fafafa; }
      .danger { background: #ffe8e8 !important; border-color: #ffc0c0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Boot components once
if "components" not in st.session_state:
    st.session_state.components = initialize_system()
if "model_cache" not in st.session_state:
    st.session_state.model_cache = None

comps = st.session_state.components
if not comps:
    st.error("初始化交易系統失敗。請確認 .env 與 configs 設定正確。")
    st.stop()

cfg = comps.get("config") or {}
lookback_default = int(cfg.get("model_lookback_window", 128))
training_symbols = get_training_symbols(cfg)

with st.sidebar:
    st.header("控制面板")
    running = bool(getattr(comps["system_state"], "is_running", False))

    # Symbol selection
    selected_symbols = st.multiselect(
        "選擇交易 Symbols",
        options=training_symbols,
        default=training_symbols[:min(3, len(training_symbols))],
    )

    # Model selection (filtered by capacity)
    if st.session_state.model_cache is None:
        st.session_state.model_cache = scan_models(lookback_default)
    models = [m for m in (st.session_state.model_cache or []) if m.get("max_symbols", 0) >= len(selected_symbols)]
    display = [f"{m['name']} (max {m['max_symbols']})" for m in models]
    model_path: Optional[str] = None
    choice = st.selectbox("選擇模型", options=["(不載入)"] + display, index=0)
    if choice and choice != "(不載入)":
        model_path = models[display.index(choice)]["path"]

    # Target capital <= equity
    acc = account_summary(comps)
    acct = acc.get("account") or {}
    equity = float((acct.get("NAV") or acct.get("equity") or 0))
    st.caption(f"目前帳戶餘額/淨值: {equity:.2f}")
    target_capital = st.number_input(
        "目標操作金額 (帳戶幣別)",
        min_value=0.0,
        value=min(1000.0, equity) if equity > 0 else 0.0,
        step=100.0,
        format="%.2f",
    )

    # Risk controls
    st.subheader("風險控管")
    rm_cfg = (cfg.get("risk_management") or {}) if isinstance(cfg, dict) else {}
    max_total_exposure_usd = st.number_input(
        "最大總曝險 (USD)",
        min_value=0.0,
        value=float(rm_cfg.get("max_total_exposure_usd", 0.0)),
        step=100.0,
        format="%.2f",
    )
    max_risk_per_trade_percent = st.number_input(
        "單筆最大風險(%)",
        min_value=0.0,
        max_value=100.0,
        value=float(rm_cfg.get("max_risk_per_trade_percent", 0.5)),
        step=0.1,
        format="%.2f",
    )
    daily_loss_limit_pct = st.number_input(
        "單日最大虧損限制(%)",
        min_value=0.0,
        max_value=100.0,
        value=float(cfg.get("daily_loss_limit_pct", 5.0)),
        step=0.1,
        format="%.2f",
    )
    use_atr = st.checkbox("使用 ATR 倉位 sizing", value=bool(rm_cfg.get("use_atr_sizing", True)))
    atr_period = st.number_input("ATR 期間", min_value=1, value=int(rm_cfg.get("atr_period", 14)))
    sl_atr = st.number_input("SL ATR 倍數", min_value=0.0, value=float(rm_cfg.get("stop_loss_atr_multiplier", 2.0)), step=0.1)
    tp_atr = st.number_input("TP ATR 倍數", min_value=0.0, value=float(rm_cfg.get("take_profit_atr_multiplier", 3.0)), step=0.1)
    sl_pips = st.number_input("固定停損(pips)", min_value=0.0, value=float(rm_cfg.get("stop_loss_pips", 10)))
    tp_pips = st.number_input("固定停利(pips)", min_value=0.0, value=float(rm_cfg.get("take_profit_pips", 10)))
    use_price_bound = st.checkbox("限制滑點 (priceBound)", value=bool((cfg.get("execution") or {}).get("use_price_bound", True)))
    max_slippage_pips = st.number_input("最大滑點(pips)", min_value=0.0, value=float((cfg.get("execution") or {}).get("max_slippage_pips", 1.0)), step=0.1)

    if st.button("套用設定", disabled=running):
        err = apply_config(
            comps,
            selected_symbols,
            model_path,
            target_capital if target_capital > 0 else None,
            {
                "max_total_exposure_usd": max_total_exposure_usd,
                "max_risk_per_trade_percent": max_risk_per_trade_percent,
                "use_atr_sizing": use_atr,
                "atr_period": atr_period,
                "stop_loss_atr_multiplier": sl_atr,
                "take_profit_atr_multiplier": tp_atr,
                "stop_loss_pips": sl_pips,
                "take_profit_pips": tp_pips,
                "daily_loss_limit_pct": daily_loss_limit_pct,
                "use_price_bound": use_price_bound,
                "max_slippage_pips": max_slippage_pips,
            },
        )
        if err:
            st.error(err)
        else:
            st.success("設定已套用")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("開始交易", type="primary", disabled=running):
            start_trading(comps, warmup_timeout=180)
            st.rerun()
    with c2:
        if st.button("停止交易", disabled=not running):
            stop_trading(comps)
            st.rerun()

    st.markdown("<div class='risk-box danger'>", unsafe_allow_html=True)
    if st.button("緊急停止 (平倉+停止)", disabled=running is False):
        try:
            comps["order_manager"].close_all_positions()
            stop_trading(comps)
            st.success("已平倉並停止")
        except Exception as e:
            st.error(f"緊急停止失敗: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------- Dashboard ---------------------
st.title("Live Trading Dashboard")
status = "RUNNING" if bool(getattr(comps["system_state"], "is_running", False)) else "STOPPED"
st.caption(f"狀態: {status}")

acc = account_summary(comps)
acct = acc.get("account") or {}
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Equity", f"{float(acct.get('NAV') or acct.get('equity') or 0):.2f}")
col2.metric("P/L", f"{float(acct.get('pl') or 0):.2f}")
col3.metric("Margin Used", f"{float(acct.get('marginUsed') or 0):.2f}")
col4.metric("Open Positions", f"{int(acct.get('openPositionCount') or 0)}")
col5.metric("Open Trades", f"{int(acct.get('openTradeCount') or 0)}")

st.divider()

st.subheader("持倉資訊")
pos = open_positions(comps)
if pos:
    st.dataframe(pd.DataFrame(pos), use_container_width=True)
else:
    st.info("目前沒有持倉")

st.divider()

st.subheader("近期成交紀錄")
tr = trade_history(comps, limit=300)
if tr:
    st.dataframe(pd.DataFrame(tr), use_container_width=True)
else:
    st.info("尚無成交紀錄")

st.divider()

st.subheader("交易績效")
perf = compute_performance(tr)
m1, m2, m3 = st.columns(3)
m1.metric("Realized PnL", f"{perf['realized_pnl']:.2f}")
m2.metric("Win Rate", f"{perf['win_rate']:.1f}%")
m3.metric("Trades", f"{perf['count']}")

if isinstance(perf.get("equity_curve"), pd.DataFrame) and not perf["equity_curve"].empty:
    st.line_chart(perf["equity_curve"].set_index("time")["cum_pnl"])
else:
    st.caption("等待更多成交以生成績效曲線…")
