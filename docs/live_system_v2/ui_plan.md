UI Plan — Options and Pages

Stack Options
1) Next.js + React + TailwindCSS (+ shadcn/ui, TanStack Table, Recharts or ECharts)
   - Pros: Modern UX, fast, component ecosystem, great charts, SSR/SPA hybrid.
   - Cons: Requires Node/PNPM; separate frontend build/run.

2) Streamlit (multi‑page) + custom components + theming
   - Pros: Fast to ship, Python‑native, minimal setup.
   - Cons: Less control over complex layouts/interactions; fewer data‑grid options.

Pages/Sections
- Dashboard: key metrics (Equity, PnL, Drawdown, Margin), equity curve, open positions, recent orders, system status.
- Symbols & Model: training symbols list, selection UI; dynamic model list filtered by symbol count; model details.
- Risk Controls: live controls for risk params; validation; presets; ability to save/load profiles.
- Capital & Session: input target capital (<= account equity), set trading window, start/stop with cold‑start status.
- Account: full account info, positions with cost basis, exposure by instrument, realized/unrealized PnL.
- History & Performance: trades table, per‑symbol stats, win rate, PF, avg R, Sharpe/Sortino (simple), export CSV.
- Logs: live event feed with levels and filters; system alerts (slippage, spreads, circuit breakers).

Realtime
- WebSocket subscription; optimistic updates for controls; fallback polling.

Design Notes
- Light/dark theme; responsive layout; sticky control pane; status chips (Running/Stopped/Error).
- Charting: candlestick + trades overlay; equity curve; bar chart by symbol PnL.

