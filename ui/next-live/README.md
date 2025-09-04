Next Live Trading UI

Run
- Set `NEXT_PUBLIC_API_BASE` to your backend base URL (default http://localhost:8000)
- Dev: `npm run dev`
- Prod: `npm run build && npm start`

Pages
- `/` Dashboard: metrics, open positions
- `/symbols` Symbols & Model: pick symbols, filtered models by capacity, target capital input
- `/risk` Risk Controls: tune risk params and push to backend
- `/account` Account summary and open positions
- `/history` Trades table
- `/logs` Live event feed over WebSocket

