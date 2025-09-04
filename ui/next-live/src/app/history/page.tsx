"use client";
import useSWR from "swr";
import { api } from "@/lib/api";

export default function HistoryPage() {
  const { data: trades } = useSWR(["trades", 100], () => api.trades(100), { refreshInterval: 5000 });
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Trade History</h1>
      <section className="card">
        <div className="card-header">Recent Trades</div>
        <div className="card-body overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left border-b">
                <th className="py-2 pr-4">Time (Open)</th>
                <th className="py-2 pr-4">Instrument</th>
                <th className="py-2 pr-4">Side</th>
                <th className="py-2 pr-4">Units</th>
                <th className="py-2 pr-4">Entry</th>
                <th className="py-2 pr-4">Close</th>
                <th className="py-2 pr-4">PnL</th>
                <th className="py-2 pr-4">Status</th>
              </tr>
            </thead>
            <tbody>
              {(trades || []).map((t: any, i:number) => (
                <tr key={i} className="border-b last:border-b-0">
                  <td className="py-2 pr-4">{t.open_timestamp}</td>
                  <td className="py-2 pr-4">{t.instrument}</td>
                  <td className="py-2 pr-4">{Number(t.units) > 0 ? 'BUY' : 'SELL'}</td>
                  <td className="py-2 pr-4">{t.units}</td>
                  <td className="py-2 pr-4">{Number(t.entry_price ?? 0).toFixed(5)}</td>
                  <td className="py-2 pr-4">{t.close_price ? Number(t.close_price).toFixed(5) : '-'}</td>
                  <td className={"py-2 pr-4 " + (Number(t.pnl ?? 0) >= 0 ? 'text-green-600' : 'text-red-600')}>{Number(t.pnl ?? 0).toFixed(2)}</td>
                  <td className="py-2 pr-4">{t.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

