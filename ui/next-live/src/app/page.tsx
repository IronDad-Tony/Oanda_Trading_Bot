"use client";
import useSWR from "swr";
import { api } from "@/lib/api";
import Link from "next/link";

export default function Dashboard() {
  const { data: health } = useSWR("health", api.health, { refreshInterval: 3000 });
  const { data: summary } = useSWR("summary", api.accountSummary, { refreshInterval: 5000 });
  const { data: positions, mutate: mutPos } = useSWR("positions", api.positions, { refreshInterval: 5000 });
  const { data: status, mutate: mutStatus } = useSWR("status", api.sessionStatus, { refreshInterval: 3000 });
  async function start() { try { await api.sessionStart(120); await Promise.all([mutStatus(), mutPos()]); } catch (e:any) { alert(e.message); } }
  async function stop() { try { await api.sessionStop(); await Promise.all([mutStatus(), mutPos()]); } catch (e:any) { alert(e.message); } }

  const equity = Number(summary?.account?.NAV ?? summary?.account?.equity ?? 0);
  const pnl = Number(summary?.account?.pl ?? 0);
  const marginUsed = Number(summary?.account?.marginUsed ?? 0);
  const openPositions = Number(summary?.account?.openPositionCount ?? positions?.length ?? 0);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <div className="flex items-center gap-2">
          <div className="text-sm rounded px-2 py-1 border bg-white dark:bg-gray-900">
            Engine: <span className={(status?.running ? "text-green-600" : "text-gray-500")}>{status?.running ? 'running' : 'stopped'}</span>
          </div>
          {status?.running ? (
            <button onClick={stop} className="px-3 py-2 rounded border text-sm">Stop</button>
          ) : (
            <button onClick={start} className="px-3 py-2 rounded bg-black text-white text-sm dark:bg-white dark:text-black">Start</button>
          )}
        </div>
      </div>
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Metric title="Equity" value={`$${equity.toFixed(2)}`} />
        <Metric title="P/L" value={`$${pnl.toFixed(2)}`} trend={pnl} />
        <Metric title="Margin Used" value={`$${marginUsed.toFixed(2)}`} />
        <Metric title="Open Positions" value={`${openPositions}`} />
      </section>

      <section className="card">
        <div className="card-header">Quick Actions</div>
        <div className="card-body flex gap-3 text-sm">
          <Link href="/symbols" className="px-3 py-2 rounded bg-black text-white dark:bg-white dark:text-black">Configure Symbols & Model</Link>
          <Link href="/risk" className="px-3 py-2 rounded border">Risk Controls</Link>
          <Link href="/history" className="px-3 py-2 rounded border">Recent Trades</Link>
          <Link href="/logs" className="px-3 py-2 rounded border">Logs</Link>
        </div>
      </section>

      <section className="card">
        <div className="card-header">Open Positions</div>
        <div className="card-body overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left border-b">
                <th className="py-2 pr-4">Instrument</th>
                <th className="py-2 pr-4">Side</th>
                <th className="py-2 pr-4">Units</th>
                <th className="py-2 pr-4">Avg Price</th>
                <th className="py-2 pr-4">Unrealized P/L</th>
              </tr>
            </thead>
            <tbody>
              {(positions || []).map((p: any, i: number) => (
                <tr key={i} className="border-b last:border-b-0">
                  <td className="py-2 pr-4">{p.instrument}</td>
                  <td className="py-2 pr-4">{p.side}</td>
                  <td className="py-2 pr-4">{p.units}</td>
                  <td className="py-2 pr-4">{Number(p.avgPrice ?? 0).toFixed(5)}</td>
                  <td className="py-2 pr-4">{Number(p.unrealizedPL ?? 0).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function Metric({ title, value, trend }: { title: string; value: string; trend?: number }) {
  return (
    <div className="card">
      <div className="card-body">
        <div className="text-sm text-gray-500">{title}</div>
        <div className="text-2xl font-semibold">{value}</div>
        {trend !== undefined && (
          <div className={"text-xs mt-1 " + (trend >= 0 ? "text-green-600" : "text-red-600")}>{trend >= 0 ? "+" : ""}{trend.toFixed(2)}</div>
        )}
      </div>
    </div>
  );
}
