"use client";
import useSWR from "swr";
import { api } from "@/lib/api";

export default function AccountPage() {
  const { data: summary } = useSWR("summary", api.accountSummary, { refreshInterval: 5000 });
  const { data: positions } = useSWR("positions", api.positions, { refreshInterval: 5000 });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Account</h1>
      <section className="card">
        <div className="card-header">Summary</div>
        <div className="card-body grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <KV k="Account ID" v={summary?.account?.id} />
          <KV k="Currency" v={summary?.account?.currency} />
          <KV k="Balance" v={fmt(summary?.account?.balance)} />
          <KV k="Equity" v={fmt(summary?.account?.NAV ?? summary?.account?.equity)} />
          <KV k="P/L" v={fmt(summary?.account?.pl)} />
          <KV k="Margin Used" v={fmt(summary?.account?.marginUsed)} />
          <KV k="Open Positions" v={String(summary?.account?.openPositionCount ?? positions?.length ?? 0)} />
          <KV k="Last Txn ID" v={summary?.account?.lastTransactionID} />
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

function KV({ k, v }: { k: string; v: any }) {
  return (
    <div>
      <div className="text-gray-500">{k}</div>
      <div className="font-medium">{String(v ?? "-")}</div>
    </div>
  );
}

function fmt(x: any) {
  const n = Number(x ?? 0);
  if (isNaN(n)) return String(x ?? "-");
  return `$${n.toFixed(2)}`;
}

