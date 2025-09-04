"use client";
import useSWR from "swr";
import { useMemo, useState } from "react";
import { api } from "@/lib/api";

export default function SymbolsModelPage() {
  const { data: symbols } = useSWR("trainingSymbols", api.trainingSymbols);
  const [selected, setSelected] = useState<string[]>([]);
  const { data: models } = useSWR(() => selected.length ? ["models", selected.length] : null, () => api.models(selected.length));
  const [model, setModel] = useState<string>("");
  const [targetCapital, setTargetCapital] = useState<string>("");
  const [saving, setSaving] = useState(false);

  const sorted = useMemo(() => (symbols?.symbols || []).slice().sort(), [symbols]);

  async function save() {
    setSaving(true);
    try {
      await api.sessionConfig({
        symbols: selected,
        model_path: model,
        target_capital_usd: Number(targetCapital || 0),
        risk: {},
      });
      alert("Configuration saved.");
    } catch (e:any) {
      alert("Failed to save: " + e.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Symbols & Model</h1>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="card">
          <div className="card-header">Select Symbols</div>
          <div className="card-body">
            {!sorted.length && <div className="text-sm text-gray-500">No symbols loaded yet.</div>}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
              {sorted.map((sym) => {
                const on = selected.includes(sym);
                return (
                  <button
                    key={sym}
                    className={`px-3 py-2 rounded border text-sm ${on ? "bg-black text-white dark:bg-white dark:text-black" : "bg-white dark:bg-gray-900"}`}
                    onClick={() => setSelected(on ? selected.filter(s=>s!==sym) : [...selected, sym])}
                  >
                    {sym}
                  </button>
                );
              })}
            </div>
            <div className="text-xs text-gray-500 mt-2">Selected: {selected.length}</div>
          </div>
        </section>

        <section className="card">
          <div className="card-header">Model & Capital</div>
          <div className="card-body space-y-4">
            <div>
              <div className="text-sm mb-1">Compatible Models (capacity ≥ {selected.length || 0})</div>
              <select className="w-full border rounded px-3 py-2 bg-white dark:bg-gray-900" value={model} onChange={e=>setModel(e.target.value)}>
                <option value="">Select model…</option>
                {(models || []).map((m:any, i:number)=> (
                  <option key={i} value={m.path}>{m.name || m.path} (max {m.max_symbols})</option>
                ))}
              </select>
            </div>
            <div>
              <div className="text-sm mb-1">Target Capital (account currency)</div>
              <input className="w-full border rounded px-3 py-2 bg-white dark:bg-gray-900" placeholder="e.g. 1000" value={targetCapital} onChange={e=>setTargetCapital(e.target.value)} />
              <div className="text-xs text-gray-500 mt-1">Must be ≤ account equity. Back-end validates.</div>
            </div>
            <div className="flex gap-2">
              <button disabled={saving || !selected.length || !model} onClick={save} className="px-3 py-2 rounded bg-black text-white disabled:opacity-50 dark:bg-white dark:text-black">Save</button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

