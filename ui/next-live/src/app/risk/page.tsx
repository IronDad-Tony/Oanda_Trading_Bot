"use client";
import { useState } from "react";
import { api } from "@/lib/api";

type RiskParams = {
  max_total_exposure_usd?: number;
  max_risk_per_trade_percent?: number;
  use_atr_sizing?: boolean;
  atr_period?: number;
  stop_loss_atr_multiplier?: number;
  take_profit_atr_multiplier?: number;
  stop_loss_pips?: number;
  take_profit_pips?: number;
  daily_loss_limit_pct?: number;
  max_concurrent_positions?: number;
  per_symbol_limit?: number;
}

export default function RiskPage() {
  const [params, setParams] = useState<RiskParams>({});
  const [saving, setSaving] = useState(false);

  function update<K extends keyof RiskParams>(k: K, v: RiskParams[K]) {
    setParams(prev => ({ ...prev, [k]: v }));
  }

  async function save() {
    setSaving(true);
    try {
      await api.riskUpdate(params);
      alert("Risk parameters updated.");
    } catch (e:any) {
      alert("Failed to update: " + e.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-6 max-w-3xl">
      <h1 className="text-2xl font-semibold">Risk Controls</h1>
      <section className="card">
        <div className="card-header">Configure</div>
        <div className="card-body grid grid-cols-1 md:grid-cols-2 gap-4">
          <Num label="Max Total Exposure (acct ccy)" onChange={v=>update('max_total_exposure_usd', v)} />
          <Num label="Max Risk per Trade (%)" onChange={v=>update('max_risk_per_trade_percent', v)} />
          <Toggle label="Use ATR Sizing" onChange={v=>update('use_atr_sizing', v)} />
          <Num label="ATR Period" onChange={v=>update('atr_period', v)} />
          <Num label="SL ATR Multiplier" onChange={v=>update('stop_loss_atr_multiplier', v)} />
          <Num label="TP ATR Multiplier" onChange={v=>update('take_profit_atr_multiplier', v)} />
          <Num label="Stop Loss (pips)" onChange={v=>update('stop_loss_pips', v)} />
          <Num label="Take Profit (pips)" onChange={v=>update('take_profit_pips', v)} />
          <Num label="Daily Loss Limit (%)" onChange={v=>update('daily_loss_limit_pct', v)} />
          <Num label="Max Concurrent Positions" onChange={v=>update('max_concurrent_positions', v)} />
          <Num label="Per-Symbol Units Cap" onChange={v=>update('per_symbol_limit', v)} />
        </div>
      </section>
      <div>
        <button disabled={saving} onClick={save} className="px-4 py-2 rounded bg-black text-white disabled:opacity-50 dark:bg-white dark:text-black">Save</button>
      </div>
    </div>
  );
}

function Num({ label, onChange }: { label: string; onChange: (v: number|undefined)=>void }) {
  return (
    <label className="text-sm">
      <div className="mb-1">{label}</div>
      <input type="number" className="w-full border rounded px-3 py-2 bg-white dark:bg-gray-900" onChange={e=>onChange(e.target.value === '' ? undefined : Number(e.target.value))} />
    </label>
  );
}

function Toggle({ label, onChange }: { label: string; onChange: (v: boolean)=>void }) {
  return (
    <label className="text-sm flex items-center gap-2">
      <input type="checkbox" onChange={e=>onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

