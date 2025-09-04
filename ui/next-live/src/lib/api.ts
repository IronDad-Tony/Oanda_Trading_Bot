const BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  // Proxy through Next API to avoid CORS issues
  const res = await fetch(`/api/proxy${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  health: () => http<{ status: string; engine: string }>("/health"),
  accountSummary: () => http<any>("/account/summary"),
  positions: () => http<any[]>("/account/positions"),
  trades: (limit=100) => http<any[]>(`/account/trades?limit=${limit}`),
  trainingSymbols: () => http<{ symbols: string[] }>("/symbols/training"),
  models: (minSymbols: number) => http<any[]>(`/models?min_symbols=${minSymbols}`),
  sessionStatus: () => http<any>("/session/status"),
  sessionConfig: (body: any) => http<{ ok: boolean }>("/session/config", { method: "POST", body: JSON.stringify(body) }),
  sessionStart: (coldTimeoutSec?: number) => http<{ started: boolean }>("/session/start", { method: "POST", body: JSON.stringify({ cold_start_timeout_sec: coldTimeoutSec }) }),
  sessionStop: () => http<{ stopped: boolean }>("/session/stop", { method: "POST" }),
  riskUpdate: (body: any) => http<{ ok: boolean }>("/risk/update", { method: "POST", body: JSON.stringify(body) }),
};

export function getWsUrl(path = "/ws") {
  // Browser connects directly; allow env override
  const base = (process.env.NEXT_PUBLIC_API_BASE || BASE).replace(/^http/, "ws");
  return `${base}${path}`;
}
