const state = {
  symbols: [],
  selectedSymbols: new Set(),
  models: [],
  selectedModelPath: null,
  equity: null,
  risk: {},
  accountCurrency: 'USD',
  running: false,
  catalog: { FX: [], Indices: [], Metals: [], Others: [] },
};

const el = (id) => document.getElementById(id);
const fmt = (n, d=2) => (n===null||n===undefined||isNaN(n))? '-': Number(n).toFixed(d);

async function api(path, opts={}) {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

function renderEngineStatus() {
  const badge = el('engineStatus');
  badge.textContent = state.running ? 'Running' : 'Stopped';
  badge.classList.toggle('running', !!state.running);
}

function renderCatalog() {
  const fill = (id, items) => {
    const sel = el(id);
    if (!sel) return;
    sel.innerHTML = '';
    items.forEach(s => { const o=document.createElement('option'); o.value=s; o.textContent=s; sel.appendChild(o); })
  };
  fill('selFX', state.catalog.FX||[]);
  fill('selIndices', state.catalog.Indices||[]);
  fill('selMetals', state.catalog.Metals||[]);
  fill('selOthers', state.catalog.Others||[]);
  // Restore current selections
  applySelectionsToUI();
}

function applySelectionsToUI() {
  const sets = [ ['selFX', state.catalog.FX||[]], ['selIndices', state.catalog.Indices||[]], ['selMetals', state.catalog.Metals||[]], ['selOthers', state.catalog.Others||[]] ];
  sets.forEach(([id, list]) => {
    const sel = el(id);
    if (!sel) return;
    [...sel.options].forEach(opt => { opt.selected = state.selectedSymbols.has(opt.value); });
  });
}

function recomputeSelectionFromUI() {
  const ids = ['selFX','selIndices','selMetals','selOthers'];
  const newSet = new Set();
  ids.forEach(id => {
    const sel = el(id); if (!sel) return;
    [...sel.selectedOptions].forEach(o => newSet.add(o.value));
  });
  state.selectedSymbols = newSet;
  refreshModels();
}

function renderModels() {
  const sel = el('modelSelect');
  sel.innerHTML = '';
  const optNone = document.createElement('option');
  optNone.value = '';
  optNone.textContent = '(不指定模型)';
  sel.appendChild(optNone);
  state.models.forEach((m) => {
    const o = document.createElement('option');
    o.value = m.path;
    o.textContent = `${m.name} (max ${m.max_symbols})`;
    sel.appendChild(o);
  });
}

function renderAccount(acc) {
  try {
    const a = acc.account || {};
    state.equity = Number(a.NAV || a.equity || 0);
    state.accountCurrency = String(a.currency || state.accountCurrency || 'USD');
    // Update currency labels
    el('accCcyLabel').textContent = `(${state.accountCurrency})`;
    el('accCcyShort').textContent = state.accountCurrency;
    const targetInput = el('targetCapital');
    if (targetInput && !targetInput.placeholder) { targetInput.placeholder = state.accountCurrency; }
    el('equityHint').textContent = `Equity: ${fmt(state.equity)} ${state.accountCurrency}`;
    const kv = el('accountSummary');
    kv.innerHTML = '';
    const add = (k,v) => { const kdv = document.createElement('div'); kdv.textContent = k; kv.appendChild(kdv); const vdv = document.createElement('div'); vdv.textContent = v; kv.appendChild(vdv); };
    add('Balance', fmt(a.balance));
    add('NAV', fmt(a.NAV || a.equity));
    add('Unrealized P/L', fmt(a.unrealizedPL));
    add('Margin Used', fmt(a.marginUsed));
    add('Margin Available', fmt(a.marginAvailable));
    add('Open Trades', (a.openTradeCount||0));
  } catch {}
}

function renderPositions(list) {
  const tbody = el('positionsTable').querySelector('tbody');
  tbody.innerHTML = '';
  (list||[]).forEach((p) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${p.instrument}</td>
      <td>${p.side}</td>
      <td>${p.units}</td>
      <td>${fmt(p.avgPrice)}</td>
      <td>${fmt(p.unrealizedPL)}</td>
      <td><button class="btn" data-ins="${p.instrument}">平倉</button></td>
    `;
    tr.querySelector('button').onclick = async (e) => {
      const ins = e.target.getAttribute('data-ins');
      try { await api(`/session/close_position?instrument=${encodeURIComponent(ins)}`, { method: 'POST' }); refreshPositions(); } catch {}
    };
    tbody.appendChild(tr);
  });
}

function renderTrades(list) {
  const tbody = el('tradesTable').querySelector('tbody');
  tbody.innerHTML = '';
  (list||[]).forEach((t) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${t.oanda_trade_id || ''}</td>
      <td>${t.instrument}</td>
      <td>${t.units}</td>
      <td>${fmt(t.entry_price)}</td>
      <td>${t.status}</td>
      <td>${fmt(t.close_price)}</td>
      <td>${fmt(t.pnl)}</td>
      <td>${t.open_timestamp || ''}</td>
      <td>${t.close_timestamp || ''}</td>
    `;
    tbody.appendChild(tr);
  });
  // Simple performance summary
  const closed = (list||[]).filter(x => x.status === 'CLOSED');
  const pnl = closed.reduce((a,b)=> a + (Number(b.pnl)||0), 0);
  const wins = closed.filter(x => (Number(x.pnl)||0) > 0).length;
  const total = closed.length || 1;
  const kv = el('perfSummary');
  kv.innerHTML = '';
  const add = (k,v) => { const kdv = document.createElement('div'); kdv.textContent = k; kv.appendChild(kdv); const vdv = document.createElement('div'); vdv.textContent = v; kv.appendChild(vdv); };
  add('Realized PnL', fmt(pnl));
  add('Closed Trades', closed.length);
  add('Win Rate', `${fmt(100*wins/total,1)}%`);
}

async function refreshSymbols() {
  // Prefer catalog
  try {
    const cat = await api('/symbols/catalog');
    state.catalog = cat.catalog || state.catalog;
    // Preserve selection against full catalog set
    const all = Object.values(state.catalog).flat();
    state.selectedSymbols = new Set([...state.selectedSymbols].filter(s => all.includes(s)));
    renderCatalog();
  } catch {
    // Fallback to flat available list
    let res;
    try { res = await api('/symbols/available'); }
    catch { res = await api('/symbols/training'); }
    state.symbols = res.symbols || [];
    state.selectedSymbols = new Set([...state.selectedSymbols].filter(s => state.symbols.includes(s)));
    // Put all into Others if no catalog
    state.catalog = { FX: [], Indices: [], Metals: [], Others: state.symbols };
    renderCatalog();
  }
}

async function refreshModels() {
  const n = state.selectedSymbols.size || 1;
  const items = await api(`/models?min_symbols=${n}`);
  state.models = items;
  renderModels();
}

async function refreshAccount() {
  const acc = await api('/account/summary');
  renderAccount(acc);
}

async function refreshPositions() {
  const pos = await api('/account/positions');
  renderPositions(pos);
}

async function refreshTrades() {
  const tr = await api('/account/trades?limit=200');
  renderTrades(tr);
  updateCharts(tr);
}

async function refreshRisk() {
  const r = await api('/config/risk');
  const rm = (r && r.risk_management) || {};
  state.risk = rm;
  // Fill fields
  el('risk_max_exposure').value = rm.max_total_exposure_usd ?? '';
  el('risk_max_risk_pct').value = rm.max_risk_per_trade_percent ?? '';
  el('risk_use_atr').value = String(rm.use_atr_sizing ?? false);
  el('risk_atr_period').value = rm.atr_period ?? '';
  el('risk_sl_atr_mult').value = rm.stop_loss_atr_multiplier ?? '';
  el('risk_tp_atr_mult').value = rm.take_profit_atr_multiplier ?? '';
  el('risk_sl_pips').value = rm.stop_loss_pips ?? '';
  el('risk_tp_pips').value = rm.take_profit_pips ?? '';
}

async function applySessionConfig() {
  const body = {
    symbols: [...state.selectedSymbols],
    model_path: el('modelSelect').value || null,
    target_capital_usd: Number(el('targetCapital').value || 0) || null,
    risk: collectRiskParams(),
  };
  await api('/session/config', { method: 'POST', body: JSON.stringify(body) });
}

function collectRiskParams() {
  const o = {};
  const put = (k,v) => { if (v !== '' && v !== null && v !== undefined) o[k]=v; };
  put('max_total_exposure_usd', Number(el('risk_max_exposure').value||0) || null);
  put('max_risk_per_trade_percent', Number(el('risk_max_risk_pct').value||0) || null);
  put('use_atr_sizing', el('risk_use_atr').value === 'true');
  put('atr_period', Number(el('risk_atr_period').value||0) || null);
  put('stop_loss_atr_multiplier', Number(el('risk_sl_atr_mult').value||0) || null);
  put('take_profit_atr_multiplier', Number(el('risk_tp_atr_mult').value||0) || null);
  put('stop_loss_pips', Number(el('risk_sl_pips').value||0) || null);
  put('take_profit_pips', Number(el('risk_tp_pips').value||0) || null);
  return o;
}

async function startSession() {
  await api('/session/start', { method: 'POST' });
  state.running = true;
  renderEngineStatus();
}

async function stopSession() {
  await api('/session/stop', { method: 'POST' });
  state.running = false;
  renderEngineStatus();
}

async function emergencyStop() {
  if (!confirm('緊急停止將平倉所有倉位並停止策略，確定？')) return;
  await api('/session/emergency_stop', { method: 'POST' });
  state.running = false;
  renderEngineStatus();
  await refreshPositions();
  await refreshAccount();
}

async function runSmoke() {
  const units = Number(el('smokeUnits').value || 1);
  if (!confirm('此為測試工具，只適用於 OANDA Practice。確定執行？')) return;
  await api(`/test/smoke_trade?units=${units}`, { method: 'POST' });
  // refresh after a short delay
  setTimeout(()=>{ refreshTrades(); refreshPositions(); refreshAccount(); }, 1500);
}

function connectWS() {
  try {
    const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${wsProto}://${location.host}/ws`);
    ws.onmessage = (evt) => {
      try {
        const obj = JSON.parse(evt.data);
        if (obj.type === 'status') {
          state.running = !!(obj.data && obj.data.running);
          renderEngineStatus();
          // Warmup info
          refreshSessionStatus();
          // Suspended alerts
          try { renderAlerts(obj.data && obj.data.suspended || {}); } catch {}
        } else if (obj.type === 'account') {
          renderAccount(obj.data);
        }
      } catch {}
    };
  } catch {}
}

function renderAlerts(map) {
  const card = el('alertsCard');
  const wrap = el('alerts');
  wrap.innerHTML = '';
  const entries = Object.entries(map||{});
  if (!entries.length) { card.style.display = 'none'; return; }
  card.style.display = '';
  entries.forEach(([sym, info]) => {
    const k = document.createElement('div'); k.textContent = sym; wrap.appendChild(k);
    const v = document.createElement('div'); v.textContent = (info && info.reason) ? info.reason : 'suspended'; wrap.appendChild(v);
  });
}

async function refreshSessionStatus() {
  try {
    const s = await api('/session/status');
    const elw = el('warmupStatus');
    if (s.warmup && s.warmup.ready) {
      elw.textContent = 'Warmup: 完成';
    } else if (s.warmup) {
      elw.textContent = `Warmup: 進行中，缺少 ${ (s.warmup.missing||[]).join(', ') }`;
    } else {
      elw.textContent = 'Warmup: -';
    }
  } catch {}
}

function bindEvents() {
  ['selFX','selIndices','selMetals','selOthers'].forEach(id => {
    const sel = el(id); if (!sel) return;
    sel.onchange = recomputeSelectionFromUI;
  });
  el('btnApply').onclick = async () => { try { await applySessionConfig(); alert('設定已套用'); } catch (e) { alert('設定失敗: ' + e); } };
  el('btnRiskUpdate').onclick = async () => {
    try {
      const body = collectRiskParams();
      await api('/risk/update', { method: 'POST', body: JSON.stringify(body) });
      alert('風險參數已更新');
    } catch(e) { alert('更新失敗: ' + e); }
  };
  el('btnStart').onclick = async () => { try { await startSession(); } catch(e) { alert('啟動失敗: ' + e);} };
  el('btnStop').onclick = async () => { try { await stopSession(); } catch(e) { alert('停止失敗: ' + e);} };
  el('btnEmergency').onclick = async () => { try { await emergencyStop(); } catch(e) { alert('緊急停止失敗: ' + e);} };
  el('btnSmoke').onclick = async () => { try { await runSmoke(); } catch(e) { alert('測試失敗: ' + e);} };
  el('modelSelect').onchange = (e) => { state.selectedModelPath = e.target.value || null; };
}

async function boot() {
  bindEvents();
  connectWS();
  await refreshSymbols();
  await refreshModels();
  await refreshAccount();
  await refreshPositions();
  await refreshTrades();
  await refreshRisk();
  setInterval(refreshPositions, 5000);
  setInterval(refreshTrades, 7000);
  setInterval(refreshSessionStatus, 5000);
}

boot();

// ----------------- Charts -----------------
let equityChart, symbolPerfChart;

function updateCharts(trades) {
  try {
    const closed = (trades||[]).filter(t => t.status === 'CLOSED' && t.close_timestamp);
    const points = closed
      .map(t => ({ t: new Date(t.close_timestamp), pnl: Number(t.pnl||0) }))
      .sort((a,b) => a.t - b.t);
    let acc = 0;
    const labels = points.map(p => p.t.toLocaleString());
    const values = points.map(p => (acc += p.pnl));

    // Equity chart (cumulative realized PnL)
    const ctx1 = document.getElementById('equityChart').getContext('2d');
    if (!equityChart) {
      equityChart = new Chart(ctx1, {
        type: 'line',
        data: { labels, datasets: [{ label: 'Cumulative PnL', data: values, borderColor: '#2ea0ff', backgroundColor: 'rgba(46,160,255,0.2)', tension: 0.2 }]},
        options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#9bb0c3'} }, y: { ticks: { color: '#9bb0c3'} } } }
      });
    } else {
      equityChart.data.labels = labels; equityChart.data.datasets[0].data = values; equityChart.update();
    }

    // Symbol performance (realized PnL per symbol)
    const bySym = {};
    closed.forEach(t => { const s=t.instrument; bySym[s] = (bySym[s]||0) + (Number(t.pnl)||0); });
    const slabels = Object.keys(bySym);
    const svalues = slabels.map(k => bySym[k]);
    const ctx2 = document.getElementById('symbolPerfChart').getContext('2d');
    if (!symbolPerfChart) {
      symbolPerfChart = new Chart(ctx2, {
        type: 'bar',
        data: { labels: slabels, datasets: [{ label: 'PnL', data: svalues, backgroundColor: '#24c47e' }]},
        options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#9bb0c3'} }, y: { ticks: { color: '#9bb0c3'} } } }
      });
    } else {
      symbolPerfChart.data.labels = slabels; symbolPerfChart.data.datasets[0].data = svalues; symbolPerfChart.update();
    }
  } catch {}
}
