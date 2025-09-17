const state = {
  catalog: {},
  pendingSymbols: new Set(),
  serverSymbols: new Set(),
  sessionDirty: false,
  models: [],
  pendingModelPath: '',
  serverModelPath: '',
  pendingTargetCapital: null,
  accountCurrency: 'USD',
  accountEquity: null,
  risk: {},
  engineRunning: false,
  warmup: { ready: false, missing: [] },
  suspended: [],
  lastHeartbeat: null,
  positions: [],
  trades: [],
  alerts: [],
  charts: { equity: null, symbol: null, perSymbol: {} },
  ws: null,
};

const $ = (id) => document.getElementById(id);
const fmtNumber = (value, decimals = 2) => {
  const num = Number(value);
  if (!Number.isFinite(num)) return '-';
  return num.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
};
const fmtCurrency = (value, decimals = 2) => {
  const num = Number(value);
  if (!Number.isFinite(num)) return '-';
  try {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: state.accountCurrency || 'USD',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(num);
  } catch (err) {
    return fmtNumber(num, decimals);
  }
};
const fmtPercent = (value, decimals = 2) => {
  const num = Number(value);
  if (!Number.isFinite(num)) return '-';
  return `${fmtNumber(num, decimals)}%`;
};
const parseNumeric = (value) => {
  if (value === null || value === undefined) return null;
  const trimmed = String(value).trim();
  if (trimmed === '') return null;
  const num = Number(trimmed);
  return Number.isFinite(num) ? num : null;
};

const normalizeInstrument = (value) => String(value || '').toUpperCase();
const toNormalizedSet = (items) => {
  const set = new Set();
  (items || []).forEach((item) => {
    const normalized = normalizeInstrument(item);
    if (normalized) set.add(normalized);
  });
  return set;
};
const cloneSet = (src) => new Set(src ? [...src] : []);
const setsEqual = (a, b) => {
  if (!a || !b) return (!a || !a.size) && (!b || !b.size);
  if (a.size !== b.size) return false;
  for (const value of a) {
    if (!b.has(value)) return false;
  }
  return true;
};
const getDisplaySymbols = () => {
  if (state.sessionDirty && state.pendingSymbols.size) return state.pendingSymbols;
  if (state.serverSymbols && state.serverSymbols.size) return state.serverSymbols;
  return new Set();
};
const filterTradesByDisplaySymbols = (trades) => {
  const symbols = getDisplaySymbols();
  if (!symbols.size) return [];
  return trades.filter((trade) => symbols.has(normalizeInstrument(trade.instrument)));
};
const getClosedTradesForDisplay = () => {
  const closed = state.trades.filter((trade) => (trade.status || '').toUpperCase() === 'CLOSED');
  return filterTradesByDisplaySymbols(closed);
};\n\nconst CATEGORY_SELECTORS = [
  { name: 'Major Pairs', id: 'symbolsMajorPairs' },
  { name: 'Minor Pairs', id: 'symbolsMinorPairs' },
  { name: 'Precious Metals', id: 'symbolsPreciousMetals' },
  { name: 'Indices', id: 'symbolsIndices' },
  { name: 'Energy', id: 'symbolsEnergy' },
  { name: 'Commodities', id: 'symbolsCommodities' },
  { name: 'Crypto', id: 'symbolsCrypto' },
  { name: 'Bonds', id: 'symbolsBonds' },
];

function ensureCategoryDropdowns() {
  const container = $('symbolDropdowns');
  if (!container) return {};
  const map = {};
  CATEGORY_SELECTORS.forEach(({ name, id }) => {
    let select = $(id);
    if (!select) {
      const wrapper = document.createElement('div');
      wrapper.className = 'form-control';
      const label = document.createElement('label');
      label.htmlFor = id;
      label.textContent = name;
      select = document.createElement('select');
      select.id = id;
      select.multiple = true;
      select.size = 8;
      wrapper.appendChild(label);
      wrapper.appendChild(select);
      container.appendChild(wrapper);
    }
    select.onchange = handleDropdownChange;
    map[id] = select;
  });
  return map;
}

function renderSymbolCatalog() {
  const dropdowns = ensureCategoryDropdowns();
  CATEGORY_SELECTORS.forEach(({ name, id }) => {
    const select = dropdowns[id];
    if (!select) return;
    const options = [...(((state.catalog || {})[name]) || [])].sort();
    const previousScroll = select.scrollTop;
    select.innerHTML = '';
    options.forEach((sym) => {
      const option = document.createElement('option');
      option.value = sym;
      option.textContent = sym;
      select.appendChild(option);
    });
    select.disabled = options.length === 0;
    select.size = Math.min(10, Math.max(4, options.length || 4));
    select.scrollTop = previousScroll;
  });
}

function applyPendingSymbolsToDropdowns() {
  const dropdowns = ensureCategoryDropdowns();
  CATEGORY_SELECTORS.forEach(({ id }) => {
    const select = dropdowns[id];
    if (!select) return;
    [...select.options].forEach((option) => {
      const normalized = normalizeInstrument(option.value);
      option.selected = state.pendingSymbols.has(normalized);
    });
  });
}


function collectSymbolsFromDropdowns() {
  const collected = new Set();
  const dropdowns = ensureCategoryDropdowns();
  CATEGORY_SELECTORS.forEach(({ id }) => {
    const select = dropdowns[id];
    if (!select) return;
    [...select.selectedOptions].forEach((option) => {
      const normalized = normalizeInstrument(option.value);
      if (normalized) collected.add(normalized);
    });
  });
  return collected;
}


function handleDropdownChange() {
  const next = collectSymbolsFromDropdowns();
  if (!setsEqual(state.pendingSymbols, next)) {
    state.pendingSymbols = next;
    markSessionDirty();
    applyPendingSymbolsToDropdowns();
    renderSelectedPreview();
    updateDirtyIndicators();
    loadModels().catch((err) => pushAlert(`Model scan failed: ${err.message}`, 'error'));
  } else {
    applyPendingSymbolsToDropdowns();
    renderSelectedPreview();
    updateDirtyIndicators();
  }
}



function removeSymbol(symbol) {
  const normalized = normalizeInstrument(symbol);
  if (!normalized || !state.pendingSymbols.has(normalized)) return;
  const next = new Set(state.pendingSymbols);
  next.delete(normalized);
  state.pendingSymbols = next;
  markSessionDirty();
  applyPendingSymbolsToDropdowns();
  renderSelectedPreview();
  updateDirtyIndicators();
  loadModels().catch((err) => pushAlert(`Model scan failed: ${err.message}`, 'error'));
}
async function loadSymbolCatalog() {
  try {
    const response = await api('/symbols/catalog');
    const incoming = response?.catalog || {};
    const normalized = {};
    Object.entries(incoming).forEach(([category, list]) => {
      normalized[category] = [...new Set((list || []).map((sym) => normalizeInstrument(sym)))];
    });
    state.catalog = normalized;
    const allSymbols = new Set(Object.values(normalized).flat());
    state.pendingSymbols = new Set([...state.pendingSymbols].filter((sym) => allSymbols.has(sym)));
    state.serverSymbols = new Set([...state.serverSymbols].filter((sym) => allSymbols.has(sym)));
    renderSymbolCatalog();
    applyPendingSymbolsToDropdowns();
    renderSelectedPreview();
    updateDirtyIndicators();
  } catch (err) {
    pushAlert(`Failed to load symbol catalog: ${err.message}`, 'error');
  }
}


async function loadModels() {
  const minSymbols = Math.max(state.pendingSymbols.size, 1);
  try {
    const items = await api(`/models?min_symbols=${minSymbols}`);
    state.models = Array.isArray(items) ? items : [];
    renderModelSelect();
  } catch (err) {
    pushAlert(`Failed to load models: ${err.message}`, 'error');
  }
}

function renderModelSelect() {
  const select = modelSelect;
  if (!select) return;
  const current = state.pendingModelPath || '';
  select.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.disabled = true;
  placeholder.hidden = true;
  placeholder.textContent = state.pendingSymbols.size ? 'Select a model...' : 'Select symbols first';
  select.appendChild(placeholder);
  const sorted = [...state.models];
  sorted.sort((a, b) => {
    const diff = (a.max_symbols || 0) - (b.max_symbols || 0);
    if (diff !== 0) return diff;
    return (a.name || '').localeCompare(b.name || '');
  });
  sorted.forEach((model) => {
    const option = document.createElement('option');
    option.value = model.path;
    option.textContent = ${model.name} | max  | ;
    select.appendChild(option);
  });
  if (current && sorted.some((m) => m.path === current)) {
    select.value = current;
  } else {
    select.value = '';
    if (current) state.pendingModelPath = '';
  }
}


async function loadRisk() {
  try {
    const response = await api('/config/risk');
    const risk = (response && response.risk_management) || {};
    state.risk = risk;
    populateRiskForm(risk);
  } catch (err) {
    pushAlert(`Failed to load risk configuration: ${err.message}`, 'error');
  }
}

function populateRiskForm(risk) {
  const setValue = (id, value) => {
    const node = $(id);
    if (!node) return;
    if (node.tagName === 'SELECT') {
      node.value = String(value ?? false);
    } else {
      node.value = value ?? '';
    }
  };
  setValue('risk_max_total_exposure', risk.max_total_exposure_usd);
  setValue('risk_max_risk_pct', risk.max_risk_per_trade_percent);
  setValue('risk_stop_loss_pips', risk.stop_loss_pips);
  setValue('risk_take_profit_pips', risk.take_profit_pips);
  setValue('risk_use_atr', risk.use_atr_sizing);
  setValue('risk_atr_period', risk.atr_period);
  setValue('risk_sl_atr_mult', risk.stop_loss_atr_multiplier);
  setValue('risk_tp_atr_mult', risk.take_profit_atr_multiplier);
}

function gatherRiskForm() {
  const readNumber = (id) => {
    const node = $(id);
    if (!node) return null;
    return parseNumeric(node.value);
  };
  const readBool = (id) => {
    const node = $(id);
    if (!node) return null;
    return node.value === 'true';
  };
  return {
    max_total_exposure_usd: readNumber('risk_max_total_exposure'),
    max_risk_per_trade_percent: readNumber('risk_max_risk_pct'),
    stop_loss_pips: readNumber('risk_stop_loss_pips'),
    take_profit_pips: readNumber('risk_take_profit_pips'),
    use_atr_sizing: readBool('risk_use_atr'),
    atr_period: readNumber('risk_atr_period'),
    stop_loss_atr_multiplier: readNumber('risk_sl_atr_mult'),
    take_profit_atr_multiplier: readNumber('risk_tp_atr_mult'),
  };
}

async function updateRisk() {
  const payload = gatherRiskForm();
  const body = {};
  Object.entries(payload).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== '') {
      body[key] = value;
    }
  });
  try {
    await api('/risk/update', { method: 'POST', body });
    state.risk = { ...state.risk, ...body };
    pushAlert('Risk parameters updated successfully.', 'success');
  } catch (err) {
    pushAlert(`Failed to update risk parameters: ${err.message}`, 'error');
  }
}

async function loadAccountSummary() {
  try {
    const data = await api('/account/summary');
    renderAccount(data);
  } catch (err) {
    pushAlert(`Failed to load account summary: ${err.message}`, 'error');
  }
}

function renderAccount(data) {
  const account = data?.account || {};
  const currency = account.currency || state.accountCurrency;
  if (currency) state.accountCurrency = String(currency).toUpperCase();
  const balance = parseNumeric(account.balance) ?? 0;
  const nav = parseNumeric(account.NAV) ?? parseNumeric(account.equity) ?? balance;
  const unrealized = parseNumeric(account.unrealizedPL) ?? 0;
  const marginUsed = parseNumeric(account.marginUsed) ?? 0;
  const marginAvail = parseNumeric(account.marginAvailable) ?? 0;
  const openTrades = parseNumeric(account.openTradeCount) ?? 0;
  state.accountEquity = nav;
  setText('peekEquity', fmtCurrency(nav));
  setText('peekBalance', fmtCurrency(balance));
  setText('peekUnrealized', fmtCurrency(unrealized));
  setText('accountBalance', fmtCurrency(balance));
  setText('accountNav', fmtCurrency(nav));
  setText('accountUnrealized', fmtCurrency(unrealized));
  setText('accountMarginUsed', fmtCurrency(marginUsed));
  setText('accountMarginAvail', fmtCurrency(marginAvail));
  setText('accountOpenTrades', openTrades ? openTrades.toString() : '0');
  const label = $('targetCapitalLabel');
  if (label) {
    label.textContent = `Target capital (${state.accountCurrency})`;
  }
  setText('footerAccountCurrency', `Account currency: ${state.accountCurrency}`);
}

async function loadPositions() {
  try {
    const positions = await api('/account/positions');
    state.positions = Array.isArray(positions) ? positions : [];
    renderPositions();
  } catch (err) {
    pushAlert(`Failed to load open positions: ${err.message}`, 'error');
  }
}

function renderPositions() {
  const tbody = document.querySelector('#positionsTable tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  if (!state.positions.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 7;
    td.style.textAlign = 'center';
    td.textContent = 'No open positions';
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  state.positions.forEach((pos) => {
    const tr = document.createElement('tr');
    const avgPrice = parseNumeric(pos.avgPrice);
    const units = parseNumeric(pos.units);
    const notional = Number.isFinite(avgPrice) && Number.isFinite(units)
      ? Math.abs(units) * avgPrice
      : null;
    tr.innerHTML = `
      <td>${pos.instrument || '-'}</td>
      <td>${pos.side || '-'}</td>
      <td>${fmtNumber(units, 0)}</td>
      <td>${fmtNumber(avgPrice)}</td>
      <td>${fmtCurrency(pos.unrealizedPL)}</td>
      <td>${notional ? fmtCurrency(notional) : '-'}</td>
      <td></td>
    `;
    const actionCell = tr.lastElementChild;
    const btn = document.createElement('button');
    btn.className = 'btn btn-muted';
    btn.textContent = 'Close';
    btn.addEventListener('click', () => closePosition(pos.instrument));
    actionCell.appendChild(btn);
    tbody.appendChild(tr);
  });
}

async function closePosition(instrument) {
  if (!instrument) return;
  try {
    await api(`/session/close_position?instrument=${encodeURIComponent(instrument)}`, { method: 'POST' });
    pushAlert(`Close request sent for ${instrument}.`, 'success');
    await Promise.all([loadPositions(), loadAccountSummary(), loadTrades()]);
  } catch (err) {
    pushAlert(`Failed to close ${instrument}: ${err.message}`, 'error');
  }
}

async function loadTrades() {
  try {
    const trades = await api('/account/trades?limit=300');
    state.trades = Array.isArray(trades) ? trades : [];
    renderTrades();
    updatePerformanceMetrics();
    updateCharts();
  } catch (err) {
    pushAlert(`Failed to load trade history: ${err.message}`, 'error');
  }
}

function renderTrades() {
  const tbody = document.querySelector('#tradesTable tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  if (!state.trades.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 9;
    td.style.textAlign = 'center';
    td.textContent = 'No trades recorded yet';
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  const sorted = [...state.trades].sort((a, b) => {
    const aTime = new Date(a.open_timestamp || a.close_timestamp || 0).getTime();
    const bTime = new Date(b.open_timestamp || b.close_timestamp || 0).getTime();
    return bTime - aTime;
  });
  sorted.forEach((trade) => {
    const tr = document.createElement('tr');
    const entryPrice = parseNumeric(trade.entry_price ?? trade.open_price ?? trade.price);
    const exitPrice = parseNumeric(trade.close_price ?? trade.exit_price);
    const pnl = parseNumeric(trade.pnl);
    const openTime = trade.open_timestamp ? new Date(trade.open_timestamp).toLocaleString() : '-';
    const closeTime = trade.close_timestamp ? new Date(trade.close_timestamp).toLocaleString() : '-';
    tr.innerHTML = `
      <td>${trade.oanda_trade_id || trade.trade_id || '-'}</td>
      <td>${trade.instrument || '-'}</td>
      <td>${fmtNumber(trade.units, 0)}</td>
      <td>${fmtNumber(entryPrice)}</td>
      <td>${fmtNumber(exitPrice)}</td>
      <td>${trade.status || '-'}</td>
      <td>${fmtCurrency(pnl)}</td>
      <td>${openTime}</td>
      <td>${closeTime}</td>
    `;
    tbody.appendChild(tr);
  });
}

function calculateMaxDrawdown(pnls) {
  let peak = 0;
  let equity = 0;
  let maxDrawdown = 0;
  pnls.forEach((pnl) => {
    equity += pnl;
    peak = Math.max(peak, equity);
    const drawdown = peak - equity;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  });
  return maxDrawdown;
}

function calculateSharpe(values) {
  if (!values.length) return null;
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
  const std = Math.sqrt(variance);
  if (!std) return null;
  return (mean / std) * Math.sqrt(values.length);
}

function updatePerformanceMetrics() {
  const closed = getClosedTradesForDisplay();
  if (!closed.length) {
    setText('metricRealized', '-');
    setText('metricAvgPnL', '-');
    setText('metricWinRate', '-');
    setText('metricTradesClosed', '0');
    setText('metricDrawdown', '-');
    setText('metricSharpe', '-');
    return;
  }
  const ordered = [...closed].sort((a, b) => {
    const at = new Date(a.close_timestamp || a.open_timestamp || 0).getTime();
    const bt = new Date(b.close_timestamp || b.open_timestamp || 0).getTime();
    return at - bt;
  });
  const pnls = ordered.map((t) => parseNumeric(t.pnl) ?? 0);
  const realized = pnls.reduce((sum, v) => sum + v, 0);
  const avg = realized / pnls.length;
  const wins = pnls.filter((v) => v > 0).length;
  const winRate = (wins / pnls.length) * 100;
  const maxDrawdown = calculateMaxDrawdown(pnls);
  const sharpe = calculateSharpe(pnls);

  setText('metricRealized', fmtCurrency(realized));
  setText('metricAvgPnL', fmtCurrency(avg));
  setText('metricWinRate', fmtPercent(winRate));
  setText('metricTradesClosed', pnls.length.toString());
  setText('metricDrawdown', fmtCurrency(maxDrawdown));
  setText('metricSharpe', sharpe !== null ? fmtNumber(sharpe, 2) : '-');
}


function resetChart(chart) {
  if (!chart) return;
  chart.data.labels = [];
  chart.data.datasets.forEach((dataset) => {
    dataset.data = [];
  });
  chart.update('none');
}

function chartLineOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (context) => `${context.dataset.label || ''}: ${fmtCurrency(context.parsed.y)}`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: '#8ea2c6', maxRotation: 0, autoSkip: true },
        grid: { color: 'rgba(78, 102, 138, 0.12)' },
      },
      y: {
        ticks: { color: '#8ea2c6' },
        grid: { color: 'rgba(78, 102, 138, 0.12)' },
      },
    },
  };
}

function updateCharts() {
  if (!window.Chart) return;
  const displaySymbols = getDisplaySymbols();
  if (!displaySymbols.size) {
    resetChart(state.charts.equity);
    resetChart(state.charts.symbol);
    updatePerSymbolCharts([]);
    return;
  }
  const closed = [...getClosedTradesForDisplay()].sort((a, b) => {
    const at = new Date(a.close_timestamp || a.open_timestamp || 0).getTime();
    const bt = new Date(b.close_timestamp || b.open_timestamp || 0).getTime();
    return at - bt;
  });
  if (!closed.length) {
    resetChart(state.charts.equity);
    resetChart(state.charts.symbol);
    updatePerSymbolCharts([]);
    return;
  }

  const equityCanvas = equityChart;
  if (equityCanvas) {
    const ctx = equityCanvas.getContext('2d');
    let cumulative = 0;
    const labels = closed.map((t) => new Date(t.close_timestamp || t.open_timestamp || Date.now()).toLocaleString());
    const values = closed.map((t) => {
      cumulative += parseNumeric(t.pnl) ?? 0;
      return Number(cumulative.toFixed(2));
    });
    if (!state.charts.equity) {
      state.charts.equity = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: 'Cumulative realized PnL',
              data: values,
              borderColor: '#4c9dff',
              backgroundColor: 'rgba(76, 157, 255, 0.15)',
              fill: true,
              tension: 0.25,
            },
          ],
        },
        options: chartLineOptions(),
      });
    } else {
      state.charts.equity.data.labels = labels;
      state.charts.equity.data.datasets[0].data = values;
      state.charts.equity.update('none');
    }
  }

  const symbolCanvas = symbolPerfChart;
  if (symbolCanvas) {
    const ctx = symbolCanvas.getContext('2d');
    const totals = new Map();
    closed.forEach((trade) => {
      const key = normalizeInstrument(trade.instrument) || 'UNKNOWN';
      totals.set(key, (totals.get(key) || 0) + (parseNumeric(trade.pnl) ?? 0));
    });
    const labels = [...totals.keys()].sort((a, b) => Math.abs(totals.get(b)) - Math.abs(totals.get(a)));
    const values = labels.map((label) => Number((totals.get(label) || 0).toFixed(2)));
    const colors = values.map((val) => (val >= 0 ? 'rgba(36, 196, 142, 0.65)' : 'rgba(255, 92, 92, 0.65)'));
    if (!state.charts.symbol) {
      state.charts.symbol = new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            {
              label: 'Realized PnL',
              data: values,
              backgroundColor: colors,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          indexAxis: 'y',
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (context) => ${context.label}: ,
              },
            },
          },
          scales: {
            x: {
              ticks: { color: '#8ea2c6' },
              grid: { color: 'rgba(78, 102, 138, 0.12)' },
            },
            y: {
              ticks: { color: '#8ea2c6' },
              grid: { display: false },
            },
          },
        },
      });
    } else {
      state.charts.symbol.data.labels = labels;
      state.charts.symbol.data.datasets[0].data = values;
      state.charts.symbol.data.datasets[0].backgroundColor = colors;
      state.charts.symbol.update('none');
    }
  }

  updatePerSymbolCharts(closed);
}


function syncPerSymbolChartContainers(symbols) {
  const wrapper = $('perSymbolCharts');
  if (!wrapper) return;
  const desired = new Set(symbols);
  const existingNodes = Array.from(wrapper.querySelectorAll('[data-symbol-card]'));
  existingNodes.forEach((node) => {
    const sym = node.dataset.symbolCard;
    if (!desired.has(sym)) {
      const chart = state.charts.perSymbol[sym];
      if (chart) {
        chart.destroy();
        delete state.charts.perSymbol[sym];
      }
      node.remove();
    }
  });
  symbols.forEach((sym) => {
    if (!wrapper.querySelector(`[data-symbol-card="${sym}"]`)) {
      const card = document.createElement('div');
      card.className = 'chart-card';
      card.dataset.symbolCard = sym;
      const label = document.createElement('label');
      label.textContent = sym;
      const canvas = document.createElement('canvas');
      card.appendChild(label);
      card.appendChild(canvas);
      wrapper.appendChild(card);
    }
  });
}

function updatePerSymbolCharts(closed) {
  if (!window.Chart) return;
  const displaySymbols = [...getDisplaySymbols()].sort();
  const wrapper = $('perSymbolCharts');
  if (!wrapper) return;
  if (!displaySymbols.length) {
    wrapper.innerHTML = '';
    Object.values(state.charts.perSymbol).forEach((chart) => {
      if (chart) chart.destroy();
    });
    state.charts.perSymbol = {};
    return;
  }
  syncPerSymbolChartContainers(displaySymbols);
  displaySymbols.forEach((sym) => {
    const card = wrapper.querySelector(`[data-symbol-card="${sym}"]`);
    if (!card) return;
    const canvas = card.querySelector('canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const series = closed
      .filter((trade) => normalizeInstrument(trade.instrument) === sym)
      .sort((a, b) => {
        const at = new Date(a.close_timestamp || a.open_timestamp || 0).getTime();
        const bt = new Date(b.close_timestamp || b.open_timestamp || 0).getTime();
        return at - bt;
      });
    if (!series.length) {
      const chart = state.charts.perSymbol[sym];
      if (chart) resetChart(chart);
      return;
    }
    let cumulative = 0;
    const labels = series.map((trade) => new Date(trade.close_timestamp || trade.open_timestamp || Date.now()).toLocaleString());
    const values = series.map((trade) => {
      cumulative += parseNumeric(trade.pnl) ?? 0;
      return Number(cumulative.toFixed(2));
    });
    if (!state.charts.perSymbol[sym]) {
      state.charts.perSymbol[sym] = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: sym,
              data: values,
              borderColor: '#8e7dff',
              backgroundColor: 'rgba(142, 125, 255, 0.18)',
              fill: true,
              tension: 0.25,
            },
          ],
        },
        options: chartLineOptions(),
      });
    } else {
      const chart = state.charts.perSymbol[sym];
      chart.data.labels = labels;
      chart.data.datasets[0].data = values;
      chart.update('none');
    }
  });
}



async function applySession() {
  if (!state.pendingSymbols.size) {
    pushAlert('Select at least one symbol before applying.', 'error');
    return;
  }
  const symbols = Array.from(state.pendingSymbols);
  const modelPath = $('modelSelect')?.value || '';
  const targetInput = $('inputTargetCapital');
  const targetValue = targetInput ? parseNumeric(targetInput.value) : null;
  const risk = gatherRiskForm();
  const riskPayload = {};
  Object.entries(risk).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== '') {
      riskPayload[key] = value;
    }
  });
  const body = {
    symbols,
    model_path: modelPath || null,
    target_capital_usd: targetValue !== null ? targetValue : null,
    risk: riskPayload,
  };
  try {
    await api('/session/config', { method: 'POST', body });
    pushAlert('Session configuration applied.', 'success');
    state.serverSymbols = new Set(symbols);
    state.pendingSymbols = new Set(symbols);
    state.serverModelPath = modelPath;
    state.pendingModelPath = modelPath;
    clearSessionDirty();
    renderSymbolCatalog();
    applyPendingSymbolsToDropdowns();
    renderSelectedPreview();
    await Promise.all([loadModels(), refreshStatus()]);
  } catch (err) {
    pushAlert(`Failed to apply session configuration: ${err.message}`, 'error');
  }
}

async function startEngine() {
  const seconds = parseNumeric($('coldStartSeconds')?.value) ?? 120;
  const timeout = Math.max(30, Math.floor(seconds));
  try {
    await api(`/session/start?cold_start_timeout_sec=${timeout}`, { method: 'POST' });
    pushAlert('Cold start initiated. Engine will activate after buffers are warm.', 'success');
    await refreshStatus();
  } catch (err) {
    pushAlert(`Failed to start engine: ${err.message}`, 'error');
  }
}

async function stopEngine() {
  try {
    await api('/session/stop', { method: 'POST' });
    pushAlert('Engine stop requested.', 'warning');
    await refreshStatus();
  } catch (err) {
    pushAlert(`Failed to stop engine: ${err.message}`, 'error');
  }
}

async function emergencyStop() {
  try {
    await api('/session/emergency_stop', { method: 'POST' });
    pushAlert('Emergency stop executed. All positions are being flattened.', 'error');
    await Promise.all([loadPositions(), loadAccountSummary(), refreshStatus()]);
  } catch (err) {
    pushAlert(`Emergency stop failed: ${err.message}`, 'error');
  }
}

async function runSmokeTest() {
  const units = parseNumeric($('smokeUnits')?.value) ?? 1;
  try {
    const res = await api(`/test/smoke_trade?units=${units}`, { method: 'POST' });
    const symbols = res?.symbols || [];
    pushAlert(`Smoke test submitted for ${symbols.length} symbol(s).`, 'success');
  } catch (err) {
    pushAlert(`Smoke test failed: ${err.message}`, 'error');
  }
}

async function refreshStatus() {
  try {
    const status = await api('/session/status');
    handleStatus(status || {});
  } catch (err) {
    pushAlert(`Failed to refresh session status: ${err.message}`, 'error');
  }
}

function handleStatus(data) {
  state.engineRunning = !!data.running;
  state.warmup = data.warmup || state.warmup;
  updateSuspendedLabel(data.suspended);

  const newServerSymbols = toNormalizedSet(data.selected_symbols);
  const serverChanged = !setsEqual(state.serverSymbols, newServerSymbols);
  if (serverChanged) {
    state.serverSymbols = newServerSymbols;
  }
  if (!state.sessionDirty && serverChanged && !setsEqual(state.pendingSymbols, newServerSymbols)) {
    state.pendingSymbols = cloneSet(newServerSymbols);
    renderSymbolCatalog();
    applyPendingSymbolsToDropdowns();
    renderSelectedPreview();
    clearSessionDirty();
  }

  const newModelPath = data.model || '';
  if (state.serverModelPath !== newModelPath) {
    state.serverModelPath = newModelPath;
  }
  if (!state.sessionDirty && state.pendingModelPath !== newModelPath) {
    state.pendingModelPath = newModelPath;
    renderModelSelect();
  }

  updateEngineStatus();
  updateWarmupBadge();
  updateModelBadge();
  state.lastHeartbeat = new Date();
  updateHeartbeatDisplay();
}



function bindEvents() {
  ensureCategoryDropdowns();
  const applyBtn = $('applySessionBtn');
  if (applyBtn) applyBtn.addEventListener('click', () => applySession());
  const riskBtn = $('updateRiskBtn');
  if (riskBtn) riskBtn.addEventListener('click', () => updateRisk());
  const startBtn = $('startEngineBtn');
  if (startBtn) startBtn.addEventListener('click', () => startEngine());
  const stopBtn = $('stopEngineBtn');
  if (stopBtn) stopBtn.addEventListener('click', () => stopEngine());
  const emergencyBtn = $('btnEmergency');
  if (emergencyBtn) emergencyBtn.addEventListener('click', () => emergencyStop());
  const smokeBtn = $('smokeTestBtn');
  if (smokeBtn) smokeBtn.addEventListener('click', () => runSmokeTest());
  const modelSelect = $('modelSelect');
  if (modelSelect) {
    modelSelect.addEventListener('change', (event) => {
      state.pendingModelPath = event.target.value || '';
      markSessionDirty();
    });
  }
  const targetInput = $('inputTargetCapital');
  if (targetInput) {
    targetInput.addEventListener('input', () => {
      state.pendingTargetCapital = parseNumeric(targetInput.value);
      markSessionDirty();
    });
  }
}

async function boot() {
  bindEvents();
  updateDirtyIndicators();
  try { await loadSymbolCatalog(); } catch (err) { pushAlert(err.message, 'error'); }
  try { await loadRisk(); } catch (err) { pushAlert(err.message, 'error'); }
  try { await refreshStatus(); } catch (err) { pushAlert(err.message, 'error'); }
  renderSymbolCatalog();
  applyPendingSymbolsToDropdowns();
  renderSelectedPreview();
  updateDirtyIndicators();
  try { await loadAccountSummary(); } catch (err) { pushAlert(err.message, 'error'); }
  try { await loadPositions(); } catch (err) { pushAlert(err.message, 'error'); }
  try { await loadTrades(); } catch (err) { pushAlert(err.message, 'error'); }
  await loadModels();
  connectWebSocket();
  setInterval(loadAccountSummary, 8000);
  setInterval(loadPositions, 6000);
  setInterval(loadTrades, 12000);
  setInterval(refreshStatus, 7000);
}

document.addEventListener('DOMContentLoaded', boot);

















