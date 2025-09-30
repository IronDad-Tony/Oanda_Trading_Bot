const API_BASE = window.location.origin;

const state = {
  symbolsCatalog: {},
  selectedSymbols: [],
  models: [],
  selectedModelPath: '',
  targetCapital: null,
  riskSettings: {},
  riskSnapshot: {},
  accountSummary: {},
  accountCurrency: 'USD',
  positions: [],
  trades: [],
  analytics: null,
  charts: {
    equity: null,
    symbol: null,
  },
  chartShells: {
    equity: null,
    symbol: null,
  },
  ws: null,
  lastHeartbeat: null,
  engineRunning: false,
  warmup: { ready: false, missing: [] },
  suspended: {},
  sessionDirty: false,
  alerts: [],
  pollingHandles: [],
  lastHaltReason: null,
};

const elements = {};

function cacheElements() {
  elements.symbolControls = document.getElementById('symbolControls');
  elements.selectedSymbolsList = document.getElementById('selectedSymbolsList');
  elements.modelSelect = document.getElementById('modelSelect');
  elements.modelDetails = document.getElementById('modelDetails');
  elements.targetCapitalInput = document.getElementById('targetCapitalInput');
  elements.applySessionBtn = document.getElementById('applySessionBtn');
  elements.refreshSessionBtn = document.getElementById('refreshSessionBtn');
  elements.sessionDirtyBadge = document.getElementById('sessionDirtyBadge');
  elements.startEngineBtn = document.getElementById('startEngineBtn');
  elements.stopEngineBtn = document.getElementById('stopEngineBtn');
  elements.emergencyStopBtn = document.getElementById('emergencyStopBtn');
  elements.smokeTestBtn = document.getElementById('smokeTestBtn');
  elements.statusMessage = document.getElementById('statusMessage');
  elements.engineStatusIndicator = document.getElementById('engineStatusIndicator');
  elements.engineStatusLabel = document.getElementById('engineStatusLabel');
  elements.warmupBadge = document.getElementById('warmupBadge');
  elements.heartbeatBadge = document.getElementById('heartbeatBadge');
  elements.riskInputs = document.querySelectorAll('[data-risk-key]');
  elements.riskUpdateBtn = document.getElementById('riskUpdateBtn');
  elements.accountEquity = document.getElementById('accountEquity');
  elements.accountBalance = document.getElementById('accountBalance');
  elements.accountUnrealized = document.getElementById('accountUnrealized');
  elements.accountMarginAvailable = document.getElementById('accountMarginAvailable');
  elements.accountMarginUsed = document.getElementById('accountMarginUsed');
  elements.accountCurrency = document.getElementById('accountCurrency');
  elements.totalExposureValue = document.getElementById('totalExposureValue');
  elements.totalExposureBar = document.getElementById('totalExposureBar');
  elements.exposureCapacityLabel = document.getElementById('exposureCapacityLabel');
  elements.exposureList = document.getElementById('exposureList');
  elements.perfTotalPnl = document.getElementById('perfTotalPnl');
  elements.perfWinRate = document.getElementById('perfWinRate');
  elements.perfDrawdown = document.getElementById('perfDrawdown');
  elements.perfTradeCount = document.getElementById('perfTradeCount');
  elements.refreshAnalyticsBtn = document.getElementById('refreshAnalyticsBtn');
  elements.positionsTableBody = document.getElementById('positionsTableBody');
  elements.equityChartShell = document.querySelector('[data-chart="equity"] .chart-shell');
  elements.symbolChartShell = document.querySelector('[data-chart="symbol"] .chart-shell');
  state.chartShells.equity = elements.equityChartShell;
  state.chartShells.symbol = elements.symbolChartShell;
  elements.positionsCountBadge = document.getElementById('positionsCountBadge');
  elements.tradesTableBody = document.getElementById('tradesTableBody');
  elements.tradesCountBadge = document.getElementById('tradesCountBadge');
  elements.alertsContainer = document.getElementById('alertsContainer');
}

function bindEvents() {
  elements.applySessionBtn.addEventListener('click', applySession);
  elements.refreshSessionBtn.addEventListener('click', refreshStatus);
  elements.startEngineBtn.addEventListener('click', startEngine);
  elements.stopEngineBtn.addEventListener('click', stopEngine);
  elements.emergencyStopBtn.addEventListener('click', emergencyStop);
  elements.smokeTestBtn.addEventListener('click', runSmokeTest);
  elements.refreshAnalyticsBtn.addEventListener('click', loadAnalytics);

  elements.modelSelect.addEventListener('change', (event) => {
    state.selectedModelPath = event.target.value || '';
    markSessionDirty();
    const model = state.models.find((m) => m.path === state.selectedModelPath);
    renderModelDetails(model);
  });

  elements.targetCapitalInput.addEventListener('input', () => {
    const value = parseFloat(elements.targetCapitalInput.value);
    state.targetCapital = Number.isFinite(value) ? value : null;
    markSessionDirty();
  });

  elements.riskInputs.forEach((input) => {
    const handler = () => markSessionDirty();
    input.addEventListener('input', handler);
    if (input.type === 'checkbox') {
      input.addEventListener('change', handler);
    }
  });

  elements.riskUpdateBtn.addEventListener('click', updateRiskSettings);
}

async function api(path, options = {}) {
  const opts = { ...options };
  opts.headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };
  const response = await fetch(`${API_BASE}${path}`, opts);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

function fmtNumber(value, digits = 2) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '—';
  return num.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

function fmtCurrency(value, digits = 2, currency = state.accountCurrency || 'USD') {
  const num = Number(value);
  if (!Number.isFinite(num)) return '—';
  try {
    return new Intl.NumberFormat(undefined, {
      style: 'currency',
      currency,
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }).format(num);
  } catch (error) {
    return `${currency} ${fmtNumber(num, digits)}`;
  }
}

function fmtPercent(value, digits = 1) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '—';
  return `${fmtNumber(num, digits)}%`;
}

function markSessionDirty() {
  state.sessionDirty = true;
  elements.sessionDirtyBadge.classList.remove('hidden');
  elements.applySessionBtn.classList.add('needs-save');
}

function clearSessionDirty() {
  state.sessionDirty = false;
  elements.sessionDirtyBadge.classList.add('hidden');
  elements.applySessionBtn.classList.remove('needs-save');
}

function pushAlert(message, type = 'info') {
  const entry = {
    id: Date.now() + Math.random(),
    message,
    type,
    timestamp: new Date().toISOString(),
  };
  state.alerts.unshift(entry);
  state.alerts = state.alerts.slice(0, 20);
  renderAlerts();
}

function renderAlerts() {
  elements.alertsContainer.innerHTML = '';
  state.alerts.forEach((alert) => {
    const item = document.createElement('div');
    item.className = `alert-item ${alert.type}`;
    const time = new Date(alert.timestamp).toLocaleTimeString();
    item.textContent = `[${time}] ${alert.message}`;
    elements.alertsContainer.appendChild(item);
  });
}

async function loadSymbolCatalog() {
  const data = await api('/symbols/catalog');
  state.symbolsCatalog = data?.catalog || {};
  renderSymbolCatalog();
}

function renderSymbolCatalog() {
  const container = elements.symbolControls;
  container.innerHTML = '';
  const categories = Object.keys(state.symbolsCatalog);
  categories.sort();
  categories.forEach((category) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'symbol-category';

    const label = document.createElement('label');
    label.textContent = category;

    const select = document.createElement('select');
    select.multiple = true;
    select.dataset.category = category;
    select.className = 'symbol-select';

    const symbols = state.symbolsCatalog[category] || [];
    symbols.forEach((symbol) => {
      const option = document.createElement('option');
      option.value = symbol;
      option.textContent = symbol;
      option.selected = state.selectedSymbols.includes(symbol);
      select.appendChild(option);
    });

    select.addEventListener('change', handleSymbolSelectionChange);

    wrapper.appendChild(label);
    wrapper.appendChild(select);
    container.appendChild(wrapper);
  });

  updateSelectedSymbolsDisplay();
  adjustSymbolChartHeight(state.selectedSymbols.length);
}


function adjustSymbolChartHeight(count = 0) {
  const shell = state.chartShells.symbol || elements.symbolChartShell;
  if (!shell) return;
  const effectiveCount = Math.max(count, state.selectedSymbols.length, 1);
  const base = 260;
  const perSymbol = 22;
  const max = 560;
  const computed = Math.max(base, Math.min(max, base + Math.max(0, effectiveCount - 6) * perSymbol));
  if (shell.style.height !== `${computed}px`) {
    shell.style.height = `${computed}px`;
  }
  if (state.charts.symbol) {
    state.charts.symbol.resize();
  }
}

function handleSymbolSelectionChange() {
  const selects = elements.symbolControls.querySelectorAll('select');
  const selected = new Set();
  selects.forEach((select) => {
    Array.from(select.selectedOptions).forEach((option) => selected.add(option.value));
  });
  state.selectedSymbols = Array.from(selected);
  updateSelectedSymbolsDisplay();
  adjustSymbolChartHeight(state.selectedSymbols.length);
  markSessionDirty();
  refreshModelList();
}

function updateSelectedSymbolsDisplay() {
  const container = elements.selectedSymbolsList;
  container.innerHTML = '';
  if (!state.selectedSymbols.length) {
    const hint = document.createElement('span');
    hint.className = 'section-help';
    hint.textContent = 'No symbols selected.';
    container.appendChild(hint);
    return;
  }
  state.selectedSymbols.sort().forEach((symbol) => {
    const chip = document.createElement('span');
    chip.className = 'symbol-chip';
    chip.textContent = symbol;
    container.appendChild(chip);
  });
}

async function refreshModelList() {
  const minSymbols = Math.max(1, state.selectedSymbols.length);
  state.models = await api(`/models?min_symbols=${minSymbols}`);
  renderModelSelect();
}

function renderModelSelect() {
  const select = elements.modelSelect;
  const current = state.selectedModelPath;
  select.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = 'Select a model';
  select.appendChild(placeholder);

  state.models.forEach((model) => {
    const option = document.createElement('option');
    option.value = model.path;
    option.textContent = `${model.name} (<= ${model.max_symbols} symbols)`;
    if (model.path === current) {
      option.selected = true;
    }
    select.appendChild(option);
  });

  const selectedModel = state.models.find((m) => m.path === current);
  renderModelDetails(selectedModel);
}

function renderModelDetails(model) {
  if (!model) {
    elements.modelDetails.innerHTML = '<p class="section-help">Select a model compatible with the chosen symbol count.</p>';
    return;
  }
  elements.modelDetails.innerHTML = `
    <div class="model-summary">
      <div><strong>Type:</strong> ${model.type}</div>
      <div><strong>Max Symbols:</strong> ${model.max_symbols}</div>
      <div><strong>Lookback:</strong> ${model.lookback_window} bars</div>
    </div>
  `;
}

async function loadModels() {
  state.models = await api('/models?min_symbols=1');
  renderModelSelect();
}

async function loadRiskConfig() {
  const data = await api('/config/risk');
  state.riskSettings = data?.risk_management || {};
  state.riskSnapshot = data?.snapshot || {};
  if (state.riskSnapshot.account_currency) {
    state.accountCurrency = state.riskSnapshot.account_currency;
  }
  populateRiskControls(state.riskSettings);
  renderRiskSnapshot(state.riskSnapshot);
  renderPositions();
}

async function loadRiskState() {
  try {
    const snapshot = await api('/risk/state');
    state.riskSnapshot = snapshot || {};
    if (state.riskSnapshot.account_currency) {
      state.accountCurrency = state.riskSnapshot.account_currency;
    }
    renderRiskSnapshot(state.riskSnapshot);
  renderPositions();
    renderPositions();
  } catch (error) {
    pushAlert(`Failed to refresh risk snapshot: ${error.message}`, 'error');
  }
}

function populateRiskControls(config) {
  elements.riskInputs.forEach((input) => {
    const key = input.dataset.riskKey;
    const value = config?.[key];
    if (input.type === 'checkbox') {
      input.checked = Boolean(value);
    } else if (value === null || value === undefined) {
      input.value = '';
    } else {
      input.value = value;
    }
  });
  clearSessionDirty();
}

function collectRiskOverrides() {
  const overrides = {};
  elements.riskInputs.forEach((input) => {
    const key = input.dataset.riskKey;
    if (input.type === 'checkbox') {
      overrides[key] = input.checked;
      return;
    }
    const raw = input.value.trim();
    if (raw === '') {
      return;
    }
    const num = Number(raw);
    overrides[key] = Number.isFinite(num) ? num : null;
  });
  return overrides;
}

function renderRiskSnapshot(snapshot) {
  if (!snapshot) return;
  if (snapshot.halt_triggered && snapshot.halt_reason && snapshot.halt_reason !== state.lastHaltReason) {
    state.lastHaltReason = snapshot.halt_reason;
    pushAlert(`Risk halt: ${snapshot.halt_reason}`, 'danger');
  } else if (!snapshot.halt_triggered) {
    state.lastHaltReason = null;
  }
  const currency = snapshot.account_currency || state.accountCurrency || 'USD';
  const totalExposure = snapshot.total_exposure || 0;
  const capacity = snapshot.exposure_capacity || 0;
  const available = snapshot.available_capacity;

  elements.totalExposureValue.textContent = `${fmtCurrency(totalExposure, 2, currency)}${capacity ? ` / ${fmtCurrency(capacity, 2, currency)}` : ''}`;
  elements.exposureCapacityLabel.textContent = capacity ? `Available capacity: ${fmtCurrency(available, 2, currency)}` : 'Capacity: —';

  const percent = capacity ? Math.min(100, (totalExposure / capacity) * 100) : 0;
  elements.totalExposureBar.style.width = `${percent}%`;

  const perSymbol = snapshot.per_symbol_exposure || {};
  const entries = Object.entries(perSymbol)
    .map(([symbol, exposure]) => ({ symbol, exposure }))
    .sort((a, b) => b.exposure - a.exposure)
    .slice(0, 8);

  elements.exposureList.innerHTML = '';
  if (!entries.length) {
    const hint = document.createElement('p');
    hint.className = 'section-help';
    hint.textContent = 'No open exposures.';
    elements.exposureList.appendChild(hint);
    return;
  }

  entries.forEach(({ symbol, exposure }) => {
    const row = document.createElement('div');
    row.className = 'exposure-item';

    const header = document.createElement('div');
    header.className = 'exposure-item-header';
    header.innerHTML = `<span>${symbol}</span><span>${fmtCurrency(exposure, 2, currency)}</span>`;

    const gauge = document.createElement('div');
    gauge.className = 'progress';
    const bar = document.createElement('div');
    bar.className = 'progress-bar';
    const ratio = capacity ? Math.min(100, (exposure / capacity) * 100) : 0;
    bar.style.width = `${ratio}%`;
    gauge.appendChild(bar);

    row.appendChild(header);
    row.appendChild(gauge);
    elements.exposureList.appendChild(row);
  });
}

async function applySession() {
  if (!state.selectedSymbols.length) {
    pushAlert('Select at least one symbol before saving the session.', 'error');
    return;
  }

  const payload = {
    symbols: state.selectedSymbols,
    model_path: state.selectedModelPath || null,
    target_capital_usd: state.targetCapital,
    risk: collectRiskOverrides(),
  };

  try {
    elements.applySessionBtn.disabled = true;
    await api('/session/config', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    clearSessionDirty();
    pushAlert('Session configuration updated.', 'success');
    await Promise.all([
      refreshStatus(),
      loadRiskState(),
      loadAnalytics(),
    ]);
  } catch (error) {
    pushAlert(`Failed to update session: ${error.message}`, 'error');
  } finally {
    elements.applySessionBtn.disabled = false;
  }
}

async function updateRiskSettings() {
  const overrides = collectRiskOverrides();
  if (!Object.keys(overrides).length) {
    pushAlert('No risk parameter changes detected.', 'info');
    return;
  }
  try {
    elements.riskUpdateBtn.disabled = true;
    await api('/risk/update', {
      method: 'POST',
      body: JSON.stringify(overrides),
    });
    pushAlert('Risk parameters updated.', 'success');
    await loadRiskState();
  } catch (error) {
    pushAlert(`Failed to update risk parameters: ${error.message}`, 'error');
  } finally {
    elements.riskUpdateBtn.disabled = false;
  }
}

async function startEngine() {
  try {
    elements.startEngineBtn.disabled = true;
    await api('/session/start', { method: 'POST' });
    pushAlert('Trading engine started.', 'success');
    await refreshStatus();
  } catch (error) {
    pushAlert(`Failed to start engine: ${error.message}`, 'error');
  } finally {
    elements.startEngineBtn.disabled = false;
  }
}

async function stopEngine() {
  try {
    elements.stopEngineBtn.disabled = true;
    await api('/session/stop', { method: 'POST' });
    pushAlert('Trading engine stopped.', 'success');
    await refreshStatus();
  } catch (error) {
    pushAlert(`Failed to stop engine: ${error.message}`, 'error');
  } finally {
    elements.stopEngineBtn.disabled = false;
  }
}

async function emergencyStop() {
  if (!window.confirm('Emergency stop will close all positions and halt trading. Continue?')) {
    return;
  }
  try {
    elements.emergencyStopBtn.disabled = true;
    await api('/session/emergency_stop', { method: 'POST' });
    pushAlert('Emergency stop executed.', 'danger');
    await Promise.all([
      refreshStatus(),
      loadPositions(),
      loadRiskState(),
    ]);
  } catch (error) {
    pushAlert(`Emergency stop failed: ${error.message}`, 'error');
  } finally {
    elements.emergencyStopBtn.disabled = false;
  }
}

async function runSmokeTest() {
  try {
    elements.smokeTestBtn.disabled = true;
    const response = await api('/test/smoke_trade', { method: 'POST' });
    const symbols = response?.symbols || [];
    pushAlert(`Smoke test dispatched for ${symbols.length} symbol(s).`, 'success');
  } catch (error) {
    pushAlert(`Smoke test failed: ${error.message}`, 'error');
  } finally {
    elements.smokeTestBtn.disabled = false;
  }
}

async function refreshStatus() {
  try {
    const status = await api('/session/status');
    state.engineRunning = Boolean(status?.running);
    state.warmup = status?.warmup || state.warmup;
    state.suspended = status?.suspended || {};
    if (!state.sessionDirty) {
      state.selectedSymbols = (status?.selected_symbols || []).map((s) => s.toUpperCase());
      renderSymbolCatalog();
      adjustSymbolChartHeight(state.selectedSymbols.length);
      state.selectedModelPath = status?.model || '';
      renderModelSelect();
      const target = status?.target_capital;
      if (target !== undefined) {
        state.targetCapital = target;
        elements.targetCapitalInput.value = target ?? '';
      }
    }
    if (status?.risk_snapshot) {
      state.riskSnapshot = status.risk_snapshot;
      renderRiskSnapshot(state.riskSnapshot);
  renderPositions();
    renderPositions();
    }
    if (status?.hard_stop_reason) {
      pushAlert(`Trading halted: ${status.hard_stop_reason}`, 'danger');
    }
    updateEngineStatus();
    updateWarmupBadge();
  } catch (error) {
    pushAlert(`Failed to refresh status: ${error.message}`, 'error');
  }
}

async function loadAccountSummary() {
  try {
    const data = await api('/account/summary');
    state.accountSummary = data?.account || {};
    if (state.accountSummary.currency) {
      state.accountCurrency = state.accountSummary.currency;
    }
    renderAccountSummary();
  } catch (error) {
    pushAlert(`Failed to load account summary: ${error.message}`, 'error');
  }
}

function renderAccountSummary() {
  const acct = state.accountSummary || {};
  const currency = state.accountCurrency || acct.currency || 'USD';
  elements.accountEquity.textContent = fmtCurrency(acct.NAV ?? acct.equity ?? acct.balance, 2, currency);
  elements.accountBalance.textContent = fmtCurrency(acct.balance, 2, currency);
  elements.accountUnrealized.textContent = fmtCurrency(acct.unrealizedPL, 2, currency);
  elements.accountMarginAvailable.textContent = fmtCurrency(acct.marginAvailable, 2, currency);
  elements.accountMarginUsed.textContent = fmtCurrency(acct.marginUsed, 2, currency);
  elements.accountCurrency.textContent = currency;
}

async function loadPositions() {
  try {
    const data = await api('/account/positions');
    state.positions = Array.isArray(data) ? data : [];
    renderPositions();
  } catch (error) {
    pushAlert(`Failed to load positions: ${error.message}`, 'error');
  }
}

function renderPositions() {
  const tbody = elements.positionsTableBody;
  tbody.innerHTML = '';
  const snapshot = state.riskSnapshot || {};
  const perSymbolExposure = snapshot.per_symbol_exposure || {};
  const currency = snapshot.account_currency || state.accountCurrency || 'USD';

  state.positions.forEach((position) => {
    const row = document.createElement('tr');
    const exposure = perSymbolExposure[position.instrument] || 0;
    row.innerHTML = `
      <td>${position.instrument}</td>
      <td>${position.side}</td>
      <td>${fmtNumber(position.units, 0)}</td>
      <td>${fmtNumber(position.avgPrice, 5)}</td>
      <td class="${Number(position.unrealizedPL) >= 0 ? 'pnl-positive' : 'pnl-negative'}">${fmtCurrency(position.unrealizedPL, 2, currency)}</td>
      <td>${fmtCurrency(exposure, 2, currency)}</td>
      <td><button class="btn subtle" data-close="${position.instrument}">Close</button></td>
    `;
    tbody.appendChild(row);
  });

  tbody.querySelectorAll('button[data-close]').forEach((btn) => {
    btn.addEventListener('click', async (event) => {
      const instrument = event.currentTarget.dataset.close;
      try {
        event.currentTarget.disabled = true;
        await api(`/session/close_position?instrument=${encodeURIComponent(instrument)}`, { method: 'POST' });
        pushAlert(`Close order submitted for ${instrument}.`, 'success');
        await Promise.all([loadPositions(), loadRiskState(), loadAnalytics()]);
      } catch (error) {
        pushAlert(`Failed to close ${instrument}: ${error.message}`, 'error');
      } finally {
        event.currentTarget.disabled = false;
      }
    });
  });

  elements.positionsCountBadge.textContent = state.positions.length;
}

async function loadTrades() {
  try {
    const data = await api('/account/trades?limit=200');
    state.trades = Array.isArray(data) ? data : [];
    renderTrades();
  } catch (error) {
    pushAlert(`Failed to load trades: ${error.message}`, 'error');
  }
}

function renderTrades() {
  const tbody = elements.tradesTableBody;
  tbody.innerHTML = '';
  const currency = state.accountCurrency || 'USD';
  state.trades.slice(0, 50).forEach((trade) => {
    const row = document.createElement('tr');
    const pnl = Number(trade.pnl);
    row.innerHTML = `
      <td>${trade.instrument}</td>
      <td>${fmtNumber(trade.units, 0)}</td>
      <td class="${pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">${fmtCurrency(pnl, 2, currency)}</td>
      <td>${formatTimestamp(trade.open_timestamp)}</td>
      <td>${formatTimestamp(trade.close_timestamp)}</td>
      <td>${trade.status}</td>
    `;
    tbody.appendChild(row);
  });
  elements.tradesCountBadge.textContent = state.trades.length;
}

async function loadAnalytics() {
  try {
    const query = state.selectedSymbols.length ? `?limit=500&symbols=${encodeURIComponent(state.selectedSymbols.join(','))}` : '?limit=500';
    state.analytics = await api(`/analytics/performance${query}`);
    renderPerformanceMetrics(state.analytics?.summary);
    updateChartsFromAnalytics(state.analytics);
  } catch (error) {
    pushAlert(`Failed to load analytics: ${error.message}`, 'error');
  }
}

function renderPerformanceMetrics(summary) {
  if (!summary) {
    elements.perfTotalPnl.textContent = '—';
    elements.perfWinRate.textContent = '—';
    elements.perfDrawdown.textContent = '—';
    elements.perfTradeCount.textContent = '—';
    return;
  }
  elements.perfTotalPnl.textContent = fmtCurrency(summary.total_pnl, 2, state.accountCurrency);
  elements.perfWinRate.textContent = fmtPercent(summary.win_rate, 1);
  elements.perfDrawdown.textContent = fmtCurrency(-Math.abs(summary.max_drawdown || 0), 2, state.accountCurrency);
  elements.perfTradeCount.textContent = fmtNumber(summary.trade_count, 0);
}

function initCharts() {
  const equityCtx = document.getElementById('equityChartCanvas');
  const symbolCtx = document.getElementById('symbolChartCanvas');

  if (elements.equityChartShell && !elements.equityChartShell.style.height) {
    elements.equityChartShell.style.height = '240px';
  }
  if (elements.symbolChartShell && !elements.symbolChartShell.style.height) {
    elements.symbolChartShell.style.height = '260px';
  }

  state.charts.equity = new Chart(equityCtx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Equity', data: [], borderColor: '#60a5fa', backgroundColor: 'rgba(96, 165, 250, 0.1)', tension: 0.25, fill: true }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: 'rgba(226, 232, 240, 0.75)' } },
        y: { ticks: { color: 'rgba(226, 232, 240, 0.75)' } },
      },
      plugins: {
        legend: { labels: { color: 'rgba(226, 232, 240, 0.8)' } },
      },
    },
  });

  state.charts.symbol = new Chart(symbolCtx, {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'PnL', data: [], backgroundColor: '#fbbf24' }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: 'rgba(226, 232, 240, 0.75)' } },
        y: { ticks: { color: 'rgba(226, 232, 240, 0.75)' } },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });

  adjustSymbolChartHeight(state.selectedSymbols.length);
}

function updateChartsFromAnalytics(analytics) {
  if (!analytics) {
    updateEquityChart([]);
    updateSymbolChart({});
    return;
  }
  updateEquityChart(analytics.equity_curve || []);
  updateSymbolChart(analytics.symbol_breakdown || {});
}

function updateEquityChart(curve) {
  const chart = state.charts.equity;
  if (!chart) return;
  const labels = curve.map((point) => formatTimestamp(point.time));
  const data = curve.map((point) => Number(point.value));
  chart.data.labels = labels;
  chart.data.datasets[0].data = data;
  chart.update('none');
}

function updateSymbolChart(breakdown) {
  const chart = state.charts.symbol;
  if (!chart) return;
  const entries = Object.entries(breakdown || {});
  const labels = entries.map(([symbol]) => symbol);
  chart.data.labels = labels;
  chart.data.datasets[0].data = entries.map(([, info]) => Number(info.total_pnl || 0));
  adjustSymbolChartHeight(labels.length);
  chart.update('none');
}

function formatTimestamp(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function updateEngineStatus() {
  const indicator = elements.engineStatusIndicator;
  indicator.classList.toggle('running', state.engineRunning);
  elements.engineStatusLabel.textContent = state.engineRunning ? 'Running' : 'Stopped';
}

function updateWarmupBadge() {
  if (!state.warmup) return;
  if (state.warmup.ready) {
    elements.warmupBadge.textContent = 'Warmup complete';
    elements.warmupBadge.classList.remove('danger');
    elements.warmupBadge.classList.add('success');
  } else {
    const missing = (state.warmup.missing || []).slice(0, 4).join(', ');
    elements.warmupBadge.textContent = missing ? `Warming: ${missing}` : 'Warming…';
    elements.warmupBadge.classList.add('danger');
    elements.warmupBadge.classList.remove('success');
  }
}

function updateHeartbeatDisplay() {
  if (!state.lastHeartbeat) {
    elements.heartbeatBadge.textContent = 'Heartbeat —';
    return;
  }
  const diffMs = Date.now() - state.lastHeartbeat.getTime();
  const seconds = Math.max(0, Math.round(diffMs / 1000));
  elements.heartbeatBadge.textContent = `Heartbeat ${seconds}s ago`;
}

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
  state.ws = socket;

  socket.addEventListener('open', () => {
    pushAlert('Realtime feed connected.', 'success');
  });

  socket.addEventListener('message', (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === 'status') {
        handleWsStatus(payload.data);
      } else if (payload.type === 'account') {
        handleWsAccount(payload.data);
      }
      state.lastHeartbeat = new Date();
      updateHeartbeatDisplay();
    } catch (error) {
      console.error('WS parse error', error);
    }
  });

  socket.addEventListener('close', () => {
    pushAlert('Realtime feed disconnected. Retrying…', 'warning');
    setTimeout(connectWebSocket, 3000);
  });
}

function handleWsStatus(data) {
  if (!data) return;
  state.engineRunning = Boolean(data.running);
  if (!state.sessionDirty) {
    if (Array.isArray(data.selected_symbols)) {
      state.selectedSymbols = data.selected_symbols.map((s) => s.toUpperCase());
      renderSymbolCatalog();
    }
    if (data.model && data.model !== state.selectedModelPath) {
      state.selectedModelPath = data.model;
      renderModelSelect();
    }
  }
  updateEngineStatus();
}

function handleWsAccount(data) {
  if (!data?.account) return;
  state.accountSummary = data.account;
  if (state.accountSummary.currency) {
    state.accountCurrency = state.accountSummary.currency;
  }
  renderAccountSummary();
}

function startPolling() {
  const intervals = [
    setInterval(loadAccountSummary, 10000),
    setInterval(loadPositions, 10000),
    setInterval(loadTrades, 15000),
    setInterval(loadRiskState, 12000),
    setInterval(() => updateHeartbeatDisplay(), 1000),
  ];
  state.pollingHandles = intervals;
}

async function init() {
  cacheElements();
  bindEvents();
  initCharts();

  try {
    updateHeartbeatDisplay();
    await Promise.all([
      loadSymbolCatalog(),
      loadModels(),
      loadRiskConfig(),
      refreshStatus(),
      loadAccountSummary(),
      loadPositions(),
      loadTrades(),
      loadAnalytics(),
    ]);
  } catch (error) {
    pushAlert(`Initialisation error: ${error.message}`, 'error');
  }

  startPolling();
  connectWebSocket();
}

document.addEventListener('DOMContentLoaded', init);






