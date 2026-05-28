const $ = (selector) => document.querySelector(selector);

const els = {
  status: $('#pipeline-status'),
  apiKey: $('#api-key-state'),
  message: $('#pipeline-message'),
  loadFlappy: $('#load-flappy-demo'),
  prepare: $('#prepare-pipeline'),
  run: $('#run-paid-inference'),
  runDirect: $('#run-direct-inference'),
  prompt: $('#pipeline-prompt'),
  model: $('#pipeline-model'),
  effort: $('#pipeline-effort'),
  topK: $('#pipeline-top-k'),
  frontierMax: $('#pipeline-frontier-max'),
  directInput: $('#pipeline-direct-input'),
  directOutput: $('#pipeline-direct-output'),
  inputRate: $('#pipeline-input-rate'),
  outputRate: $('#pipeline-output-rate'),
  localDraft: $('#pipeline-local-draft'),
  mode: $('#metric-mode'),
  turns: $('#metric-turns'),
  frontierInput: $('#metric-frontier-input'),
  inputSaved: $('#metric-input-saved'),
  outputSaved: $('#metric-output-saved'),
  latency: $('#pipeline-latency'),
  tableDirectInput: $('#table-direct-input'),
  tableDirectOutput: $('#table-direct-output'),
  tableDirectTotal: $('#table-direct-total'),
  tableDmlInput: $('#table-dml-input'),
  tableDmlOutput: $('#table-dml-output'),
  tableDmlTotal: $('#table-dml-total'),
  costDirectInput: $('#cost-direct-input'),
  costDirectOutput: $('#cost-direct-output'),
  costDirectTotal: $('#cost-direct-total'),
  costDmlInput: $('#cost-dml-input'),
  costDmlOutput: $('#cost-dml-output'),
  costDmlTotal: $('#cost-dml-total'),
  costSavedInput: $('#cost-saved-input'),
  costSavedOutput: $('#cost-saved-output'),
  costSavedTotal: $('#cost-saved-total'),
  costRateNote: $('#cost-rate-note'),
  countDmlContext: $('#count-dml-context'),
  countLocalDraft: $('#count-local-draft'),
  countRetrieved: $('#count-retrieved'),
  countLedger: $('#count-ledger'),
  artifactStatus: $('#artifact-status'),
  dmlContext: $('#pipeline-dml-context'),
  localOutput: $('#pipeline-local-output'),
  frontierPrompt: $('#pipeline-frontier-prompt'),
  frontierResponse: $('#pipeline-frontier-response'),
  directPrompt: $('#pipeline-direct-prompt'),
  directResponse: $('#pipeline-direct-response'),
  dmlCount: $('#dml-pipeline-count'),
  draftCount: $('#local-draft-count'),
  promptCount: $('#frontier-prompt-count'),
  responseCount: $('#frontier-response-count'),
  directPromptCount: $('#direct-prompt-count'),
  directResponseCount: $('#direct-response-count'),
};

const state = {
  scenario: null,
  lastTelemetry: null,
  dmlUsage: null,
  directUsage: null,
};

function formatNumber(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '-';
  return new Intl.NumberFormat().format(Math.round(numeric));
}

function formatMs(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '-';
  if (numeric >= 1000) return `${(numeric / 1000).toFixed(2)} s`;
  return `${Math.round(numeric)} ms`;
}

function formatPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '0%';
  return `${Math.round(numeric)}%`;
}

function formatCurrency(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '-';
  if (numeric < 0.01) return `$${numeric.toFixed(4)}`;
  return `$${numeric.toFixed(2)}`;
}

function estimateTokens(text) {
  const value = String(text || '').trim();
  return value ? Math.max(1, Math.ceil(value.length / 4)) : 0;
}

function setStatus(el, text, tone = 'neutral') {
  if (!el) return;
  el.textContent = text;
  el.dataset.tone = tone;
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  if (label) button.textContent = label;
}

function setMetric(el, main, sub = '') {
  if (!el) return;
  const mainEl = document.createElement('span');
  mainEl.className = 'metric-main';
  mainEl.textContent = main;
  if (!sub) {
    el.replaceChildren(mainEl);
    return;
  }
  const subEl = document.createElement('span');
  subEl.className = 'metric-sub';
  subEl.textContent = sub;
  el.replaceChildren(mainEl, subEl);
}

async function requestJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || response.statusText);
  }
  return payload;
}

function payload() {
  return {
    prompt: els.prompt.value,
    tenant_id: 'openclaw',
    session_id: state.scenario?.session_id || 'daystrom-inference-pipeline',
    model: els.model.value,
    reasoning_effort: els.effort.value,
    top_k: Number(els.topK.value || 8),
    frontier_max_tokens: Number(els.frontierMax.value || 512),
    direct_input_tokens_estimate: Number(els.directInput.value || 0),
    direct_output_tokens_estimate: Number(els.directOutput.value || 0),
    include_local_draft: els.localDraft.checked,
  };
}

function costRates() {
  return {
    input: Number(els.inputRate.value || 0),
    output: Number(els.outputRate.value || 0),
  };
}

function tokenCost(inputTokens, outputTokens) {
  const rates = costRates();
  return ((Number(inputTokens || 0) * rates.input) + (Number(outputTokens || 0) * rates.output)) / 1_000_000;
}

function renderTokenTable(telemetry = {}) {
  const directInput = Number(telemetry.direct_input_tokens_estimate || 0);
  const directOutput = Number(telemetry.direct_output_tokens_estimate || 0);
  const frontierInput = Number(telemetry.frontier_input_tokens || 0);
  const frontierOutput = Number(telemetry.frontier_output_tokens_estimate || 0);
  els.tableDirectInput.textContent = formatNumber(directInput);
  els.tableDirectOutput.textContent = formatNumber(directOutput);
  els.tableDirectTotal.textContent = formatNumber(directInput + directOutput);
  els.tableDmlInput.textContent = formatNumber(frontierInput);
  els.tableDmlOutput.textContent = formatNumber(frontierOutput);
  els.tableDmlTotal.textContent = formatNumber(frontierInput + frontierOutput);
}

function renderCostTable(telemetry = {}) {
  const directInput = Number(state.directUsage?.input_tokens || telemetry.direct_input_tokens_estimate || 0);
  const directOutput = Number(state.directUsage?.output_tokens || telemetry.direct_output_tokens_estimate || 0);
  const dmlInput = Number(state.dmlUsage?.input_tokens || telemetry.frontier_input_tokens || 0);
  const dmlOutput = Number(state.dmlUsage?.output_tokens || telemetry.frontier_output_tokens_estimate || 0);
  const rates = costRates();

  els.costDirectInput.textContent = formatNumber(directInput);
  els.costDirectOutput.textContent = formatNumber(directOutput);
  els.costDirectTotal.textContent = formatCurrency(tokenCost(directInput, directOutput));
  els.costDmlInput.textContent = formatNumber(dmlInput);
  els.costDmlOutput.textContent = formatNumber(dmlOutput);
  els.costDmlTotal.textContent = formatCurrency(tokenCost(dmlInput, dmlOutput));
  els.costSavedInput.textContent = formatNumber(Math.max(0, directInput - dmlInput));
  els.costSavedOutput.textContent = formatNumber(Math.max(0, directOutput - dmlOutput));
  els.costSavedTotal.textContent = formatCurrency(Math.max(0, tokenCost(directInput, directOutput) - tokenCost(dmlInput, dmlOutput)));
  els.costRateNote.textContent = `Rates: ${formatCurrency(rates.input)} input / ${formatCurrency(rates.output)} output per 1M tokens`;
}

function renderPrepared(data) {
  const telemetry = data.telemetry || {};
  state.lastTelemetry = telemetry;
  setStatus(els.status, data.mode || 'prepared', 'good');
  setStatus(els.apiKey, data.api_key_configured ? 'key configured' : 'key missing', data.api_key_configured ? 'good' : 'warn');
  setStatus(els.artifactStatus, 'prepared', 'good');
  setMetric(els.mode, state.scenario ? 'Flappy Bird' : data.mode || '-', data.mode || '');
  setMetric(
    els.turns,
    state.scenario ? `${formatNumber(state.scenario.dml_turns)} / ${formatNumber(state.scenario.traditional_turns)}` : '-',
    state.scenario ? 'compressed continuity vs full transcript' : '',
  );
  setMetric(
    els.frontierInput,
    formatNumber(telemetry.frontier_input_tokens),
    `${formatNumber(telemetry.dml_context_tokens)} context tokens`,
  );
  setMetric(
    els.inputSaved,
    formatPercent(telemetry.input_savings_pct_estimate),
    `${formatNumber(telemetry.input_tokens_saved_estimate)} estimated tokens`,
  );
  setMetric(
    els.outputSaved,
    formatPercent(telemetry.output_savings_pct_estimate),
    `${formatNumber(telemetry.output_tokens_saved_estimate)} budget tokens`,
  );
  els.latency.textContent = formatMs(telemetry.latency_ms);
  els.countDmlContext.textContent = formatNumber(telemetry.dml_context_tokens);
  els.countLocalDraft.textContent = formatNumber(telemetry.local_draft_tokens);
  els.countRetrieved.textContent = formatNumber(telemetry.retrieved_items);
  els.countLedger.textContent = telemetry.survival_ledger_included ? 'yes' : 'no';
  els.dmlContext.textContent = data.dml_context || 'No DML context retrieved.';
  els.localOutput.textContent = data.local_draft || 'No local draft generated.';
  els.frontierPrompt.textContent = data.frontier_prompt || '';
  els.dmlCount.textContent = `${formatNumber(telemetry.dml_context_tokens)} tokens`;
  els.draftCount.textContent = `${formatNumber(telemetry.local_draft_tokens)} tokens`;
  els.promptCount.textContent = `${formatNumber(telemetry.frontier_input_tokens)} tokens`;
  renderTokenTable(telemetry);
  renderCostTable(telemetry);
}

function renderScenario(data) {
  state.scenario = data;
  state.dmlUsage = null;
  state.directUsage = null;
  els.prompt.value = data.prompt || els.prompt.value;
  els.topK.value = data.top_k || els.topK.value;
  els.frontierMax.value = data.frontier_max_tokens || els.frontierMax.value;
  els.directInput.value = data.direct_input_tokens_estimate || 0;
  els.directOutput.value = data.direct_output_tokens_estimate || 0;
  els.localDraft.checked = false;
  els.directPrompt.textContent = data.direct_prompt || '';
  els.directPromptCount.textContent = `${formatNumber(data.direct_input_tokens_estimate || estimateTokens(data.direct_prompt))} tokens`;
  els.directResponse.textContent = 'Run the direct baseline to compare output without DML-assisted context compression.';
  els.directResponseCount.textContent = '0 tokens';
  setMetric(els.mode, 'Flappy Bird', 'canned coding demo');
  setMetric(els.turns, `${formatNumber(data.dml_turns)} / ${formatNumber(data.traditional_turns)}`, 'DML continuity vs baseline');
  els.message.textContent = `Loaded canned Flappy Bird build: ${formatNumber(data.memory_count)} memories and ${formatNumber(data.traditional_turns)} baseline turns.`;
}

async function loadFlappyDemo() {
  setBusy(els.loadFlappy, true, 'Loading');
  setStatus(els.status, 'seeding demo', 'warn');
  els.message.textContent = '';
  try {
    const data = await requestJSON('/inference/scenarios/flappy-bird', { method: 'POST' });
    renderScenario(data);
    await prepareOnly();
  } catch (error) {
    setStatus(els.status, 'error', 'bad');
    els.message.textContent = error.message;
  } finally {
    setBusy(els.loadFlappy, false, 'Reset Canned Demo');
  }
}

async function prepareOnly() {
  setBusy(els.prepare, true, 'Preparing');
  setStatus(els.status, 'preparing', 'warn');
  els.message.textContent = '';
  try {
    const data = await requestJSON('/inference/prepare', {
      method: 'POST',
      body: JSON.stringify(payload()),
    });
    renderPrepared(data);
    els.message.textContent = 'Prepared without calling the paid endpoint.';
  } catch (error) {
    setStatus(els.status, 'error', 'bad');
    els.message.textContent = error.message;
  } finally {
    setBusy(els.prepare, false, 'Prepare Pipeline');
  }
}

async function runPaidInference() {
  setBusy(els.run, true, 'Running');
  setStatus(els.status, 'paid call', 'warn');
  els.message.textContent = '';
  try {
    const data = await requestJSON('/inference/run', {
      method: 'POST',
      body: JSON.stringify(payload()),
    });
    renderPrepared(data.prepared || {});
    const output = data.inference?.output_text || '';
    const usage = data.inference?.raw?.usage || {};
    state.dmlUsage = usage;
    renderCostTable(state.lastTelemetry || {});
    els.frontierResponse.textContent = output || JSON.stringify(data.inference?.raw || {}, null, 2);
    els.responseCount.textContent = `${formatNumber(data.telemetry?.frontier_output_tokens_observed || estimateTokens(output))} tokens`;
    setStatus(els.status, 'complete', 'good');
    els.message.textContent = `Paid endpoint completed in ${formatMs(data.inference?.latency_ms)}.`;
  } catch (error) {
    setStatus(els.status, 'error', 'bad');
    els.message.textContent = error.message;
  } finally {
    setBusy(els.run, false, 'Run Paid Inference');
  }
}

async function runDirectInference() {
  if (!state.scenario?.direct_prompt) {
    els.message.textContent = 'Load the Flappy Bird demo first so the direct baseline prompt is available.';
    return;
  }
  setBusy(els.runDirect, true, 'Running');
  setStatus(els.status, 'direct call', 'warn');
  els.message.textContent = '';
  try {
    const data = await requestJSON('/inference/direct/run', {
      method: 'POST',
      body: JSON.stringify({
        prompt: state.scenario.direct_prompt,
        model: els.model.value,
        reasoning_effort: els.effort.value,
        max_output_tokens: Number(els.directOutput.value || els.frontierMax.value || 512),
      }),
    });
    const output = data.inference?.output_text || '';
    const usage = data.telemetry?.usage || {};
    state.directUsage = usage;
    renderCostTable(state.lastTelemetry || {});
    const outputTokens = usage.output_tokens || data.telemetry?.direct_output_tokens_observed || estimateTokens(output);
    els.directResponse.textContent = output || JSON.stringify(data.inference?.raw || {}, null, 2);
    els.directResponseCount.textContent = `${formatNumber(outputTokens)} tokens`;
    els.message.textContent = `Direct baseline completed in ${formatMs(data.inference?.latency_ms)}.`;
    setStatus(els.status, 'complete', 'good');
  } catch (error) {
    setStatus(els.status, 'error', 'bad');
    els.message.textContent = error.message;
  } finally {
    setBusy(els.runDirect, false, 'Run Direct Baseline');
  }
}

els.loadFlappy.addEventListener('click', loadFlappyDemo);
els.prepare.addEventListener('click', prepareOnly);
els.run.addEventListener('click', runPaidInference);
els.runDirect.addEventListener('click', runDirectInference);
els.inputRate.addEventListener('input', () => renderCostTable(state.lastTelemetry || {}));
els.outputRate.addEventListener('input', () => renderCostTable(state.lastTelemetry || {}));
loadFlappyDemo();
