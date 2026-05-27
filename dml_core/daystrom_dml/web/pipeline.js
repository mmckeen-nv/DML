const $ = (selector) => document.querySelector(selector);

const els = {
  status: $('#pipeline-status'),
  apiKey: $('#api-key-state'),
  message: $('#pipeline-message'),
  prepare: $('#prepare-pipeline'),
  run: $('#run-paid-inference'),
  prompt: $('#pipeline-prompt'),
  model: $('#pipeline-model'),
  effort: $('#pipeline-effort'),
  topK: $('#pipeline-top-k'),
  frontierMax: $('#pipeline-frontier-max'),
  directInput: $('#pipeline-direct-input'),
  directOutput: $('#pipeline-direct-output'),
  localDraft: $('#pipeline-local-draft'),
  mode: $('#metric-mode'),
  frontierInput: $('#metric-frontier-input'),
  inputSaved: $('#metric-input-saved'),
  outputSaved: $('#metric-output-saved'),
  latency: $('#pipeline-latency'),
  barDirectInput: $('#bar-direct-input'),
  barDirectOutput: $('#bar-direct-output'),
  barFrontierInput: $('#bar-frontier-input'),
  barFrontierOutput: $('#bar-frontier-output'),
  directTokenLabel: $('#direct-token-label'),
  frontierTokenLabel: $('#frontier-token-label'),
  countDmlContext: $('#count-dml-context'),
  countLocalDraft: $('#count-local-draft'),
  countRetrieved: $('#count-retrieved'),
  countLedger: $('#count-ledger'),
  artifactStatus: $('#artifact-status'),
  dmlContext: $('#pipeline-dml-context'),
  localOutput: $('#pipeline-local-output'),
  frontierPrompt: $('#pipeline-frontier-prompt'),
  frontierResponse: $('#pipeline-frontier-response'),
  dmlCount: $('#dml-pipeline-count'),
  draftCount: $('#local-draft-count'),
  promptCount: $('#frontier-prompt-count'),
  responseCount: $('#frontier-response-count'),
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
    session_id: 'daystrom-inference-pipeline',
    model: els.model.value,
    reasoning_effort: els.effort.value,
    top_k: Number(els.topK.value || 8),
    frontier_max_tokens: Number(els.frontierMax.value || 512),
    direct_input_tokens_estimate: Number(els.directInput.value || 0),
    direct_output_tokens_estimate: Number(els.directOutput.value || 0),
    include_local_draft: els.localDraft.checked,
  };
}

function renderBars(telemetry = {}) {
  const directInput = Number(telemetry.direct_input_tokens_estimate || 0);
  const directOutput = Number(telemetry.direct_output_tokens_estimate || 0);
  const frontierInput = Number(telemetry.frontier_input_tokens || 0);
  const frontierOutput = Number(telemetry.frontier_output_tokens_estimate || 0);
  const maxTotal = Math.max(1, directInput + directOutput, frontierInput + frontierOutput);
  const setWidth = (el, value) => {
    el.style.width = `${Math.max(1, (Number(value || 0) / maxTotal) * 100)}%`;
  };
  setWidth(els.barDirectInput, directInput);
  setWidth(els.barDirectOutput, directOutput);
  setWidth(els.barFrontierInput, frontierInput);
  setWidth(els.barFrontierOutput, frontierOutput);
  els.directTokenLabel.textContent = formatNumber(directInput + directOutput);
  els.frontierTokenLabel.textContent = formatNumber(frontierInput + frontierOutput);
}

function renderPrepared(data) {
  const telemetry = data.telemetry || {};
  setStatus(els.status, data.mode || 'prepared', 'good');
  setStatus(els.apiKey, data.api_key_configured ? 'key configured' : 'key missing', data.api_key_configured ? 'good' : 'warn');
  setStatus(els.artifactStatus, 'prepared', 'good');
  els.mode.textContent = data.mode || '-';
  els.frontierInput.textContent = formatNumber(telemetry.frontier_input_tokens);
  els.inputSaved.textContent = `${formatNumber(telemetry.input_tokens_saved_estimate)} (${telemetry.input_savings_pct_estimate || 0}%)`;
  els.outputSaved.textContent = `${formatNumber(telemetry.output_tokens_saved_estimate)} (${telemetry.output_savings_pct_estimate || 0}%)`;
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
  renderBars(telemetry);
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

els.prepare.addEventListener('click', prepareOnly);
els.run.addEventListener('click', runPaidInference);
prepareOnly();
