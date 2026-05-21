const $ = (selector) => document.querySelector(selector);

const state = {
  health: null,
  stats: null,
  knowledge: null,
};

const elements = {
  serviceStatus: $('#service-status'),
  refreshStatus: $('#refresh-status'),
  memories: $('#metric-memories'),
  dmlTokens: $('#metric-dml-tokens'),
  fidelity: $('#metric-fidelity'),
  visualizerMetric: $('#metric-visualizer'),
  prompt: $('#prompt'),
  topK: $('#top-k'),
  maxTokens: $('#max-tokens'),
  runQuery: $('#run-query'),
  queryStatus: $('#query-status'),
  queryMode: $('#query-mode'),
  contextCount: $('#context-count'),
  dmlContext: $('#dml-context'),
  response: $('#dml-response'),
  responseMode: $('#response-mode'),
  runLatency: $('#run-latency'),
  signalPromptTokens: $('#signal-prompt-tokens'),
  signalContextTokens: $('#signal-context-tokens'),
  signalRagTokens: $('#signal-rag-tokens'),
  signalFidelity: $('#signal-fidelity'),
  ragBackends: $('#rag-backends'),
  memoryText: $('#memory-text'),
  memorySource: $('#memory-source'),
  ingestMemory: $('#ingest-memory'),
  uploadForm: $('#upload-form'),
  fileInput: $('#file-input'),
  selectedFiles: $('#selected-files'),
  ingestStatus: $('#ingest-status'),
  knowledgeStatus: $('#knowledge-status'),
  knowledgeList: $('#knowledge-list'),
  launchVisualizer: $('#launch-visualizer'),
  openVisualizer: $('#open-visualizer'),
  visualizerStatus: $('#visualizer-status'),
  visualizerFrame: $('#visualizer-frame'),
  visualizerPlaceholder: $('#visualizer-placeholder'),
};

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '-';
  }
  return new Intl.NumberFormat().format(Number(value));
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '-';
  }
  const normalized = Number(value) <= 1 ? Number(value) * 100 : Number(value);
  return `${normalized.toFixed(1)}%`;
}

function setStatus(element, message, tone = 'neutral') {
  if (!element) return;
  element.textContent = message;
  element.dataset.tone = tone;
}

function setBusy(button, busy, label) {
  if (!button) return;
  button.disabled = busy;
  if (label) button.textContent = label;
}

async function requestJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: options.body instanceof FormData ? undefined : { 'Content-Type': 'application/json' },
    ...options,
  });
  let payload = null;
  const text = await response.text();
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text };
    }
  }
  if (!response.ok) {
    const detail = payload?.detail || payload?.error || response.statusText;
    throw new Error(detail);
  }
  return payload || {};
}

function dmlEntries(knowledge = state.knowledge) {
  return knowledge?.dml?.entries || [];
}

function averageFidelity(entries = dmlEntries()) {
  const values = entries
    .map((entry) => Number(entry.fidelity))
    .filter((value) => !Number.isNaN(value));
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function extractStatsMemories(stats) {
  return stats?.memories ?? stats?.memory_count ?? stats?.store?.memories ?? null;
}

function renderMetrics() {
  const adapter = state.health?.components?.adapter || {};
  const visualizer = state.health?.components?.visualizer || {};
  const knowledge = state.knowledge || {};
  const dml = knowledge.dml || {};
  const entries = dmlEntries(knowledge);
  const memories = adapter.memories ?? dml.count ?? extractStatsMemories(state.stats);

  elements.memories.textContent = formatNumber(memories);
  elements.dmlTokens.textContent = formatNumber(dml.total_tokens);
  elements.fidelity.textContent = formatPercent(averageFidelity(entries));
  elements.visualizerMetric.textContent = visualizer.status || '-';

  const status = state.health?.status || 'unknown';
  setStatus(elements.serviceStatus, status, status === 'ok' ? 'good' : 'warn');
}

function renderKnowledge() {
  const entries = dmlEntries();
  const dml = state.knowledge?.dml || {};
  const count = dml.count ?? entries.length;
  const limit = dml.display_limit ?? entries.length;
  const statusText = count
    ? `${formatNumber(entries.length)} shown / ${formatNumber(count)} memories`
    : 'empty';
  setStatus(elements.knowledgeStatus, statusText, count ? 'good' : 'neutral');

  if (!entries.length) {
    elements.knowledgeList.innerHTML = '<p class="empty-copy">No DML memories are available yet.</p>';
    return;
  }

  elements.knowledgeList.innerHTML = entries.slice(0, limit).map((entry) => {
    const source = entry.meta?.source || entry.meta?.doc_path || entry.meta?.filename || 'memory';
    const summary = escapeHTML(entry.summary || entry.text || '');
    return `
      <article class="knowledge-row">
        <div>
          <strong>${escapeHTML(source)}</strong>
          <p>${summary}</p>
        </div>
        <dl>
          <div><dt>Level</dt><dd>${formatNumber(entry.level)}</dd></div>
          <div><dt>Tokens</dt><dd>${formatNumber(entry.tokens)}</dd></div>
          <div><dt>Fidelity</dt><dd>${formatPercent(entry.fidelity)}</dd></div>
        </dl>
      </article>
    `;
  }).join('');
}

function escapeHTML(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

async function refreshStatus() {
  setStatus(elements.serviceStatus, 'checking', 'neutral');
  try {
    const [health, stats, knowledge] = await Promise.all([
      requestJSON('/health'),
      requestJSON('/stats').catch(() => ({})),
      requestJSON('/knowledge').catch(() => ({})),
    ]);
    state.health = health;
    state.stats = stats;
    state.knowledge = knowledge;
    renderMetrics();
    renderKnowledge();
  } catch (error) {
    setStatus(elements.serviceStatus, 'degraded', 'bad');
    setStatus(elements.knowledgeStatus, 'unavailable', 'bad');
    elements.knowledgeList.innerHTML = `<p class="empty-copy">${escapeHTML(error.message)}</p>`;
  }
}

async function ingestMemory() {
  const text = elements.memoryText.value.trim();
  if (!text) {
    setStatus(elements.ingestStatus, 'Add memory text first.', 'warn');
    return;
  }

  setBusy(elements.ingestMemory, true, 'Ingesting');
  setStatus(elements.ingestStatus, 'Writing memory...', 'neutral');
  try {
    await requestJSON('/ingest', {
      method: 'POST',
      body: JSON.stringify({
        text,
        meta: {
          source: elements.memorySource.value.trim() || 'playground',
          kind: 'playground_note',
          created_by: 'dml_playground',
        },
      }),
    });
    setStatus(elements.ingestStatus, 'Memory ingested.', 'good');
    await refreshStatus();
  } catch (error) {
    setStatus(elements.ingestStatus, error.message, 'bad');
  } finally {
    setBusy(elements.ingestMemory, false, 'Ingest');
  }
}

function updateSelectedFiles() {
  const files = Array.from(elements.fileInput.files || []);
  elements.selectedFiles.innerHTML = files
    .slice(0, 8)
    .map((file) => `<li>${escapeHTML(file.name)} <span>${formatNumber(file.size)} bytes</span></li>`)
    .join('');
  if (files.length > 8) {
    elements.selectedFiles.insertAdjacentHTML('beforeend', `<li>${files.length - 8} more files</li>`);
  }
}

async function uploadFiles(event) {
  event.preventDefault();
  const files = Array.from(elements.fileInput.files || []);
  if (!files.length) {
    setStatus(elements.ingestStatus, 'Choose files first.', 'warn');
    return;
  }

  const submitButton = elements.uploadForm.querySelector('button[type="submit"]');
  const data = new FormData();
  files.forEach((file) => data.append('files', file));

  setBusy(submitButton, true, 'Uploading');
  setStatus(elements.ingestStatus, `Uploading ${formatNumber(files.length)} file(s)...`, 'neutral');
  try {
    const payload = await requestJSON('/upload', { method: 'POST', body: data });
    const chunks = formatNumber(payload.chunks || 0);
    const status = payload.status === 'partial' ? 'warn' : 'good';
    setStatus(elements.ingestStatus, `Ingested ${chunks} chunks from ${formatNumber(payload.files_ingested || files.length)} file(s).`, status);
    await refreshStatus();
  } catch (error) {
    setStatus(elements.ingestStatus, error.message, 'bad');
  } finally {
    setBusy(submitButton, false, 'Upload Files');
  }
}

function contextFromCompare(payload) {
  return payload?.dml?.context || payload?.dml_context || payload?.context || '';
}

function responseFromCompare(payload) {
  return payload?.dml?.response || payload?.dml_response || payload?.response || payload?.answer || '';
}

function entriesFromCompare(payload) {
  const entries = payload?.dml?.entries || payload?.dml?.items || payload?.dml?.summaries || [];
  return Array.isArray(entries) ? entries : [];
}

function renderBackends(payload) {
  const backends = payload?.rag_backends || payload?.rag?.backends || payload?.rag || [];
  const list = Array.isArray(backends) ? backends : Object.values(backends || {});
  if (!list.length) {
    elements.ragBackends.innerHTML = '<p class="empty-copy">No RAG backend comparison data for this run.</p>';
    return;
  }

  elements.ragBackends.innerHTML = list.map((backend) => {
    const name = backend.name || backend.backend || backend.label || 'backend';
    const tokens = backend.tokens || backend.context_tokens || backend.total_tokens || 0;
    const docs = backend.documents?.length || backend.docs?.length || backend.count || 0;
    return `
      <article class="backend-row">
        <strong>${escapeHTML(name)}</strong>
        <span>${formatNumber(tokens)} tokens</span>
        <span>${formatNumber(docs)} docs</span>
      </article>
    `;
  }).join('');
}

function renderRun(payload, fallbackMode = 'compare') {
  const context = contextFromCompare(payload);
  const response = responseFromCompare(payload);
  const entries = entriesFromCompare(payload);
  const stats = payload?.stats || {};
  const dml = payload?.dml || {};
  const ragTokens = payload?.rag_context_tokens || payload?.rag?.tokens || payload?.rag?.context_tokens;
  const contextTokens = dml.context_tokens || payload?.context_tokens || payload?.tokens;
  const latency = dml.generation_latency_ms || dml.latency_ms || payload?.latency_ms;

  elements.dmlContext.textContent = context || 'No DML context was returned for this prompt.';
  elements.dmlContext.classList.toggle('empty', !context);
  elements.response.textContent = response || 'No generated response returned. The context panel still shows retrieved memory.';
  elements.response.classList.toggle('empty', !response);
  elements.contextCount.textContent = `${formatNumber(entries.length || dml.entry_count || dml.entries || 0)} nodes`;
  elements.responseMode.textContent = payload?.mode || fallbackMode;
  elements.queryMode.textContent = payload?.mode || fallbackMode;
  elements.runLatency.textContent = latency ? `${formatNumber(latency)} ms` : '-';
  elements.signalPromptTokens.textContent = formatNumber(payload?.prompt_tokens_est);
  elements.signalContextTokens.textContent = formatNumber(contextTokens);
  elements.signalRagTokens.textContent = formatNumber(ragTokens);
  elements.signalFidelity.textContent = formatPercent(dml.average_fidelity || dml.avg_fidelity || averageFidelity(entries));
  renderBackends(payload);

  if (stats.memories || stats.memory_count) {
    state.stats = { ...state.stats, ...stats };
    renderMetrics();
  }
}

async function runQuery() {
  const prompt = elements.prompt.value.trim();
  if (!prompt) {
    setStatus(elements.queryStatus, 'Add a prompt first.', 'warn');
    return;
  }

  setBusy(elements.runQuery, true, 'Running');
  setStatus(elements.queryStatus, 'Running DML comparison...', 'neutral');
  const topK = Number(elements.topK.value || 0);
  const maxTokens = Number(elements.maxTokens.value || 512);
  const started = performance.now();

  try {
    const payload = await requestJSON('/rag/compare', {
      method: 'POST',
      body: JSON.stringify({ prompt, top_k: topK, max_new_tokens: maxTokens }),
    });
    renderRun(payload, 'compare');
    setStatus(elements.queryStatus, 'Comparison complete.', 'good');
  } catch (compareError) {
    try {
      setStatus(elements.queryStatus, 'Generator unavailable; using DML retrieval path.', 'warn');
      const payload = await requestJSON('/query', {
        method: 'POST',
        body: JSON.stringify({ prompt }),
      });
      renderRun({ ...payload, latency_ms: Math.round(performance.now() - started) }, 'query');
      setStatus(elements.queryStatus, 'DML query complete.', 'good');
    } catch (queryError) {
      setStatus(elements.queryStatus, queryError.message || compareError.message, 'bad');
    }
  } finally {
    setBusy(elements.runQuery, false, 'Run');
    refreshStatus();
  }
}

async function launchVisualizer() {
  setBusy(elements.launchVisualizer, true, 'Launching');
  setStatus(elements.visualizerStatus, 'starting', 'neutral');
  try {
    const payload = await requestJSON('/visualizer/launch', { method: 'POST' });
    const target = payload.embed_url || payload.url || '/visualizer';
    elements.visualizerFrame.src = target;
    elements.openVisualizer.href = payload.url || '/visualizer';
    elements.visualizerPlaceholder.hidden = true;
    setStatus(elements.visualizerStatus, payload.status || 'ready', 'good');
    await refreshStatus();
  } catch (error) {
    setStatus(elements.visualizerStatus, error.message, 'bad');
  } finally {
    setBusy(elements.launchVisualizer, false, 'Launch');
  }
}

elements.refreshStatus?.addEventListener('click', refreshStatus);
elements.ingestMemory?.addEventListener('click', ingestMemory);
elements.fileInput?.addEventListener('change', updateSelectedFiles);
elements.uploadForm?.addEventListener('submit', uploadFiles);
elements.runQuery?.addEventListener('click', runQuery);
elements.launchVisualizer?.addEventListener('click', launchVisualizer);

refreshStatus();
