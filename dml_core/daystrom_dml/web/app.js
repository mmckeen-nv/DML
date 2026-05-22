const $ = (selector) => document.querySelector(selector);

const state = {
  health: null,
  stats: null,
  knowledge: null,
  highlightedNodeIds: new Set(),
};

const elements = {
  serviceStatus: $('#service-status'),
  refreshStatus: $('#refresh-status'),
  memories: $('#metric-memories'),
  dmlTokens: $('#metric-dml-tokens'),
  fidelity: $('#metric-fidelity'),
  visualizerMetric: $('#metric-visualizer'),
  runLatencyMetric: $('#metric-run-latency'),
  runLatencyDetail: $('#metric-run-latency-detail'),
  runDmlTokens: $('#metric-run-dml-tokens'),
  runNodes: $('#metric-run-nodes'),
  runRagTokens: $('#metric-run-rag-tokens'),
  runDocs: $('#metric-run-docs'),
  runTokenDelta: $('#metric-run-token-delta'),
  runTokenRatio: $('#metric-run-token-ratio'),
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
  ragResponse: $('#rag-response'),
  ragMode: $('#rag-mode'),
  ragContext: $('#rag-context'),
  ragContextCount: $('#rag-context-count'),
  baseResponse: $('#base-response'),
  baseMode: $('#base-mode'),
  runLatency: $('#run-latency'),
  signalPromptTokens: $('#signal-prompt-tokens'),
  signalContextTokens: $('#signal-context-tokens'),
  signalRagTokens: $('#signal-rag-tokens'),
  signalFidelity: $('#signal-fidelity'),
  dmlTokenMeter: $('#dml-token-meter'),
  dmlTokenMeterLabel: $('#dml-token-meter-label'),
  ragTokenMeter: $('#rag-token-meter'),
  ragTokenMeterLabel: $('#rag-token-meter-label'),
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
  latticeSvg: $('#lattice-svg'),
  visualizerPlaceholder: $('#visualizer-placeholder'),
  presetButtons: Array.from(document.querySelectorAll('.preset-button')),
  tabButtons: Array.from(document.querySelectorAll('.tab-button')),
  tabPanels: Array.from(document.querySelectorAll('.tab-panel')),
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

function formatMilliseconds(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '-';
  if (numeric >= 1000) return `${(numeric / 1000).toFixed(2)} s`;
  return `${formatNumber(Math.round(numeric))} ms`;
}

function formatSignedNumber(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '-';
  return `${numeric > 0 ? '+' : ''}${formatNumber(Math.round(numeric))}`;
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
  return stats?.memories ?? stats?.memory_count ?? stats?.count ?? stats?.store?.memories ?? null;
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

function entryId(entry) {
  const raw = entry?.id ?? entry?.meta?.memory_id;
  const numeric = Number(raw);
  return Number.isFinite(numeric) ? numeric : null;
}

function squareLatticeEntries(entries = dmlEntries()) {
  return entries.filter((entry) => {
    const meta = entry.meta || {};
    return meta.synthetic_lattice === 'square'
      && Number.isFinite(Number(meta.lattice_row))
      && Number.isFinite(Number(meta.lattice_col));
  });
}

function renderLattice(entries = dmlEntries(), highlightedEntries = []) {
  if (!elements.latticeSvg) return;
  const square = squareLatticeEntries(entries);
  const highlighted = new Set([
    ...state.highlightedNodeIds,
    ...highlightedEntries.map(entryId).filter((id) => id !== null),
  ]);

  if (!square.length) {
    elements.latticeSvg.innerHTML = '';
    elements.visualizerPlaceholder.hidden = false;
    elements.visualizerPlaceholder.textContent = 'No square lattice nodes are available yet.';
    setStatus(elements.visualizerStatus, 'empty', 'warn');
    return;
  }

  const rows = square.map((entry) => Number(entry.meta.lattice_row));
  const cols = square.map((entry) => Number(entry.meta.lattice_col));
  const maxRow = Math.max(...rows);
  const maxCol = Math.max(...cols);
  const cell = 54;
  const pad = 38;
  const width = pad * 2 + Math.max(1, maxCol) * cell;
  const height = pad * 2 + Math.max(1, maxRow) * cell;
  const byId = new Map(square.map((entry) => [entryId(entry), entry]));
  const lines = [];
  const circles = [];

  for (const entry of square) {
    const id = entryId(entry);
    const meta = entry.meta || {};
    const x = pad + Number(meta.lattice_col) * cell;
    const y = pad + Number(meta.lattice_row) * cell;
    for (const neighbor of meta.lattice_neighbors || []) {
      const neighborEntry = byId.get(Number(neighbor));
      if (!neighborEntry || Number(neighbor) < id) continue;
      const neighborMeta = neighborEntry.meta || {};
      const x2 = pad + Number(neighborMeta.lattice_col) * cell;
      const y2 = pad + Number(neighborMeta.lattice_row) * cell;
      const active = highlighted.has(id) && highlighted.has(Number(neighbor));
      lines.push(
        `<line class="${active ? 'active' : ''}" x1="${x}" y1="${y}" x2="${x2}" y2="${y2}"></line>`
      );
    }
  }

  for (const entry of square) {
    const id = entryId(entry);
    const meta = entry.meta || {};
    const x = pad + Number(meta.lattice_col) * cell;
    const y = pad + Number(meta.lattice_row) * cell;
    const active = highlighted.has(id);
    const source = escapeHTML(meta.source || `node ${id}`);
    const label = escapeHTML(meta.summary || entry.summary || entry.text || source);
    circles.push(
      `<g class="lattice-node ${active ? 'active' : ''}" tabindex="0">`
      + `<circle cx="${x}" cy="${y}" r="${active ? 9 : 6}"></circle>`
      + `<title>${source}\n${label}</title>`
      + `</g>`
    );
  }

  elements.latticeSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  elements.latticeSvg.innerHTML = [
    '<g class="lattice-edges">',
    ...lines,
    '</g>',
    '<g class="lattice-nodes">',
    ...circles,
    '</g>',
  ].join('');
  elements.visualizerPlaceholder.hidden = true;
  setStatus(
    elements.visualizerStatus,
    highlighted.size ? `${formatNumber(highlighted.size)} active` : `${formatNumber(square.length)} nodes`,
    'good',
  );
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
    const title = escapeHTML(source);
    return `
      <details class="knowledge-row">
        <summary>
          <span class="knowledge-title">${title}</span>
          <span class="knowledge-meta">
            <span>L${formatNumber(entry.level)}</span>
            <span>${formatNumber(entry.tokens)} tokens</span>
            <span>${formatPercent(entry.fidelity)}</span>
          </span>
        </summary>
        <div class="knowledge-body">
          <p>${summary}</p>
          <dl>
            <div><dt>Level</dt><dd>${formatNumber(entry.level)}</dd></div>
            <div><dt>Tokens</dt><dd>${formatNumber(entry.tokens)}</dd></div>
            <div><dt>Fidelity</dt><dd>${formatPercent(entry.fidelity)}</dd></div>
          </dl>
        </div>
      </details>
    `;
  }).join('');
  renderLattice(entries);
}

function collapseSiblingKnowledgeRows(event) {
  const opened = event.target;
  if (!(opened instanceof HTMLDetailsElement) || !opened.open) return;
  elements.knowledgeList
    ?.querySelectorAll('details.knowledge-row[open]')
    .forEach((row) => {
      if (row !== opened) row.open = false;
    });
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

function primaryRagBackend(payload) {
  const backends = payload?.rag_backends || payload?.rag?.backends || payload?.rag || [];
  const list = Array.isArray(backends) ? backends : Object.values(backends || {});
  return list.find((backend) => backend?.available !== false) || list[0] || null;
}

function ragTokenTotal(payload) {
  const breakdown = payload?.rag_token_breakdown || [];
  if (Array.isArray(breakdown) && breakdown.length) {
    return breakdown.reduce((sum, item) => sum + Number(item.tokens || 0), 0);
  }
  const backend = primaryRagBackend(payload);
  return backend?.context_tokens || backend?.tokens || backend?.total_tokens || payload?.rag_context_tokens || null;
}

function renderBackends(payload) {
  const backends = payload?.rag_backends || payload?.rag?.backends || payload?.rag || [];
  const list = Array.isArray(backends) ? backends : Object.values(backends || {});
  if (!list.length) {
    elements.ragBackends.innerHTML = '<p class="empty-copy">No RAG backend comparison data for this run.</p>';
    return;
  }

  elements.ragBackends.innerHTML = list.map((backend) => {
    const name = backend.label || backend.name || backend.backend || backend.id || 'backend';
    const tokens = backend.tokens || backend.context_tokens || backend.total_tokens || 0;
    const docs = backend.documents?.length || backend.docs?.length || backend.count || 0;
    const grade = backend.grade?.grade ? `grade ${backend.grade.grade}` : backend.available === false ? 'offline' : 'ready';
    return `
      <article class="backend-row">
        <strong>${escapeHTML(name)}</strong>
        <span>${formatNumber(tokens)} tokens</span>
        <span>${formatNumber(docs)} docs</span>
        <span>${escapeHTML(grade)}</span>
      </article>
    `;
  }).join('');
}

function updateTokenMeter(meter, label, value, maxValue) {
  const numeric = Number(value || 0);
  const max = Math.max(1, Number(maxValue || 0));
  if (meter) {
    meter.max = max;
    meter.value = Math.max(0, numeric);
  }
  if (label) {
    label.textContent = `${formatNumber(numeric)} / ${formatNumber(max)}`;
  }
}

function renderRunTelemetry({ latency, retrievalLatency, generationLatency, contextTokens, ragTokens, dmlNodes, ragDocs }) {
  const tokenDelta = Number(ragTokens || 0) - Number(contextTokens || 0);
  const denominator = Math.max(1, Number(ragTokens || 0));
  const tokenRatio = Number(contextTokens || 0) / denominator;
  const tokenRatioText = ragTokens
    ? `${formatPercent(tokenRatio)} of RAG context`
    : 'No RAG baseline';
  const detailParts = [];
  if (retrievalLatency !== null && retrievalLatency !== undefined) {
    detailParts.push(`${formatMilliseconds(retrievalLatency)} retrieval`);
  }
  if (generationLatency !== null && generationLatency !== undefined) {
    detailParts.push(`${formatMilliseconds(generationLatency)} generation`);
  }

  elements.runLatencyMetric.textContent = formatMilliseconds(latency);
  elements.runLatencyDetail.textContent = detailParts.join(' + ') || 'Measured client-side';
  elements.runDmlTokens.textContent = formatNumber(contextTokens);
  elements.runNodes.textContent = `${formatNumber(dmlNodes)} nodes`;
  elements.runRagTokens.textContent = formatNumber(ragTokens);
  elements.runDocs.textContent = `${formatNumber(ragDocs)} docs`;
  elements.runTokenDelta.textContent = formatSignedNumber(tokenDelta);
  elements.runTokenRatio.textContent = tokenRatioText;
  elements.runTokenDelta.dataset.tone = tokenDelta >= 0 ? 'good' : 'warn';

  const meterMax = Math.max(Number(contextTokens || 0), Number(ragTokens || 0), 1);
  updateTokenMeter(elements.dmlTokenMeter, elements.dmlTokenMeterLabel, contextTokens, meterMax);
  updateTokenMeter(elements.ragTokenMeter, elements.ragTokenMeterLabel, ragTokens, meterMax);
}

function renderRun(payload, fallbackMode = 'compare') {
  const context = contextFromCompare(payload);
  const response = responseFromCompare(payload);
  const entries = entriesFromCompare(payload);
  const ragBackend = primaryRagBackend(payload);
  const ragResponse = ragBackend?.response || payload?.rag?.response || '';
  const ragContext = ragBackend?.context || payload?.rag?.context || '';
  const baseResponse = payload?.base?.response || payload?.base_response || '';
  const stats = payload?.stats || {};
  const dml = payload?.dml || {};
  const ragTokens = ragTokenTotal(payload);
  const contextTokens = dml.context_tokens || payload?.context_tokens || payload?.tokens || 0;
  const retrievalLatency = dml.retrieval_latency_ms ?? payload?.retrieval_latency_ms ?? null;
  const generationLatency = dml.generation_latency_ms ?? payload?.generation_latency_ms ?? null;
  const latency = (Number(retrievalLatency || 0) + Number(generationLatency || 0)) || dml.latency_ms || payload?.latency_ms;
  const dmlNodeCount = entries.length || Number(dml.entry_count || 0);
  const ragDocCount = ragBackend?.documents?.length || ragBackend?.docs?.length || 0;

  elements.dmlContext.textContent = context || 'No DML context was returned for this prompt.';
  elements.dmlContext.classList.toggle('empty', !context);
  elements.response.textContent = response || 'No generated response returned. The context panel still shows retrieved memory.';
  elements.response.classList.toggle('empty', !response);
  elements.ragResponse.textContent = ragResponse || 'No RAG backend response returned.';
  elements.ragResponse.classList.toggle('empty', !ragResponse);
  elements.ragMode.textContent = ragBackend?.label || ragBackend?.id || 'rag';
  elements.ragContext.textContent = ragContext || 'No RAG context returned.';
  elements.ragContext.classList.toggle('empty', !ragContext);
  elements.ragContextCount.textContent = `${formatNumber(ragBackend?.documents?.length || ragBackend?.docs?.length || 0)} docs`;
  elements.baseResponse.textContent = baseResponse || 'No base response returned.';
  elements.baseResponse.classList.toggle('empty', !baseResponse);
  elements.baseMode.textContent = payload?.base ? 'base' : 'waiting';
  elements.contextCount.textContent = `${formatNumber(dmlNodeCount)} nodes`;
  elements.responseMode.textContent = payload?.mode || fallbackMode;
  elements.queryMode.textContent = payload?.mode || fallbackMode;
  elements.runLatency.textContent = formatMilliseconds(latency);
  elements.signalPromptTokens.textContent = formatNumber(payload?.prompt_tokens_est);
  elements.signalContextTokens.textContent = formatNumber(contextTokens);
  elements.signalRagTokens.textContent = formatNumber(ragTokens);
  elements.signalFidelity.textContent = formatPercent(dml.average_fidelity || dml.avg_fidelity || averageFidelity(entries));
  renderRunTelemetry({
    latency,
    retrievalLatency,
    generationLatency,
    contextTokens,
    ragTokens,
    dmlNodes: dmlNodeCount,
    ragDocs: ragDocCount,
  });
  renderBackends(payload);
  state.highlightedNodeIds = new Set(entries.map(entryId).filter((id) => id !== null));
  renderLattice(dmlEntries(), entries);

  if (stats.memories || stats.memory_count) {
    state.stats = { ...state.stats, ...stats };
    renderMetrics();
  }
}

function activateTab(name) {
  elements.tabButtons.forEach((button) => {
    button.classList.toggle('active', button.dataset.tab === name);
  });
  elements.tabPanels.forEach((panel) => {
    panel.classList.toggle('active', panel.dataset.panel === name);
  });
}

async function runQuery() {
  const prompt = elements.prompt.value.trim();
  if (!prompt) {
    setStatus(elements.queryStatus, 'Add a prompt first.', 'warn');
    return;
  }

  setBusy(elements.runQuery, true, 'Running');
  setStatus(elements.queryStatus, 'Running DML comparison...', 'neutral');
  activateTab('answer');
  const topK = Number(elements.topK.value || 0);
  const maxTokens = Number(elements.maxTokens.value || 512);
  const started = performance.now();

  try {
    const payload = await requestJSON('/rag/compare', {
      method: 'POST',
      body: JSON.stringify({ prompt, top_k: topK, max_new_tokens: maxTokens }),
    });
    renderRun({ ...payload, latency_ms: Math.round(performance.now() - started) }, 'compare');
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
  setBusy(elements.launchVisualizer, true, 'Refreshing');
  setStatus(elements.visualizerStatus, 'refreshing', 'neutral');
  try {
    await refreshStatus();
    setStatus(elements.visualizerStatus, 'ready', 'good');
  } catch (error) {
    setStatus(elements.visualizerStatus, error.message, 'bad');
  } finally {
    setBusy(elements.launchVisualizer, false, 'Refresh Lattice');
  }
}

function applyPromptPreset(event) {
  const prompt = event.currentTarget?.dataset?.prompt;
  if (!prompt) return;
  elements.prompt.value = prompt;
  elements.prompt.focus();
}

elements.refreshStatus?.addEventListener('click', refreshStatus);
elements.ingestMemory?.addEventListener('click', ingestMemory);
elements.fileInput?.addEventListener('change', updateSelectedFiles);
elements.uploadForm?.addEventListener('submit', uploadFiles);
elements.runQuery?.addEventListener('click', runQuery);
elements.launchVisualizer?.addEventListener('click', launchVisualizer);
elements.knowledgeList?.addEventListener('toggle', collapseSiblingKnowledgeRows, true);
elements.presetButtons.forEach((button) => button.addEventListener('click', applyPromptPreset));
elements.tabButtons.forEach((button) => {
  button.addEventListener('click', () => activateTab(button.dataset.tab));
});

refreshStatus();
