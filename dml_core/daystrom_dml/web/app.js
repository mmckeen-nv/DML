const $ = (selector) => document.querySelector(selector);

const state = {
  health: null,
  stats: null,
  knowledge: null,
  highlightedNodeIds: new Set(),
  latticeView: {
    angle: -0.75,
    dragMode: 'pan',
    dragging: false,
    lastX: 0,
    lastY: 0,
    panX: 0,
    panY: 0,
    zoom: 1,
  },
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
  prompt: $('#prompt'),
  topK: $('#top-k'),
  maxTokens: $('#max-tokens'),
  runQuery: $('#run-query'),
  queryStatus: $('#query-status'),
  queryMode: $('#query-mode'),
  response: $('#dml-response'),
  responseMode: $('#response-mode'),
  ragResponse: $('#rag-response'),
  ragMode: $('#rag-mode'),
  runLatency: $('#run-latency'),
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

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function nodeMath(entry, highlighted) {
  const meta = entry.meta || {};
  const tokenWeight = clamp(Number(entry.tokens || 0) / 120, 0, 1);
  const row = Number(meta.lattice_row || 0);
  const col = Number(meta.lattice_col || 0);
  const size = Math.max(1, Number(meta.lattice_size || 1) - 1);
  const positionWeight = clamp((row + col) / Math.max(1, size * 2), 0, 1);
  const salience = Number(entry.salience ?? meta.salience ?? (0.25 + tokenWeight * 0.45 + positionWeight * 0.3));
  const fidelity = Number(entry.fidelity ?? 1);
  const level = Number(entry.level ?? 0);
  const degree = Number(meta.lattice_degree ?? (meta.lattice_neighbors || []).length ?? 0);
  const height = 10
    + clamp(salience, 0, 1) * 46
    + clamp(fidelity, 0, 1) * 18
    + clamp(degree / 4, 0, 1) * 14
    + level * 16
    + (highlighted ? 38 : 0);
  const radius = 4.8 + clamp(salience, 0, 1) * 4.2 + (highlighted ? 3.4 : 0);
  return {
    degree,
    fidelity: clamp(fidelity, 0, 1),
    height,
    radius,
    salience: clamp(salience, 0, 1),
  };
}

function projectPoint(x, y, z, originX, originY, view = state.latticeView) {
  const angle = Number(view.angle || 0);
  const zoom = Number(view.zoom || 1);
  const panX = Number(view.panX || 0);
  const panY = Number(view.panY || 0);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const rx = x * cos - y * sin;
  const ry = x * sin + y * cos;
  return {
    x: originX + panX + (rx - ry) * 32 * zoom,
    y: originY + panY + (rx + ry) * 18 * zoom - z * zoom,
  };
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
  const width = 940;
  const height = 620;
  const originX = width / 2;
  const originY = 150;
  const byId = new Map(square.map((entry) => [entryId(entry), entry]));
  const positions = new Map();
  const nodeRows = [];
  const lines = [];
  const columns = [];
  const nodes = [];

  for (const entry of square) {
    const id = entryId(entry);
    const meta = entry.meta || {};
    const x = Number(meta.lattice_col) - maxCol / 2;
    const y = Number(meta.lattice_row) - maxRow / 2;
    const active = highlighted.has(id);
    const math = nodeMath(entry, active);
    const floor = projectPoint(x, y, 0, originX, originY);
    const top = projectPoint(x, y, math.height, originX, originY);
    positions.set(id, { active, entry, floor, math, top });
  }

  for (const entry of square) {
    const id = entryId(entry);
    const meta = entry.meta || {};
    const position = positions.get(id);
    if (!position) continue;
    for (const neighbor of meta.lattice_neighbors || []) {
      const neighborEntry = byId.get(Number(neighbor));
      if (!neighborEntry || Number(neighbor) < id) continue;
      const neighborPosition = positions.get(Number(neighbor));
      if (!neighborPosition) continue;
      const active = highlighted.has(id) && highlighted.has(Number(neighbor));
      lines.push(
        `<line class="${active ? 'active' : ''}" x1="${position.top.x.toFixed(2)}" y1="${position.top.y.toFixed(2)}" x2="${neighborPosition.top.x.toFixed(2)}" y2="${neighborPosition.top.y.toFixed(2)}"></line>`
      );
    }
  }

  for (const [id, position] of positions.entries()) {
    const { active, entry, floor, math, top } = position;
    const meta = entry.meta || {};
    const source = escapeHTML(meta.source || `node ${id}`);
    const label = escapeHTML(meta.summary || entry.summary || entry.text || source);
    columns.push(
      `<line class="${active ? 'active' : ''}" x1="${floor.x.toFixed(2)}" y1="${floor.y.toFixed(2)}" x2="${top.x.toFixed(2)}" y2="${top.y.toFixed(2)}"></line>`
    );
    nodeRows.push({
      markup:
      `<g class="lattice-node ${active ? 'active' : ''}" tabindex="0" style="--fidelity:${math.fidelity.toFixed(3)}">`
      + `<circle cx="${top.x.toFixed(2)}" cy="${top.y.toFixed(2)}" r="${math.radius.toFixed(2)}"></circle>`
      + `<title>${source}\n${label}</title>`
      + `</g>`,
      sortY: top.y,
    });
  }

  nodeRows.sort((a, b) => a.sortY - b.sortY);
  nodes.push(...nodeRows.map((row) => row.markup));

  const baseCorners = [
    projectPoint(-maxCol / 2, -maxRow / 2, 0, originX, originY),
    projectPoint(maxCol / 2, -maxRow / 2, 0, originX, originY),
    projectPoint(maxCol / 2, maxRow / 2, 0, originX, originY),
    projectPoint(-maxCol / 2, maxRow / 2, 0, originX, originY),
  ];
  const basePath = baseCorners
    .map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`)
    .join(' ');

  const axes = [
    ['x', projectPoint(-maxCol / 2, maxRow / 2 + 0.7, 0, originX, originY), projectPoint(maxCol / 2, maxRow / 2 + 0.7, 0, originX, originY)],
    ['y', projectPoint(maxCol / 2 + 0.7, -maxRow / 2, 0, originX, originY), projectPoint(maxCol / 2 + 0.7, maxRow / 2, 0, originX, originY)],
    ['z', projectPoint(maxCol / 2 + 0.9, maxRow / 2 + 0.9, 0, originX, originY), projectPoint(maxCol / 2 + 0.9, maxRow / 2 + 0.9, 96, originX, originY)],
  ].map(
    ([name, start, end]) =>
      `<line class="axis ${name}" x1="${start.x.toFixed(2)}" y1="${start.y.toFixed(2)}" x2="${end.x.toFixed(2)}" y2="${end.y.toFixed(2)}"></line>`
    );

  elements.latticeSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  elements.latticeSvg.innerHTML = [
    `<polygon class="lattice-plane" points="${basePath}"></polygon>`,
    '<g class="lattice-axes">',
    ...axes,
    '</g>',
    '<g class="lattice-columns">',
    ...columns,
    '</g>',
    '<g class="lattice-edges">',
    ...lines,
    '</g>',
    '<g class="lattice-nodes">',
    ...nodes,
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

function renderRunTelemetry({ latency, retrievalLatency, generationLatency, contextTokens, ragTokens, dmlNodes, ragDocs }) {
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
  elements.runNodes.textContent = formatNumber(dmlNodes);
  elements.runRagTokens.textContent = formatNumber(ragTokens);
  elements.runDocs.textContent = formatNumber(ragDocs);
}

function renderRun(payload, fallbackMode = 'compare') {
  const response = responseFromCompare(payload);
  const entries = entriesFromCompare(payload);
  const ragBackend = primaryRagBackend(payload);
  const ragResponse = ragBackend?.response || payload?.rag?.response || '';
  const stats = payload?.stats || {};
  const dml = payload?.dml || {};
  const ragTokens = ragTokenTotal(payload);
  const contextTokens = dml.context_tokens || payload?.context_tokens || payload?.tokens || 0;
  const retrievalLatency = dml.retrieval_latency_ms ?? payload?.retrieval_latency_ms ?? null;
  const generationLatency = dml.generation_latency_ms ?? payload?.generation_latency_ms ?? null;
  const latency = (Number(retrievalLatency || 0) + Number(generationLatency || 0)) || dml.latency_ms || payload?.latency_ms;
  const dmlNodeCount = entries.length || Number(dml.entry_count || 0);
  const ragDocCount = ragBackend?.documents?.length || ragBackend?.docs?.length || ragBackend?.count || 0;

  elements.response.textContent = response || 'No DML response returned.';
  elements.response.classList.toggle('empty', !response);
  elements.ragResponse.textContent = ragResponse || 'No RAG backend response returned.';
  elements.ragResponse.classList.toggle('empty', !ragResponse);
  elements.ragMode.textContent = ragBackend?.label || ragBackend?.id || 'rag';
  elements.responseMode.textContent = payload?.mode || fallbackMode;
  elements.queryMode.textContent = payload?.mode || fallbackMode;
  elements.runLatency.textContent = formatMilliseconds(latency);
  renderRunTelemetry({
    latency,
    retrievalLatency,
    generationLatency,
    contextTokens,
    ragTokens,
    dmlNodes: dmlNodeCount,
    ragDocs: ragDocCount,
  });
  state.highlightedNodeIds = new Set(entries.map(entryId).filter((id) => id !== null));
  renderLattice(dmlEntries(), entries);

  if (stats.memories || stats.memory_count) {
    state.stats = { ...state.stats, ...stats };
    renderMetrics();
  }
}

function pointerDeltaToViewBox(svg, deltaX, deltaY) {
  const rect = svg.getBoundingClientRect();
  const viewBox = svg.viewBox?.baseVal;
  if (!rect.width || !rect.height || !viewBox) {
    return { x: deltaX, y: deltaY };
  }
  return {
    x: deltaX * (viewBox.width / rect.width),
    y: deltaY * (viewBox.height / rect.height),
  };
}

function updateLatticeView({ deltaAngle = 0, deltaPanX = 0, deltaPanY = 0, zoomMultiplier = 1 } = {}) {
  state.latticeView.angle += deltaAngle;
  state.latticeView.panX += deltaPanX;
  state.latticeView.panY += deltaPanY;
  state.latticeView.zoom = clamp(state.latticeView.zoom * zoomMultiplier, 0.62, 1.75);
  renderLattice(dmlEntries());
}

function setupLatticeControls() {
  const svg = elements.latticeSvg;
  if (!svg) return;

  const beginDrag = (event) => {
    state.latticeView.dragging = true;
    state.latticeView.dragMode = event.shiftKey ? 'rotate' : 'pan';
    state.latticeView.lastX = event.clientX;
    state.latticeView.lastY = event.clientY;
    svg.classList.add('dragging');
  };
  const moveDrag = (event) => {
    if (!state.latticeView.dragging) return;
    const deltaX = event.clientX - state.latticeView.lastX;
    const deltaY = event.clientY - state.latticeView.lastY;
    state.latticeView.lastX = event.clientX;
    state.latticeView.lastY = event.clientY;
    if (state.latticeView.dragMode === 'rotate') {
      updateLatticeView({ deltaAngle: deltaX * 0.008 });
      return;
    }
    const pan = pointerDeltaToViewBox(svg, deltaX, deltaY);
    updateLatticeView({ deltaPanX: pan.x, deltaPanY: pan.y });
  };
  const endDrag = () => {
    state.latticeView.dragging = false;
    svg.classList.remove('dragging');
  };

  svg.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    beginDrag(event);
    svg.setPointerCapture?.(event.pointerId);
  });
  window.addEventListener('pointermove', moveDrag);
  window.addEventListener('pointerup', endDrag);
  window.addEventListener('pointercancel', endDrag);
  svg.addEventListener('wheel', (event) => {
    event.preventDefault();
    updateLatticeView({ zoomMultiplier: event.deltaY < 0 ? 1.08 : 0.92 });
  }, { passive: false });
  svg.addEventListener('dblclick', () => {
    state.latticeView.angle = -0.75;
    state.latticeView.panX = 0;
    state.latticeView.panY = 0;
    state.latticeView.zoom = 1;
    renderLattice(dmlEntries());
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

setupLatticeControls();
refreshStatus();
