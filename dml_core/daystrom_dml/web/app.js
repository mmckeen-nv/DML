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
  runBaseLatencyMetric: $('#metric-run-base-latency'),
  runBaseLatencyDetail: $('#metric-run-base-detail'),
  runDmlLatencyMetric: $('#metric-run-dml-latency'),
  runDmlLatencyDetail: $('#metric-run-dml-latency-detail'),
  runRagLatencyMetric: $('#metric-run-rag-latency'),
  runRagLatencyDetail: $('#metric-run-rag-latency-detail'),
  graphBaseLatency: $('#graph-base-latency'),
  graphDmlLatency: $('#graph-dml-latency'),
  graphRagLatency: $('#graph-rag-latency'),
  graphBaseTokens: $('#graph-base-tokens'),
  graphDmlTokens: $('#graph-dml-tokens'),
  graphRagTokens: $('#graph-rag-tokens'),
  graphBaseAccuracy: $('#graph-base-accuracy'),
  graphDmlAccuracy: $('#graph-dml-accuracy'),
  graphRagAccuracy: $('#graph-rag-accuracy'),
  barBaseGeneration: $('#bar-base-generation'),
  barDmlRetrieval: $('#bar-dml-retrieval'),
  barDmlGeneration: $('#bar-dml-generation'),
  barRagRetrieval: $('#bar-rag-retrieval'),
  barRagGeneration: $('#bar-rag-generation'),
  barBaseInput: $('#bar-base-input'),
  barBaseOutput: $('#bar-base-output'),
  barDmlInput: $('#bar-dml-input'),
  barDmlOutput: $('#bar-dml-output'),
  barRagInput: $('#bar-rag-input'),
  barRagOutput: $('#bar-rag-output'),
  barBaseAccuracy: $('#bar-base-accuracy'),
  barDmlAccuracy: $('#bar-dml-accuracy'),
  barRagAccuracy: $('#bar-rag-accuracy'),
  runDmlTokens: $('#metric-run-dml-tokens'),
  runNodes: $('#metric-run-nodes'),
  runRagTokens: $('#metric-run-rag-tokens'),
  runDocs: $('#metric-run-docs'),
  accuracyKey: $('#metric-accuracy-key'),
  inferenceModel: $('#metric-inference-model'),
  inferenceBackend: $('#metric-inference-backend'),
  prompt: $('#prompt'),
  topK: $('#top-k'),
  maxTokens: $('#max-tokens'),
  runQuery: $('#run-query'),
  queryStatus: $('#query-status'),
  queryMode: $('#query-mode'),
  dmlContext: $('#dml-context'),
  dmlContextCount: $('#dml-context-count'),
  ragContext: $('#rag-context'),
  ragContextCount: $('#rag-context-count'),
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

function formatAccuracy(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  return `${Math.round(Number(value) * 100)}%`;
}

function formatMilliseconds(value) {
  if (value === null || value === undefined || value === '') return '-';
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

function estimateTokens(value) {
  const text = String(value || '').trim();
  if (!text) return 0;
  return Math.max(1, Math.ceil(text.length / 4));
}

function usageToken(usage, ...keys) {
  if (!usage) return null;
  for (const key of keys) {
    const value = usage[key];
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return numeric;
  }
  return null;
}

function formatInference(inference = {}) {
  const model = inference.model || 'unknown model';
  const backend = inference.backend || 'unknown backend';
  const backendLabel = backend;
  return { backendLabel, model };
}

function cleanDemoText(value) {
  return String(value || '')
    .replace(/\n?\[[^\]]*completion truncated\]/g, '')
    .trim();
}

function cleanDmlContext(value) {
  return cleanDemoText(value)
    .split('\n')
    .filter((line) => ![
      'Initializing Daystrom Memory Lattice v1.0',
      'Semantic coherence field stabilized.',
      'Cognitive resonance online.',
    ].includes(line.trim()))
    .join('\n')
    .trim();
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
  renderInference(adapter.inference);

  const status = state.health?.status || 'unknown';
  setStatus(elements.serviceStatus, status, status === 'ok' ? 'good' : 'warn');
}

function renderInference(inference = {}) {
  if (!elements.inferenceModel || !elements.inferenceBackend) return;
  const { backendLabel, model } = formatInference(inference);
  elements.inferenceModel.textContent = model;
  elements.inferenceBackend.textContent = backendLabel;
}

function entryId(entry) {
  const raw = entry?.id ?? entry?.meta?.memory_id;
  const numeric = Number(raw);
  return Number.isFinite(numeric) ? numeric : null;
}

function firstFinite(...values) {
  for (const value of values) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return numeric;
  }
  return null;
}

function mergeLatticeEntries(entries = [], highlightedEntries = []) {
  const merged = [...entries];
  const seen = new Set(merged.map(entryId).filter((id) => id !== null));
  for (const entry of highlightedEntries || []) {
    const id = entryId(entry);
    if (id === null || seen.has(id)) continue;
    merged.push(entry);
    seen.add(id);
  }
  return merged;
}

function visualLatticeEntries(entries = dmlEntries(), highlightedEntries = []) {
  const merged = mergeLatticeEntries(entries, highlightedEntries);
  const gridEntries = merged.filter((entry) => {
    const meta = entry.meta || {};
    return Number.isFinite(Number(meta.lattice_row)) && Number.isFinite(Number(meta.lattice_col));
  });
  const explicitLayers = merged
    .map((entry) => {
      const meta = entry.meta || {};
      return firstFinite(meta.lattice_layer, meta.layer, entry.level, meta.level, meta.cluster_index);
    })
    .filter((layer) => layer !== null);
  const hasLayerVariety = new Set(explicitLayers.map((layer) => Math.round(layer))).size > 1;
  const maxGridRow = gridEntries.length
    ? Math.max(...gridEntries.map((entry) => Number(entry.meta.lattice_row)))
    : 0;
  const maxGridCol = gridEntries.length
    ? Math.max(...gridEntries.map((entry) => Number(entry.meta.lattice_col)))
    : 0;
  const fallbackWidth = Math.max(7, maxGridCol + 1, Math.ceil(Math.sqrt(Math.max(1, merged.length))));
  let fallbackIndex = 0;

  return merged.map((entry) => {
    const meta = entry.meta || {};
    const row = Number(meta.lattice_row);
    const col = Number(meta.lattice_col);
    const hasGridPosition = Number.isFinite(row) && Number.isFinite(col);
    const explicitLayer = firstFinite(
      meta.lattice_layer,
      meta.layer,
      entry.level,
      meta.level,
      meta.cluster_index,
    );
    const derivedRow = hasGridPosition
      ? row
      : maxGridRow + 2 + Math.floor(fallbackIndex / fallbackWidth);
    const derivedCol = hasGridPosition
      ? col
      : fallbackIndex % fallbackWidth;
    const derivedIndex = fallbackIndex;
    fallbackIndex += hasGridPosition ? 0 : 1;
    const layer = explicitLayer !== null && hasLayerVariety
      ? Math.max(0, Math.round(explicitLayer))
      : hasGridPosition
        ? Math.floor((derivedRow + derivedCol) / Math.max(2, Math.ceil((maxGridRow + maxGridCol + 2) / 4)))
        : Math.floor(derivedIndex / Math.max(1, fallbackWidth * 2)) % 5;
    return {
      col: derivedCol,
      entry,
      id: entryId(entry),
      layer: clamp(layer, 0, 7),
      row: derivedRow,
    };
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
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const rx = x * cos - y * sin;
  const ry = x * sin + y * cos;
  return {
    x: originX + (rx - ry) * 32,
    y: originY + (rx + ry) * 18 - z,
  };
}

function fittedViewBoxForPoints(points, targetWidth, targetHeight, padding = 48) {
  const xs = points.map((point) => point.x).filter(Number.isFinite);
  const ys = points.map((point) => point.y).filter(Number.isFinite);
  if (!xs.length || !ys.length) return { x: 0, y: 0, width: targetWidth, height: targetHeight };
  let minX = Math.min(...xs) - padding;
  let maxX = Math.max(...xs) + padding;
  let minY = Math.min(...ys) - padding;
  let maxY = Math.max(...ys) + padding;
  let width = Math.max(1, maxX - minX);
  let height = Math.max(1, maxY - minY);
  const targetAspect = targetWidth / targetHeight;
  const aspect = width / height;
  if (aspect > targetAspect) {
    const nextHeight = width / targetAspect;
    const delta = (nextHeight - height) / 2;
    minY -= delta;
    height = nextHeight;
  } else {
    const nextWidth = height * targetAspect;
    const delta = (nextWidth - width) / 2;
    minX -= delta;
    width = nextWidth;
  }
  return { x: minX, y: minY, width, height };
}

function cameraViewBoxForPoints(points, targetWidth, targetHeight) {
  const base = fittedViewBoxForPoints(points, targetWidth, targetHeight);
  const zoom = clamp(Number(state.latticeView.zoom || 1), 0.62, 1.75);
  const width = base.width / zoom;
  const height = base.height / zoom;
  const x = base.x + (base.width - width) / 2 - Number(state.latticeView.panX || 0);
  const y = base.y + (base.height - height) / 2 - Number(state.latticeView.panY || 0);
  return `${x.toFixed(2)} ${y.toFixed(2)} ${width.toFixed(2)} ${height.toFixed(2)}`;
}

function renderLattice(entries = dmlEntries(), highlightedEntries = []) {
  if (!elements.latticeSvg) return;
  const latticeNodes = visualLatticeEntries(entries, highlightedEntries);
  const highlighted = new Set([
    ...state.highlightedNodeIds,
    ...highlightedEntries.map(entryId).filter((id) => id !== null),
  ]);

  if (!latticeNodes.length) {
    elements.latticeSvg.innerHTML = '';
    elements.visualizerPlaceholder.hidden = false;
    elements.visualizerPlaceholder.textContent = 'No lattice nodes are available yet.';
    setStatus(elements.visualizerStatus, 'empty', 'warn');
    return;
  }

  const rows = latticeNodes.map((node) => node.row);
  const cols = latticeNodes.map((node) => node.col);
  const maxRow = Math.max(...rows);
  const minRow = Math.min(...rows);
  const maxCol = Math.max(...cols);
  const minCol = Math.min(...cols);
  const maxLayer = Math.max(...latticeNodes.map((node) => node.layer));
  const layerGap = 24;
  const width = 940;
  const height = 620;
  const originX = width / 2;
  const originY = 150;
  const byId = new Map(latticeNodes.map((node) => [node.id, node]));
  const positions = new Map();
  const nodeRows = [];
  const lines = [];
  const columns = [];
  const nodes = [];
  const layerPlanes = [];
  const projectedPoints = [];

  for (let layer = 0; layer <= maxLayer; layer += 1) {
    const z = layer * layerGap;
    const corners = [
      projectPoint(minCol - maxCol / 2 - 0.35, minRow - maxRow / 2 - 0.35, z, originX, originY),
      projectPoint(maxCol - maxCol / 2 + 0.35, minRow - maxRow / 2 - 0.35, z, originX, originY),
      projectPoint(maxCol - maxCol / 2 + 0.35, maxRow - maxRow / 2 + 0.35, z, originX, originY),
      projectPoint(minCol - maxCol / 2 - 0.35, maxRow - maxRow / 2 + 0.35, z, originX, originY),
    ];
    projectedPoints.push(...corners);
    const points = corners.map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' ');
    layerPlanes.push(`<polygon class="lattice-plane layer-plane" style="--delay:${(layer * 0.18).toFixed(2)}s" points="${points}"></polygon>`);
  }

  for (const node of latticeNodes) {
    const { entry, id, layer } = node;
    const x = node.col - maxCol / 2;
    const y = node.row - maxRow / 2;
    const active = highlighted.has(id);
    const math = nodeMath(entry, active);
    const baseZ = layer * layerGap;
    const floor = projectPoint(x, y, baseZ, originX, originY);
    const top = projectPoint(x, y, baseZ + math.height, originX, originY);
    projectedPoints.push(floor, top);
    positions.set(id, { active, entry, floor, layer, math, top });
  }

  for (const node of latticeNodes) {
    const { entry, id } = node;
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
        `<line class="${active ? 'active signal-flow' : ''}" pathLength="100" x1="${position.top.x.toFixed(2)}" y1="${position.top.y.toFixed(2)}" x2="${neighborPosition.top.x.toFixed(2)}" y2="${neighborPosition.top.y.toFixed(2)}"></line>`
      );
    }
  }

  for (const [id, position] of positions.entries()) {
    const { active, entry, floor, layer, math, top } = position;
    const meta = entry.meta || {};
    const source = escapeHTML(meta.source || `node ${id}`);
    const label = escapeHTML(meta.summary || entry.summary || entry.text || source);
    const delay = ((id % 9) * 0.11).toFixed(2);
    columns.push(
      `<line class="${active ? 'active signal-flow' : ''}" pathLength="100" x1="${floor.x.toFixed(2)}" y1="${floor.y.toFixed(2)}" x2="${top.x.toFixed(2)}" y2="${top.y.toFixed(2)}"></line>`
    );
    nodeRows.push({
      markup:
      `<g class="lattice-node ${active ? 'active' : ''}" tabindex="0" style="--fidelity:${math.fidelity.toFixed(3)}; --delay:${delay}s">`
      + (active ? `<circle class="activation-ring" cx="${top.x.toFixed(2)}" cy="${top.y.toFixed(2)}" r="${(math.radius + 2).toFixed(2)}"></circle>` : '')
      + `<circle cx="${top.x.toFixed(2)}" cy="${top.y.toFixed(2)}" r="${math.radius.toFixed(2)}"></circle>`
      + `<title>${source}\nLayer ${layer}\n${label}</title>`
      + `</g>`,
      sortY: top.y,
    });
  }

  nodeRows.sort((a, b) => a.sortY - b.sortY);
  nodes.push(...nodeRows.map((row) => row.markup));

  elements.latticeSvg.setAttribute('viewBox', cameraViewBoxForPoints(projectedPoints, width, height));
  elements.latticeSvg.innerHTML = [
    ...layerPlanes,
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
    highlighted.size
      ? `${formatNumber([...highlighted].filter((id) => positions.has(id)).length)} active / ${formatNumber(latticeNodes.length)} nodes`
      : `${formatNumber(latticeNodes.length)} nodes · ${formatNumber(maxLayer + 1)} layers`,
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
  return cleanDemoText(payload?.dml?.response || payload?.dml_response || payload?.response || payload?.answer || '');
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

function latencyBreakdown(retrievalLatency, generationLatency) {
  const detailParts = [];
  if (retrievalLatency !== null && retrievalLatency !== undefined) {
    detailParts.push(`${formatMilliseconds(retrievalLatency)} retrieval`);
  }
  if (generationLatency !== null && generationLatency !== undefined) {
    detailParts.push(`${formatMilliseconds(generationLatency)} generation`);
  }
  return detailParts.join(' + ');
}

function combinedLatency(retrievalLatency, generationLatency, fallback = null) {
  const hasRetrieval = retrievalLatency !== null && retrievalLatency !== undefined;
  const hasGeneration = generationLatency !== null && generationLatency !== undefined;
  if (!hasRetrieval && !hasGeneration) return fallback;
  return Number(retrievalLatency || 0) + Number(generationLatency || 0);
}

function setSegmentWidth(element, value, maxValue) {
  if (!element) return;
  const numeric = Number(value || 0);
  const max = Math.max(Number(maxValue || 0), 1);
  element.style.width = `${Math.max(0, Math.min(100, (numeric / max) * 100))}%`;
}

function setAccuracyWidth(element, accuracy) {
  if (!element) return;
  if (!accuracy?.scored || accuracy.score === null || accuracy.score === undefined) {
    element.style.width = '0%';
    return;
  }
  setSegmentWidth(element, Number(accuracy.score), 1);
}

function renderRunTelemetry({
  baseLatency,
  retrievalLatency,
  generationLatency,
  ragRetrievalLatency,
  ragGenerationLatency,
  promptTokens,
  contextTokens,
  ragTokens,
  baseOutputTokens,
  dmlOutputTokens,
  ragOutputTokens,
  baseAccuracy,
  dmlAccuracy,
  ragAccuracy,
  answerKey,
  dmlNodes,
  ragDocs,
}) {
  const baseTotalTokens = promptTokens + baseOutputTokens;
  const dmlLatency = combinedLatency(retrievalLatency, generationLatency);
  const ragLatency = combinedLatency(ragRetrievalLatency, ragGenerationLatency);
  const dmlInputTokens = promptTokens + Number(contextTokens || 0);
  const ragInputTokens = promptTokens + Number(ragTokens || 0);
  const dmlTotalTokens = dmlInputTokens + dmlOutputTokens;
  const ragTotalTokens = ragInputTokens + ragOutputTokens;
  const dmlDetail = latencyBreakdown(retrievalLatency, generationLatency);
  const ragDetail = latencyBreakdown(ragRetrievalLatency, ragGenerationLatency);
  const maxLatency = Math.max(Number(baseLatency || 0), Number(dmlLatency || 0), Number(ragLatency || 0), 1);
  const maxTokens = Math.max(baseTotalTokens, dmlTotalTokens, ragTotalTokens, 1);

  elements.runBaseLatencyMetric.textContent = formatMilliseconds(baseLatency);
  elements.runBaseLatencyDetail.textContent = `${formatNumber(promptTokens)} in / ${formatNumber(baseOutputTokens)} out`;
  elements.runDmlLatencyMetric.textContent = formatMilliseconds(dmlLatency);
  elements.runDmlLatencyDetail.textContent = dmlDetail || 'not reported';
  elements.runRagLatencyMetric.textContent = formatMilliseconds(ragLatency);
  elements.runRagLatencyDetail.textContent = ragDetail || 'not reported';
  elements.graphBaseLatency.textContent = formatMilliseconds(baseLatency);
  elements.graphDmlLatency.textContent = formatMilliseconds(dmlLatency);
  elements.graphRagLatency.textContent = formatMilliseconds(ragLatency);
  elements.graphBaseTokens.textContent = formatNumber(baseTotalTokens);
  elements.graphDmlTokens.textContent = formatNumber(dmlTotalTokens);
  elements.graphRagTokens.textContent = formatNumber(ragTotalTokens);
  elements.graphBaseAccuracy.textContent = baseAccuracy?.scored ? formatAccuracy(baseAccuracy.score) : '-';
  elements.graphDmlAccuracy.textContent = dmlAccuracy?.scored ? formatAccuracy(dmlAccuracy.score) : '-';
  elements.graphRagAccuracy.textContent = ragAccuracy?.scored ? formatAccuracy(ragAccuracy.score) : '-';
  setSegmentWidth(elements.barBaseGeneration, baseLatency, maxLatency);
  setSegmentWidth(elements.barDmlRetrieval, retrievalLatency, maxLatency);
  setSegmentWidth(elements.barDmlGeneration, generationLatency, maxLatency);
  setSegmentWidth(elements.barRagRetrieval, ragRetrievalLatency, maxLatency);
  setSegmentWidth(elements.barRagGeneration, ragGenerationLatency, maxLatency);
  setSegmentWidth(elements.barBaseInput, promptTokens, maxTokens);
  setSegmentWidth(elements.barBaseOutput, baseOutputTokens, maxTokens);
  setSegmentWidth(elements.barDmlInput, dmlInputTokens, maxTokens);
  setSegmentWidth(elements.barDmlOutput, dmlOutputTokens, maxTokens);
  setSegmentWidth(elements.barRagInput, ragInputTokens, maxTokens);
  setSegmentWidth(elements.barRagOutput, ragOutputTokens, maxTokens);
  setAccuracyWidth(elements.barBaseAccuracy, baseAccuracy);
  setAccuracyWidth(elements.barDmlAccuracy, dmlAccuracy);
  setAccuracyWidth(elements.barRagAccuracy, ragAccuracy);
  elements.runDmlTokens.textContent = formatNumber(contextTokens);
  elements.runNodes.textContent = formatNumber(dmlNodes);
  elements.runRagTokens.textContent = formatNumber(ragTokens);
  elements.runDocs.textContent = formatNumber(ragDocs);
  elements.accuracyKey.textContent = answerKey?.facts?.length ? `${formatNumber(answerKey.facts.length)} facts` : 'not scored';
}

function renderRun(payload, fallbackMode = 'compare') {
  const response = responseFromCompare(payload);
  const entries = entriesFromCompare(payload);
  const ragBackend = primaryRagBackend(payload);
  const base = payload?.base || {};
  const baseResponse = cleanDemoText(base.response || '');
  const ragResponse = cleanDemoText(ragBackend?.response || payload?.rag?.response || '');
  const ragContext = cleanDemoText(ragBackend?.context || payload?.rag?.context || '');
  const stats = payload?.stats || {};
  const dml = payload?.dml || {};
  const dmlContext = cleanDmlContext(dml.context || payload?.dml_context || payload?.context || '');
  const ragTokens = ragTokenTotal(payload);
  const contextTokens = dml.context_tokens || payload?.context_tokens || payload?.tokens || 0;
  const promptTokens = estimateTokens(payload?.prompt || elements.prompt.value);
  const baseLatency = base.generation_latency_ms ?? null;
  const retrievalLatency = dml.retrieval_latency_ms ?? payload?.retrieval_latency_ms ?? null;
  const generationLatency = dml.generation_latency_ms ?? payload?.generation_latency_ms ?? null;
  const ragRetrievalLatency = ragBackend?.retrieval_latency_ms ?? payload?.rag?.retrieval_latency_ms ?? null;
  const ragGenerationLatency = ragBackend?.generation_latency_ms ?? payload?.rag?.generation_latency_ms ?? null;
  const latency = combinedLatency(retrievalLatency, generationLatency, dml.latency_ms || payload?.latency_ms);
  const wallLatency = payload?.latency_ms ?? latency;
  const dmlNodeCount = entries.length || Number(dml.entry_count || 0);
  const ragDocCount = ragBackend?.documents?.length || ragBackend?.docs?.length || ragBackend?.count || 0;
  const baseOutputTokens = usageToken(base.usage, 'completion_tokens', 'output_tokens', 'generated_tokens') ?? estimateTokens(baseResponse);
  const dmlOutputTokens = usageToken(dml.usage, 'completion_tokens', 'output_tokens', 'generated_tokens') ?? estimateTokens(response);
  const ragOutputTokens = usageToken(ragBackend?.usage, 'completion_tokens', 'output_tokens', 'generated_tokens') ?? estimateTokens(ragResponse);
  const baseAccuracy = base.accuracy || null;
  const dmlAccuracy = dml.accuracy || null;
  const ragAccuracy = ragBackend?.accuracy || payload?.rag?.accuracy || null;

  elements.dmlContext.textContent = dmlContext || 'No DML context was returned for this prompt.';
  elements.dmlContext.classList.toggle('empty', !dmlContext);
  elements.dmlContextCount.textContent = `${formatNumber(contextTokens)} tokens`;
  elements.ragContext.textContent = ragContext || 'No RAG context was returned for this prompt.';
  elements.ragContext.classList.toggle('empty', !ragContext);
  elements.ragContextCount.textContent = `${formatNumber(ragTokens)} tokens`;
  elements.response.textContent = response || 'No DML response returned.';
  elements.response.classList.toggle('empty', !response);
  elements.ragResponse.textContent = ragResponse || 'No RAG backend response returned.';
  elements.ragResponse.classList.toggle('empty', !ragResponse);
  elements.ragMode.textContent = ragBackend?.label || ragBackend?.id || 'rag';
  elements.responseMode.textContent = payload?.mode || fallbackMode;
  elements.queryMode.textContent = payload?.mode || fallbackMode;
  elements.runLatency.textContent = `wall ${formatMilliseconds(wallLatency)}`;
  renderInference(payload?.inference || state.health?.components?.adapter?.inference);
  renderRunTelemetry({
    baseLatency,
    retrievalLatency,
    generationLatency,
    ragRetrievalLatency,
    ragGenerationLatency,
    promptTokens,
    contextTokens,
    ragTokens,
    baseOutputTokens,
    dmlOutputTokens,
    ragOutputTokens,
    baseAccuracy,
    dmlAccuracy,
    ragAccuracy,
    answerKey: payload?.answer_key || null,
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
    state.latticeView.dragMode = event.button === 1 || event.button === 2 || event.shiftKey || event.altKey
      ? 'pan'
      : 'rotate';
    state.latticeView.lastX = event.clientX;
    state.latticeView.lastY = event.clientY;
    svg.classList.add('dragging');
    svg.classList.toggle('panning', state.latticeView.dragMode === 'pan');
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
    svg.classList.remove('panning');
  };

  svg.addEventListener('pointerdown', (event) => {
    if (![0, 1, 2].includes(event.button)) return;
    event.preventDefault();
    beginDrag(event);
    svg.setPointerCapture?.(event.pointerId);
  });
  window.addEventListener('pointermove', moveDrag);
  window.addEventListener('pointerup', endDrag);
  window.addEventListener('pointercancel', endDrag);
  svg.addEventListener('contextmenu', (event) => event.preventDefault());
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
