const uploadForm = document.querySelector('#upload-form');
const fileInput = document.querySelector('#file-input');
const uploadStatus = document.querySelector('#upload-status');
const promptInput = document.querySelector('#prompt');
const topKInput = document.querySelector('#top-k');
const maxTokensInput = document.querySelector('#max-tokens');
const runCompareButton = document.querySelector('#run-compare');
const compareStatus = document.querySelector('#compare-status');
const resultsPanel = document.querySelector('#results');
const tokenPromptMetric = document.querySelector('#metric-prompt-tokens');
const tokenRagMetric = document.querySelector('#metric-rag-tokens');
const tokenDmlMetric = document.querySelector('#metric-dml-tokens');
const tokenDeltaMetric = document.querySelector('#metric-token-delta');
const dmlFidelityMetric = document.querySelector('#metric-dml-fidelity');
const ragDocsMetric = document.querySelector('#metric-rag-docs');
const dmlEntriesMetric = document.querySelector('#metric-dml-entries');
const baseOutput = document.querySelector('#base-output');
const dmlOutput = document.querySelector('#dml-output');
const integratedOutput = document.querySelector('#integrated-output');
const baseUsage = document.querySelector('#base-usage');
const dmlUsage = document.querySelector('#dml-usage');
const integratedUsage = document.querySelector('#integrated-usage');
const ragContext = document.querySelector('#rag-context');
const ragContextLabel = document.querySelector('#rag-context-label');
const dmlContext = document.querySelector('#dml-context');
const dmlSummaryList = document.querySelector('#dml-summary-list');
const ragContextLlmOutput = document.querySelector('#rag-context-llm-output');
const dmlContextLlmOutput = document.querySelector('#dml-context-llm-output');
const integratedContextLlmOutput = document.querySelector('#integrated-context-llm-output');
const insightCopy = document.querySelector('#insight-copy');
const ragDocumentsTable = document.querySelector('#rag-documents tbody');
const ragTokenGraph = document.querySelector('#rag-token-graph');
const ragResponseTabList = document.querySelector('#rag-response-tablist');
const ragResponsePanels = document.querySelector('#rag-response-panels');
const contextTabButtons = Array.from(document.querySelectorAll('[data-context-tab]'));
const contextPanels = Array.from(document.querySelectorAll('[data-context-panel]'));
const dmlEntriesTable = document.querySelector('#dml-entries tbody');
const knowledgeStatus = document.querySelector('#knowledge-status');
const ragKnowledgeTable = document.querySelector('#rag-knowledge tbody');
const dmlKnowledgeTable = document.querySelector('#dml-knowledge tbody');
const ragKnowledgeCount = document.querySelector('#knowledge-rag-count');
const ragKnowledgeTokens = document.querySelector('#knowledge-rag-tokens');
const dmlKnowledgeCount = document.querySelector('#knowledge-dml-count');
const dmlKnowledgeTokens = document.querySelector('#knowledge-dml-tokens');
const nimImageInput = document.querySelector('#nim-image');
const ngcApiKeyInput = document.querySelector('#ngc-api-key');
const configureNimButton = document.querySelector('#configure-nim');
const nimStatus = document.querySelector('#nim-status');
const nimDetails = document.querySelector('#nim-details');
const nimConfigSummary = document.querySelector('#nim-config-summary');
const startNimButton = document.querySelector('#start-nim');
const stopNimButton = document.querySelector('#stop-nim');
const nimRuntimeStatus = document.querySelector('#nim-runtime-status');
const visualizeButton = document.querySelector('#visualize-button');
const visualizerFrame = document.querySelector('#visualizer-frame');
const visualizerStatus = document.querySelector('#visualizer-status');
const visualizerInlineLink = document.querySelector('#visualizer-inline-link');

const API = {
  upload: '/upload',
  compare: '/rag/compare',
  nimOptions: '/nim/options',
  nimConfigure: '/nim/configure',
  nimStart: '/nim/start',
  nimStop: '/nim/stop',
  knowledge: '/knowledge',
  visualizerLaunch: '/visualizer/launch',
};

let nimConfigured = false;
const state = {
  ragBackends: [],
  activeRagId: null,
  lastComparison: null,
  lastTokenBreakdown: [],
  lastRequest: null,
  visualizer: {
    embedUrl: null,
    targetUrl: null,
    lastPrompt: null,
    mode: 'auto',
  },
};

if (nimImageInput && configureNimButton && nimStatus) {
  loadNimStatus();
  configureNimButton.addEventListener('click', configureNimEndpoint);
}

if (visualizeButton) {
  configureVisualizerButton();
}

if (visualizerFrame && visualizerStatus) {
  prepareEmbeddedVisualizer();
}

if (startNimButton && stopNimButton) {
  startNimButton.disabled = true;
  stopNimButton.disabled = true;
  startNimButton.addEventListener('click', startNimContainer);
  stopNimButton.addEventListener('click', stopNimContainer);
}

if (contextTabButtons.length && contextPanels.length) {
  contextTabButtons.forEach((button) => {
    button.addEventListener('click', () => activateContextTab(button.dataset.contextTab));
  });
  activateContextTab(contextTabButtons[0]?.dataset.contextTab || 'rag');
}

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!fileInput.files || fileInput.files.length === 0) {
    uploadStatus.textContent = 'Please choose a file first.';
    return;
  }
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  uploadStatus.textContent = 'Uploading…';
  try {
    const response = await fetch(API.upload, { method: 'POST', body: formData });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Upload failed');
    }
    const payload = await response.json();
    uploadStatus.textContent = `Ingested ${payload.chunks} chunk(s) (~${payload.tokens} tokens) into RAG and the DML.`;
    fileInput.value = '';
    refreshKnowledge();
  } catch (err) {
    console.error(err);
    uploadStatus.textContent = `Error: ${err.message}`;
  }
});

runCompareButton.addEventListener('click', async () => {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    compareStatus.textContent = 'Enter a prompt to compare.';
    return;
  }
  compareStatus.textContent = 'Fetching context and generating responses…';
  resultsPanel.classList.add('hidden');
  try {
    const body = {
      prompt,
      top_k: Number(topKInput.value) || 0,
      max_new_tokens: Number(maxTokensInput.value) || 512,
    };
    state.lastRequest = { ...body };
    const response = await fetch(API.compare, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Compare request failed');
    }
    const payload = await response.json();
    renderResults(payload);
    compareStatus.textContent = 'Done.';
  } catch (err) {
    console.error(err);
    compareStatus.textContent = `Error: ${err.message}`;
  }
});

function renderResults(payload) {
  resultsPanel.classList.remove('hidden');
  const nf = new Intl.NumberFormat('en-US');
  const promptTokens = payload.prompt_tokens_est ?? 0;

  state.lastComparison = payload;
  state.lastTokenBreakdown = Array.isArray(payload.rag_token_breakdown)
    ? payload.rag_token_breakdown
    : [];
  if (payload.prompt) {
    state.visualizer.mode = 'auto';
    updateVisualizerPrompt(payload.prompt);
  }
  const ragBackends = Array.isArray(payload.rag_backends) ? payload.rag_backends : [];
  state.ragBackends = ragBackends;
  if (!state.activeRagId || !ragBackends.some((backend) => backend.id === state.activeRagId)) {
    state.activeRagId = ragBackends.length ? ragBackends[0].id : null;
  }
  buildRagResponseTabs(ragBackends);
  setActiveRagBackend(state.activeRagId);

  const primaryRag = payload.rag || getActiveRagBackend();
  const ragTokens = primaryRag?.context_tokens ?? 0;
  const dmlTokens = payload.dml?.context_tokens ?? 0;
  setMetricValue(tokenPromptMetric, promptTokens);
  setMetricValue(tokenRagMetric, ragTokens);
  setMetricValue(tokenDmlMetric, dmlTokens);
  const tokenDelta = dmlTokens && ragTokens ? ragTokens - dmlTokens : 0;
  tokenDeltaMetric.textContent = tokenDelta
    ? `${tokenDelta > 0 ? '−' : '+'}${nf.format(Math.abs(tokenDelta))} tokens`
    : '0';
  dmlFidelityMetric.textContent = formatFloat(payload.dml?.avg_fidelity);
  ragDocsMetric.textContent = nf.format(primaryRag?.documents?.length ?? 0);
  dmlEntriesMetric.textContent = nf.format(payload.dml?.entries?.length ?? 0);

  if (baseOutput) baseOutput.textContent = payload.base?.response || '';
  if (dmlOutput) dmlOutput.textContent = payload.dml?.response || '';
  if (integratedOutput) integratedOutput.textContent = payload.integrated?.response || '';

  if (baseUsage) {
    const usageText = formatUsage(payload.base?.usage);
    baseUsage.textContent = usageText;
    baseUsage.classList.toggle('hidden', !usageText);
  }
  if (dmlUsage) {
    const usageText = formatUsage(payload.dml?.usage);
    dmlUsage.textContent = usageText;
    dmlUsage.classList.toggle('hidden', !usageText);
  }
  if (integratedUsage) {
    const usageText = formatUsage(payload.integrated?.usage);
    integratedUsage.textContent = usageText;
    integratedUsage.classList.toggle('hidden', !usageText);
  }

  if (dmlContext)
    dmlContext.textContent = payload.dml?.context || 'No DML memories matched this prompt yet.';
  renderDmlEntries(payload.dml?.entries || []);
  renderDmlSummaries(payload.dml?.entries || []);

  if (dmlContextLlmOutput) {
    dmlContextLlmOutput.textContent = payload.dml?.response || 'No DML response generated yet.';
  }
  if (integratedContextLlmOutput) {
    integratedContextLlmOutput.textContent = payload.integrated?.response || 'No integrated response generated yet.';
  }

  if (insightCopy) {
    insightCopy.textContent = buildInsightCopy({
      promptTokens,
      ragTokens,
      dmlTokens,
      tokenDelta,
      avgFidelity: payload.dml?.avg_fidelity,
      ragCount: primaryRag?.documents?.length || 0,
      dmlCount: payload.dml?.entries?.length || 0,
    });
  }

  renderTokenGraph(state.lastTokenBreakdown);
  refreshKnowledge();
}

function getActiveRagBackend() {
  if (!state.activeRagId) {
    return state.ragBackends[0] || null;
  }
  return state.ragBackends.find((backend) => backend.id === state.activeRagId) || null;
}

function buildRagResponseTabs(backends) {
  if (!ragResponseTabList || !ragResponsePanels) {
    return;
  }
  ragResponseTabList.innerHTML = '';
  ragResponsePanels.innerHTML = '';
  if (!backends.length) {
    const emptyMessage = document.createElement('p');
    emptyMessage.className = 'empty-state';
    emptyMessage.textContent = 'No RAG responses generated yet.';
    ragResponsePanels.appendChild(emptyMessage);
    return;
  }
  const selectedId = state.activeRagId;
  backends.forEach((backend, index) => {
    const safeId = `rag-tab-${String(backend.id || index).replace(/[^a-z0-9-_]/gi, '-')}`;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'tab-chip';
    button.dataset.backendId = backend.id;
    button.id = safeId;
    button.setAttribute('role', 'tab');
    button.textContent = backend.label || backend.id || `Backend ${index + 1}`;
    button.addEventListener('click', () => setActiveRagBackend(backend.id));
    ragResponseTabList.appendChild(button);

    const panel = document.createElement('div');
    panel.className = 'rag-response-panel';
    panel.dataset.backendId = backend.id;
    panel.setAttribute('role', 'tabpanel');
    panel.setAttribute('aria-labelledby', safeId);
    const responsePre = document.createElement('pre');
    responsePre.className = 'rag-response-output';
    responsePre.textContent = backend.response || 'No response generated yet.';
    const usageFooter = document.createElement('footer');
    usageFooter.className = 'usage';
    usageFooter.textContent = formatUsage(backend.usage);
    panel.appendChild(responsePre);
    panel.appendChild(usageFooter);
    ragResponsePanels.appendChild(panel);
  });
}

function setActiveRagBackend(backendId) {
  if (!state.ragBackends.length) {
    state.activeRagId = null;
  } else if (backendId && state.ragBackends.some((backend) => backend.id === backendId)) {
    state.activeRagId = backendId;
  } else {
    state.activeRagId = state.ragBackends[0].id;
  }
  if (ragResponseTabList && ragResponsePanels) {
    const buttons = Array.from(ragResponseTabList.querySelectorAll('[data-backend-id]'));
    const panels = Array.from(ragResponsePanels.querySelectorAll('.rag-response-panel'));
    buttons.forEach((button) => {
      const isActive = button.dataset.backendId === state.activeRagId;
      button.classList.toggle('active', isActive);
      button.setAttribute('aria-selected', isActive ? 'true' : 'false');
      button.setAttribute('tabindex', isActive ? '0' : '-1');
    });
    panels.forEach((panel) => {
      const isActive = panel.dataset.backendId === state.activeRagId;
      panel.classList.toggle('active', isActive);
      panel.hidden = !isActive;
      if (isActive) {
        const backend = getActiveRagBackend();
        if (backend) {
          const responsePre = panel.querySelector('.rag-response-output');
          if (responsePre) {
            responsePre.textContent = backend.response || 'No response generated yet.';
          }
          const usageFooter = panel.querySelector('.usage');
          if (usageFooter) {
            usageFooter.textContent = formatUsage(backend.usage);
          }
        }
      }
    });
  }
  updateRagContextView();
  renderTokenGraph(state.lastTokenBreakdown || []);
}

function updateRagContextView() {
  const backend = getActiveRagBackend();
  if (ragContextLabel) {
    ragContextLabel.textContent = backend
      ? `${backend.label || backend.id} context`
      : 'No RAG backend selected.';
  }
  if (ragContext) {
    ragContext.textContent = backend?.context || 'No RAG context retrieved for this prompt.';
  }
  renderRagDocuments(backend?.documents || []);
  if (ragContextLlmOutput) {
    ragContextLlmOutput.textContent = backend?.response || 'No RAG response generated yet.';
  }
}

function activateContextTab(tabId) {
  if (!tabId) {
    return;
  }
  contextTabButtons.forEach((button) => {
    const isActive = button.dataset.contextTab === tabId;
    button.classList.toggle('active', isActive);
    button.setAttribute('aria-selected', isActive ? 'true' : 'false');
    button.setAttribute('tabindex', isActive ? '0' : '-1');
  });
  contextPanels.forEach((panel) => {
    const isActive = panel.dataset.contextPanel === tabId;
    panel.classList.toggle('active', isActive);
    panel.hidden = !isActive;
  });
}

function renderTokenGraph(breakdown) {
  if (!ragTokenGraph) {
    return;
  }
  state.lastTokenBreakdown = Array.isArray(breakdown) ? breakdown : [];
  const entries = state.lastTokenBreakdown;
  ragTokenGraph.innerHTML = '';
  if (!entries.length) {
    const emptyMessage = document.createElement('p');
    emptyMessage.className = 'empty-state';
    emptyMessage.textContent = 'No RAG retrievals yet.';
    ragTokenGraph.appendChild(emptyMessage);
    return;
  }
  const nf = new Intl.NumberFormat('en-US');
  const maxTokens = Math.max(...entries.map((entry) => Number(entry.tokens) || 0), 1);
  entries.forEach((entry) => {
    const bar = document.createElement('div');
    bar.className = 'token-graph-bar';
    if (entry.id === state.activeRagId) {
      bar.classList.add('active');
    }
    const label = document.createElement('span');
    label.className = 'token-graph-label';
    label.textContent = entry.label || entry.id || 'Backend';
    const meter = document.createElement('div');
    meter.className = 'token-graph-meter';
    const fill = document.createElement('div');
    fill.className = 'token-graph-fill';
    const ratio = Math.min(100, ((Number(entry.tokens) || 0) / maxTokens) * 100);
    fill.style.width = `${ratio}%`;
    meter.appendChild(fill);
    const value = document.createElement('span');
    value.className = 'token-graph-value';
    value.textContent = `${nf.format(Number(entry.tokens) || 0)} tok`;
    bar.appendChild(label);
    bar.appendChild(meter);
    bar.appendChild(value);
    ragTokenGraph.appendChild(bar);
  });
}

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function setMetricValue(target, value) {
  if (!target) return;
  const nf = new Intl.NumberFormat('en-US');
  const num = Number(value || 0);
  target.textContent = nf.format(num);
}

function formatFloat(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '–';
  }
  return Number(value).toFixed(2);
}

function formatUsage(usage) {
  if (!usage) {
    return '';
  }
  const nf = new Intl.NumberFormat('en-US');
  const prompt = usage.prompt_tokens ?? usage.promptTokens;
  const completion = usage.completion_tokens ?? usage.completionTokens;
  const total = usage.total_tokens ?? usage.totalTokens;
  const pieces = [];
  if (prompt !== undefined) pieces.push(`Prompt: ${nf.format(prompt)}`);
  if (completion !== undefined) pieces.push(`Completion: ${nf.format(completion)}`);
  if (total !== undefined) pieces.push(`Total: ${nf.format(total)}`);
  return pieces.length ? pieces.join(' | ') : '';
}

function getVisualizerTopK() {
  if (state.lastRequest && typeof state.lastRequest.top_k === 'number') {
    const requestTopK = Number(state.lastRequest.top_k);
    if (!Number.isNaN(requestTopK) && requestTopK > 0) {
      return requestTopK;
    }
  }
  if (topKInput && topKInput.value) {
    const direct = Number(topKInput.value);
    if (!Number.isNaN(direct) && direct > 0) {
      return direct;
    }
  }
  return 6;
}

function buildVisualizerUrl(baseUrl, prompt, topK, mode = 'auto') {
  if (!baseUrl) {
    return null;
  }
  try {
    const resolved = new URL(baseUrl, window.location.origin);
    if (prompt) {
      resolved.searchParams.set('prompt', prompt);
    } else {
      resolved.searchParams.delete('prompt');
    }
    if (topK) {
      resolved.searchParams.set('top_k', String(topK));
    } else {
      resolved.searchParams.delete('top_k');
    }
    if (mode) {
      resolved.searchParams.set('mode', mode);
    }
    resolved.searchParams.set('ts', Date.now().toString());
    return resolved.toString();
  } catch (err) {
    console.warn('Unable to build visualizer URL:', err);
    return baseUrl;
  }
}

function updateVisualizerPrompt(prompt) {
  state.visualizer.lastPrompt = prompt;
  const topK = getVisualizerTopK();
  const mode = state.visualizer.mode || 'auto';
  const embedUrl = buildVisualizerUrl(state.visualizer.embedUrl, prompt, topK, mode);
  const targetUrl = buildVisualizerUrl(state.visualizer.targetUrl, prompt, topK, mode);
  if (embedUrl && visualizerFrame) {
    if (visualizerFrame.src !== embedUrl) {
      visualizerFrame.src = embedUrl;
    }
  }
  if (targetUrl && visualizerInlineLink) {
    visualizerInlineLink.href = targetUrl;
  }
  if (targetUrl && visualizeButton) {
    visualizeButton.dataset.launchUrl = targetUrl;
    visualizeButton.href = targetUrl;
  }
}

function renderRagDocuments(documents) {
  if (!ragDocumentsTable) {
    return;
  }
  ragDocumentsTable.innerHTML = '';
  if (!documents.length) {
    const emptyRow = document.createElement('tr');
    emptyRow.innerHTML = '<td colspan="4">No matching RAG documents for this backend yet.</td>';
    ragDocumentsTable.appendChild(emptyRow);
    return;
  }
  documents.forEach((doc, idx) => {
    const row = document.createElement('tr');
    const source = doc.meta?.doc_path || doc.meta?.source || 'uploaded document';
    row.innerHTML = `
      <td>${idx + 1}</td>
      <td>${Number(doc.score ?? 0).toFixed(3)}</td>
      <td>${doc.tokens ?? 0}</td>
      <td>${escapeHtml(source)}</td>
    `;
    ragDocumentsTable.appendChild(row);
  });
}

function renderDmlEntries(entries) {
  if (!dmlEntriesTable) {
    return;
  }
  dmlEntriesTable.innerHTML = '';
  if (!entries.length) {
    const emptyRow = document.createElement('tr');
    emptyRow.innerHTML = '<td colspan="5">No DML memories retrieved.</td>';
    dmlEntriesTable.appendChild(emptyRow);
    return;
  }
  entries.forEach((entry) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${entry.id}</td>
      <td>L${entry.level}</td>
      <td>${Number(entry.fidelity ?? 0).toFixed(2)}</td>
      <td>${entry.tokens ?? 0}</td>
      <td>${escapeHtml(entry.summary ?? '')}</td>
    `;
    dmlEntriesTable.appendChild(row);
  });
}

function renderDmlSummaries(entries) {
  if (!dmlSummaryList) {
    return;
  }
  dmlSummaryList.innerHTML = '';
  if (!entries.length) {
    const emptyItem = document.createElement('li');
    emptyItem.className = 'summary-empty';
    emptyItem.textContent = 'No DML memories retrieved.';
    dmlSummaryList.appendChild(emptyItem);
    return;
  }
  entries.forEach((entry) => {
    const item = document.createElement('li');
    const meta = document.createElement('div');
    meta.className = 'summary-meta';
    const details = [];
    details.push(entry.level !== undefined && entry.level !== null ? `L${entry.level}` : 'L?');
    if (entry.fidelity !== undefined && entry.fidelity !== null) {
      details.push(`f=${Number(entry.fidelity).toFixed(2)}`);
    }
    if (entry.tokens !== undefined && entry.tokens !== null) {
      details.push(`${entry.tokens} tok`);
    }
    meta.textContent = details.join(' • ');
    const summary = document.createElement('p');
    summary.textContent = entry.summary || 'No summary available.';
    item.appendChild(meta);
    item.appendChild(summary);
    dmlSummaryList.appendChild(item);
  });
}

async function refreshKnowledge() {
  if (!knowledgeStatus) {
    return;
  }
  knowledgeStatus.textContent = 'Refreshing knowledge summaries…';
  try {
    const response = await fetch(API.knowledge);
    if (!response.ok) {
      throw new Error('Failed to load knowledge summaries');
    }
    const payload = await response.json();
    renderKnowledge(payload);
    const hasKnowledge = (payload.rag?.count || 0) + (payload.dml?.count || 0) > 0;
    knowledgeStatus.textContent = hasKnowledge
      ? ''
      : 'No documents have been ingested into the knowledge bases yet.';
  } catch (err) {
    console.error(err);
    knowledgeStatus.textContent = `Error: ${err.message}`;
  }
}

function renderKnowledge(payload) {
  if (!payload) {
    return;
  }
  setMetricValue(ragKnowledgeCount, payload.rag?.count ?? 0);
  setMetricValue(ragKnowledgeTokens, payload.rag?.total_tokens ?? 0);
  setMetricValue(dmlKnowledgeCount, payload.dml?.count ?? 0);
  setMetricValue(dmlKnowledgeTokens, payload.dml?.total_tokens ?? 0);
  renderRagKnowledge(payload.rag?.documents || []);
  renderDmlKnowledge(payload.dml?.entries || []);
}

function renderRagKnowledge(documents) {
  if (!ragKnowledgeTable) {
    return;
  }
  ragKnowledgeTable.innerHTML = '';
  if (!documents.length) {
    const row = document.createElement('tr');
    row.innerHTML = '<td colspan="3">No documents ingested yet.</td>';
    ragKnowledgeTable.appendChild(row);
    return;
  }
  documents.forEach((doc) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${doc.index}</td>
      <td>${doc.tokens ?? 0}</td>
      <td>${escapeHtml(doc.source || 'uploaded document')}</td>
    `;
    ragKnowledgeTable.appendChild(row);
  });
}

function renderDmlKnowledge(entries) {
  if (!dmlKnowledgeTable) {
    return;
  }
  dmlKnowledgeTable.innerHTML = '';
  if (!entries.length) {
    const row = document.createElement('tr');
    row.innerHTML = '<td colspan="5">No DML memories stored yet.</td>';
    dmlKnowledgeTable.appendChild(row);
    return;
  }
  entries.forEach((entry) => {
    const summary = truncateText(entry.summary || '');
    const fidelity = entry.fidelity !== undefined && entry.fidelity !== null
      ? Number(entry.fidelity).toFixed(2)
      : '–';
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${entry.id}</td>
      <td>L${entry.level}</td>
      <td>${fidelity}</td>
      <td>${entry.tokens ?? 0}</td>
      <td>${escapeHtml(summary || 'No summary available.')}</td>
    `;
    dmlKnowledgeTable.appendChild(row);
  });
}

function truncateText(value, limit = 160) {
  if (!value) {
    return '';
  }
  const normalized = value.replace(/\s+/g, ' ').trim();
  if (normalized.length <= limit) {
    return normalized;
  }
  return `${normalized.slice(0, limit - 1)}…`;
}

function buildInsightCopy({ promptTokens, ragTokens, dmlTokens, tokenDelta, avgFidelity, ragCount, dmlCount }) {
  if (!ragTokens && !dmlTokens) {
    return 'No retrieval context has been generated yet. Upload documents to populate RAG and the DML.';
  }
  const nf = new Intl.NumberFormat('en-US');
  const parts = [];
  if (ragCount) {
    parts.push(`RAG contributed ${nf.format(ragCount)} document chunk${ragCount === 1 ? '' : 's'} totalling ${nf.format(ragTokens)} tokens.`);
  }
  if (dmlCount) {
    const fidelityText = avgFidelity !== undefined && avgFidelity !== null ? ` with an average fidelity of ${Number(avgFidelity).toFixed(2)}` : '';
    parts.push(`The Daystrom Memory Lattice surfaced ${nf.format(dmlCount)} memory node${dmlCount === 1 ? '' : 's'}${fidelityText} and ${nf.format(dmlTokens)} contextual tokens.`);
  }
  if (tokenDelta) {
    const direction = tokenDelta > 0 ? 'fewer' : 'more';
    parts.push(`Compared to RAG alone, the DML context uses ${nf.format(Math.abs(tokenDelta))} ${direction} tokens, highlighting the distinct retrieval category.`);
  }
  parts.push(`The user prompt spans approximately ${nf.format(promptTokens)} tokens.`);
  return parts.join(' ');
}

async function loadNimStatus() {
  nimStatus.textContent = 'Checking NVIDIA NIM configuration…';
  try {
    const response = await fetch(API.nimOptions);
    if (!response.ok) {
      throw new Error('Failed to load NIM status');
    }
    const payload = await response.json();
    nimConfigured = Boolean(payload.current);
    if (nimImageInput && !nimImageInput.value && payload.default?.image) {
      nimImageInput.value = payload.default.image;
    }
    updateRuntimeStatus(payload.runtime, nimConfigured);
    if (payload.current) {
      renderNimSummary(payload.current, 'Using previously configured NIM.', payload);
    } else {
      nimStatus.textContent = 'Input the NVIDIA NIM image and provide your NGC API key to begin.';
    }
  } catch (err) {
    console.error(err);
    nimStatus.textContent = `Error: ${err.message}`;
  }
}

async function configureNimEndpoint() {
  const nimImage = (nimImageInput.value || '').trim();
  const apiKey = (ngcApiKeyInput.value || '').trim();
  if (!apiKey) {
    nimStatus.textContent = 'Enter your NGC API key to continue.';
    return;
  }
  configureNimButton.disabled = true;
  nimStatus.textContent = 'Pulling NIM container and configuring service…';
  try {
    const response = await fetch(API.nimConfigure, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nim_image: nimImage, api_key: apiKey }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to configure NIM');
    }
    const payload = await response.json();
    ngcApiKeyInput.value = '';
    const message = buildStatusMessage(payload);
    nimConfigured = true;
    renderNimSummary(payload.nim, message, payload);
  } catch (err) {
    console.error(err);
    nimStatus.textContent = `Error: ${err.message}`;
  } finally {
    configureNimButton.disabled = false;
  }
}

function buildStatusMessage(payload) {
  if (!payload) {
    return 'Configured.';
  }
  if (payload.pull_status === 'ok') {
    return `Configured ${payload.nim.label}. Docker image pulled successfully.`;
  }
  if (payload.pull_status === 'skipped') {
    return `Configured ${payload.nim.label}. Docker image pull skipped: ${payload.logs?.[0] || 'Docker unavailable.'}`;
  }
  return `Configured ${payload.nim.label} with warnings.`;
}

function renderNimSummary(nim, message, payload) {
  nimStatus.textContent = message;
  if (!nim) {
    nimDetails.classList.add('hidden');
    nimConfigSummary.textContent = '';
    updateRuntimeStatus(payload?.runtime, nimConfigured, message || payload?.message, payload?.logs);
    return;
  }
  const summary = {
    id: nim.id,
    label: nim.label,
    model_name: nim.model_name,
    api_base: nim.api_base,
    image: nim.image,
    pull_status: payload?.pull_status,
  };
  if (Array.isArray(payload?.logs) && payload.logs.length) {
    summary.logs = payload.logs;
  }
  nimConfigSummary.textContent = JSON.stringify(summary, null, 2);
  nimDetails.classList.remove('hidden');
  updateRuntimeStatus(payload?.runtime, nimConfigured, message || payload?.message, payload?.logs);
}

function updateRuntimeStatus(runtime, isConfigured, message, logs) {
  if (!nimRuntimeStatus) {
    return;
  }
  const lines = [];
  if (message) {
    lines.push(message);
  }
  if (!isConfigured) {
    lines.push('Configure a NIM to enable runtime controls.');
    if (startNimButton) startNimButton.disabled = true;
    if (stopNimButton) stopNimButton.disabled = true;
    if (Array.isArray(logs) && logs.length) {
      lines.push(...logs);
    }
    nimRuntimeStatus.textContent = lines.join('\n');
    return;
  }
  if (!runtime) {
    lines.push('Runtime status unavailable.');
    if (startNimButton) startNimButton.disabled = false;
    if (stopNimButton) stopNimButton.disabled = true;
    if (Array.isArray(logs) && logs.length) {
      lines.push(...logs);
    }
    nimRuntimeStatus.textContent = lines.join('\n');
    return;
  }
  if (runtime.docker_available === false) {
    lines.push('Docker is not available on this server.');
  }
  if (runtime.running) {
    lines.push(runtime.healthy ? 'NIM container is running.' : 'NIM container is starting…');
  } else {
    lines.push('NIM container is stopped.');
  }
  if (runtime.container_id) {
    const containerIdStr = String(runtime.container_id);
    const shortId = containerIdStr.slice(0, 12);
    const truncated = containerIdStr.length > shortId.length ? '…' : '';
    lines.push(`Container ID: ${shortId}${truncated}`);
  }
  if (Array.isArray(logs) && logs.length) {
    lines.push(...logs);
  }
  const dockerMissing = runtime.docker_available === false;
  if (startNimButton) {
    startNimButton.disabled = !isConfigured || runtime.running || dockerMissing;
  }
  if (stopNimButton) {
    stopNimButton.disabled = !isConfigured || !runtime.running;
  }
  nimRuntimeStatus.textContent = lines.join('\n');
}

async function startNimContainer() {
  if (!nimConfigured) {
    updateRuntimeStatus(null, false, 'Configure a NIM before starting it.');
    return;
  }
  updateRuntimeStatus({ running: false }, true, 'Starting NIM…');
  if (startNimButton) startNimButton.disabled = true;
  if (stopNimButton) stopNimButton.disabled = true;
  try {
    const response = await fetch(API.nimStart, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to start NIM');
    }
    const payload = await response.json();
    updateRuntimeStatus(payload.runtime, nimConfigured, payload.message, payload.logs);
  } catch (err) {
    console.error(err);
    updateRuntimeStatus(null, nimConfigured, `Error: ${err.message}`);
  }
}

async function stopNimContainer() {
  updateRuntimeStatus({ running: true }, nimConfigured, 'Stopping NIM…');
  if (startNimButton) startNimButton.disabled = true;
  if (stopNimButton) stopNimButton.disabled = true;
  try {
    const response = await fetch(API.nimStop, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to stop NIM');
    }
    const payload = await response.json();
    updateRuntimeStatus(payload.runtime, nimConfigured, payload.message, payload.logs);
  } catch (err) {
    console.error(err);
    updateRuntimeStatus(null, nimConfigured, `Error: ${err.message}`);
  }
}

function configureVisualizerButton() {
  const handler = async (event) => {
    event.preventDefault();
    if (visualizeButton) {
      visualizeButton.classList.add('busy');
    }
    try {
      const response = await fetch(API.visualizerLaunch, { method: 'POST' });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || 'Failed to start visualiser');
      }
      const embedUrl = payload && payload.embed_url ? payload.embed_url : null;
      const targetUrl = payload && payload.url ? payload.url : '/visualizer/redirect';
      if (embedUrl) {
        state.visualizer.embedUrl = embedUrl;
      }
      if (targetUrl) {
        state.visualizer.targetUrl = targetUrl;
      }
      const prompt = state.visualizer.lastPrompt || (promptInput ? promptInput.value : '');
      updateVisualizerPrompt(prompt || '');
      const launchUrl = visualizeButton?.dataset.launchUrl || targetUrl;
      window.open(launchUrl, '_blank', 'noopener');
    } catch (err) {
      console.error('Visualizer launch failed:', err);
      if (visualizerStatus) {
        visualizerStatus.textContent = `Visualizer unavailable: ${err.message}`;
      }
    } finally {
      if (visualizeButton) {
        visualizeButton.classList.remove('busy');
      }
    }
  };

  if (visualizeButton) {
    visualizeButton.href = '#';
    visualizeButton.addEventListener('click', handler);
  }
  if (visualizerInlineLink) {
    visualizerInlineLink.href = '#';
    visualizerInlineLink.addEventListener('click', handler);
  }
}

async function prepareEmbeddedVisualizer() {
  if (visualizerStatus) {
    visualizerStatus.textContent = 'Preparing visualizer…';
  }
  try {
    const response = await fetch(API.visualizerLaunch, { method: 'POST' });
    let payload = {};
    try {
      payload = await response.json();
    } catch (err) {
      payload = {};
    }
    if (!response.ok) {
      const message = payload && payload.detail ? payload.detail : 'Failed to start visualiser';
      throw new Error(message);
    }
    const embedUrl = payload && typeof payload.embed_url === 'string' && payload.embed_url ? payload.embed_url : null;
    const targetUrl = payload && typeof payload.url === 'string' && payload.url ? payload.url : null;

    if (embedUrl) {
      state.visualizer.embedUrl = embedUrl;
    }
    if (targetUrl) {
      state.visualizer.targetUrl = targetUrl;
    }

    if (embedUrl && visualizerFrame) {
      visualizerFrame.src = embedUrl;
      visualizerFrame.classList.remove('hidden');
      if (visualizerStatus) {
        visualizerStatus.textContent = 'Visualizer ready.';
      }
    } else {
      if (visualizerFrame) {
        visualizerFrame.classList.add('hidden');
        visualizerFrame.removeAttribute('src');
      }
      if (visualizerStatus) {
        const message = targetUrl
          ? 'Visualizer ready. Open the dedicated visualizer page to view it.'
          : 'Visualizer ready. URL unavailable.';
        visualizerStatus.textContent = message;
      }
    }
    updateVisualizerPrompt(state.visualizer.lastPrompt || (promptInput ? promptInput.value : ''));
  } catch (err) {
    console.error('Visualizer launch failed:', err);
    if (visualizerStatus) {
      visualizerStatus.textContent = `Visualizer unavailable: ${err.message}`;
    }
    if (visualizerFrame) {
      visualizerFrame.classList.add('hidden');
      visualizerFrame.removeAttribute('src');
    }
  }
}

refreshKnowledge();
