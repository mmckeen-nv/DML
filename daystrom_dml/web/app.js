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
const tokenDmtMetric = document.querySelector('#metric-dmt-tokens');
const tokenCombinedMetric = document.querySelector('#metric-combined-tokens');
const tokenDeltaMetric = document.querySelector('#metric-token-delta');
const dmtFidelityMetric = document.querySelector('#metric-dmt-fidelity');
const ragDocsMetric = document.querySelector('#metric-rag-docs');
const dmtEntriesMetric = document.querySelector('#metric-dmt-entries');
const baseOutput = document.querySelector('#base-output');
const ragOutput = document.querySelector('#rag-output');
const dmtOutput = document.querySelector('#dmt-output');
const combinedOutput = document.querySelector('#combined-output');
const baseUsage = document.querySelector('#base-usage');
const ragUsage = document.querySelector('#rag-usage');
const dmtUsage = document.querySelector('#dmt-usage');
const combinedUsage = document.querySelector('#combined-usage');
const ragContext = document.querySelector('#rag-context');
const dmtContext = document.querySelector('#dmt-context');
const insightCopy = document.querySelector('#insight-copy');
const ragDocumentsTable = document.querySelector('#rag-documents tbody');
const dmtEntriesTable = document.querySelector('#dmt-entries tbody');
const nimImageInput = document.querySelector('#nim-image');
const ngcApiKeyInput = document.querySelector('#ngc-api-key');
const configureNimButton = document.querySelector('#configure-nim');
const nimStatus = document.querySelector('#nim-status');
const nimDetails = document.querySelector('#nim-details');
const nimConfigSummary = document.querySelector('#nim-config-summary');
const startNimButton = document.querySelector('#start-nim');
const stopNimButton = document.querySelector('#stop-nim');
const nimRuntimeStatus = document.querySelector('#nim-runtime-status');

const API = {
  upload: '/upload',
  compare: '/rag/compare',
  nimOptions: '/nim/options',
  nimConfigure: '/nim/configure',
  nimStart: '/nim/start',
  nimStop: '/nim/stop',
};

let nimConfigured = false;

if (nimImageInput && configureNimButton && nimStatus) {
  loadNimStatus();
  configureNimButton.addEventListener('click', configureNimEndpoint);
}

if (startNimButton && stopNimButton) {
  startNimButton.disabled = true;
  stopNimButton.disabled = true;
  startNimButton.addEventListener('click', startNimContainer);
  stopNimButton.addEventListener('click', stopNimContainer);
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
    uploadStatus.textContent = `Ingested ${payload.chunks} chunk(s) (~${payload.tokens} tokens) into RAG and the DMT.`;
    fileInput.value = '';
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
  const ragTokens = payload.rag?.context_tokens ?? 0;
  const dmtTokens = payload.dmt?.context_tokens ?? 0;
  const combinedTokens = payload.combined?.context_tokens ?? ragTokens + dmtTokens;
  setMetricValue(tokenPromptMetric, promptTokens);
  setMetricValue(tokenRagMetric, ragTokens);
  setMetricValue(tokenDmtMetric, dmtTokens);
  setMetricValue(tokenCombinedMetric, combinedTokens);
  const tokenDelta = dmtTokens && ragTokens ? ragTokens - dmtTokens : 0;
  tokenDeltaMetric.textContent = tokenDelta
    ? `${tokenDelta > 0 ? '−' : '+'}${nf.format(Math.abs(tokenDelta))} tokens`
    : '0';
  dmtFidelityMetric.textContent = formatFloat(payload.dmt?.avg_fidelity);
  ragDocsMetric.textContent = nf.format(payload.rag?.documents?.length ?? 0);
  dmtEntriesMetric.textContent = nf.format(payload.dmt?.entries?.length ?? 0);

  baseOutput.textContent = payload.base?.response || '';
  ragOutput.textContent = payload.rag?.response || '';
  dmtOutput.textContent = payload.dmt?.response || '';
  combinedOutput.textContent = payload.combined?.response || '';

  baseUsage.textContent = formatUsage(payload.base?.usage);
  ragUsage.textContent = formatUsage(payload.rag?.usage);
  dmtUsage.textContent = formatUsage(payload.dmt?.usage);
  combinedUsage.textContent = formatUsage(payload.combined?.usage);

  ragContext.textContent = payload.rag?.context || 'No RAG context retrieved for this prompt.';
  dmtContext.textContent = payload.dmt?.context || 'No DMT memories matched this prompt yet.';
  renderRagDocuments(payload.rag?.documents || []);
  renderDmtEntries(payload.dmt?.entries || []);

  insightCopy.textContent = buildInsightCopy({ promptTokens, ragTokens, dmtTokens, tokenDelta, avgFidelity: payload.dmt?.avg_fidelity, ragCount: payload.rag?.documents?.length || 0, dmtCount: payload.dmt?.entries?.length || 0 });
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
    return 'Usage data unavailable from backend.';
  }
  const nf = new Intl.NumberFormat('en-US');
  const prompt = usage.prompt_tokens ?? usage.promptTokens;
  const completion = usage.completion_tokens ?? usage.completionTokens;
  const total = usage.total_tokens ?? usage.totalTokens;
  const pieces = [];
  if (prompt !== undefined) pieces.push(`Prompt: ${nf.format(prompt)}`);
  if (completion !== undefined) pieces.push(`Completion: ${nf.format(completion)}`);
  if (total !== undefined) pieces.push(`Total: ${nf.format(total)}`);
  return pieces.length ? pieces.join(' | ') : 'Usage data unavailable from backend.';
}

function renderRagDocuments(documents) {
  ragDocumentsTable.innerHTML = '';
  if (!documents.length) {
    const emptyRow = document.createElement('tr');
    emptyRow.innerHTML = '<td colspan="4">No matching RAG documents ingested yet.</td>';
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

function renderDmtEntries(entries) {
  dmtEntriesTable.innerHTML = '';
  if (!entries.length) {
    const emptyRow = document.createElement('tr');
    emptyRow.innerHTML = '<td colspan="5">No DMT memories retrieved.</td>';
    dmtEntriesTable.appendChild(emptyRow);
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
    dmtEntriesTable.appendChild(row);
  });
}

function buildInsightCopy({ promptTokens, ragTokens, dmtTokens, tokenDelta, avgFidelity, ragCount, dmtCount }) {
  if (!ragTokens && !dmtTokens) {
    return 'No retrieval context has been generated yet. Upload documents to populate RAG and the DMT.';
  }
  const nf = new Intl.NumberFormat('en-US');
  const parts = [];
  if (ragCount) {
    parts.push(`RAG contributed ${nf.format(ragCount)} document chunk${ragCount === 1 ? '' : 's'} totalling ${nf.format(ragTokens)} tokens.`);
  }
  if (dmtCount) {
    const fidelityText = avgFidelity !== undefined && avgFidelity !== null ? ` with an average fidelity of ${Number(avgFidelity).toFixed(2)}` : '';
    parts.push(`The Daystrom Memory Lattice surfaced ${nf.format(dmtCount)} memory node${dmtCount === 1 ? '' : 's'}${fidelityText} and ${nf.format(dmtTokens)} contextual tokens.`);
  }
  if (tokenDelta) {
    const direction = tokenDelta > 0 ? 'fewer' : 'more';
    parts.push(`Compared to RAG alone, the DMT context uses ${nf.format(Math.abs(tokenDelta))} ${direction} tokens, highlighting the compression gains of the lattice.`);
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
  if (!nimImage) {
    nimStatus.textContent = 'Input a NVIDIA NIM image first.';
    return;
  }
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
