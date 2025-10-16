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
const tokenCombinedMetric = document.querySelector('#metric-combined-tokens');
const tokenDeltaMetric = document.querySelector('#metric-token-delta');
const dmlFidelityMetric = document.querySelector('#metric-dml-fidelity');
const ragDocsMetric = document.querySelector('#metric-rag-docs');
const dmlEntriesMetric = document.querySelector('#metric-dml-entries');
const baseOutput = document.querySelector('#base-output');
const ragOutput = document.querySelector('#rag-output');
const dmlOutput = document.querySelector('#dml-output');
const combinedOutput = document.querySelector('#combined-output');
const baseUsage = document.querySelector('#base-usage');
const ragUsage = document.querySelector('#rag-usage');
const dmlUsage = document.querySelector('#dml-usage');
const combinedUsage = document.querySelector('#combined-usage');
const ragContext = document.querySelector('#rag-context');
const dmlContext = document.querySelector('#dml-context');
const dmlSummaryList = document.querySelector('#dml-summary-list');
const dmlContextLlmOutput = document.querySelector('#dml-context-llm-output');
const combinedContextLlmOutput = document.querySelector('#combined-context-llm-output');
const insightCopy = document.querySelector('#insight-copy');
const ragDocumentsTable = document.querySelector('#rag-documents tbody');
const dmlEntriesTable = document.querySelector('#dml-entries tbody');
const nimModelInput = document.querySelector('#nim-model');
const ngcApiKeyInput = document.querySelector('#ngc-api-key');
const configureNimButton = document.querySelector('#configure-nim');
const testNimHealthButton = document.querySelector('#test-nim-health');
const nimStatus = document.querySelector('#nim-status');
const nimHealthStatus = document.querySelector('#nim-health-status');
const nimDetails = document.querySelector('#nim-details');
const nimConfigSummary = document.querySelector('#nim-config-summary');

const API = {
  upload: '/upload',
  compare: '/rag/compare',
  nimOptions: '/nim/options',
  nimConfigure: '/nim/configure',
  nimHealth: '/nim/health',
};

let nimConfigured = false;

if (nimModelInput && configureNimButton && nimStatus) {
  loadNimStatus();
  configureNimButton.addEventListener('click', configureNimEndpoint);
}

if (testNimHealthButton) {
  testNimHealthButton.addEventListener('click', testNimHealth);
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
    const message = err.message === 'NIM not Running, Start a NIM.'
      ? err.message
      : `Error: ${err.message}`;
    compareStatus.textContent = message;
  }
});

function renderResults(payload) {
  resultsPanel.classList.remove('hidden');
  const nf = new Intl.NumberFormat('en-US');
  const promptTokens = payload.prompt_tokens_est ?? 0;
  const ragTokens = payload.rag?.context_tokens ?? 0;
  const dmlTokens = payload.dml?.context_tokens ?? 0;
  const combinedTokens = payload.combined?.context_tokens ?? ragTokens + dmlTokens;
  setMetricValue(tokenPromptMetric, promptTokens);
  setMetricValue(tokenRagMetric, ragTokens);
  setMetricValue(tokenDmlMetric, dmlTokens);
  setMetricValue(tokenCombinedMetric, combinedTokens);
  const tokenDelta = dmlTokens && ragTokens ? ragTokens - dmlTokens : 0;
  tokenDeltaMetric.textContent = tokenDelta
    ? `${tokenDelta > 0 ? '−' : '+'}${nf.format(Math.abs(tokenDelta))} tokens`
    : '0';
  dmlFidelityMetric.textContent = formatFloat(payload.dml?.avg_fidelity);
  ragDocsMetric.textContent = nf.format(payload.rag?.documents?.length ?? 0);
  dmlEntriesMetric.textContent = nf.format(payload.dml?.entries?.length ?? 0);

  baseOutput.textContent = payload.base?.response || '';
  ragOutput.textContent = payload.rag?.response || '';
  dmlOutput.textContent = payload.dml?.response || '';
  combinedOutput.textContent = payload.combined?.response || '';

  baseUsage.textContent = formatUsage(payload.base?.usage);
  ragUsage.textContent = formatUsage(payload.rag?.usage);
  dmlUsage.textContent = formatUsage(payload.dml?.usage);
  combinedUsage.textContent = formatUsage(payload.combined?.usage);

  ragContext.textContent = payload.rag?.context || 'No RAG context retrieved for this prompt.';
  dmlContext.textContent = payload.dml?.context || 'No DML memories matched this prompt yet.';
  renderRagDocuments(payload.rag?.documents || []);
  renderDmlEntries(payload.dml?.entries || []);
  renderDmlSummaries(payload.dml?.entries || []);

  if (dmlContextLlmOutput) {
    dmlContextLlmOutput.textContent = payload.dml?.response || 'No DML response generated yet.';
  }
  if (combinedContextLlmOutput) {
    combinedContextLlmOutput.textContent =
      payload.combined?.response || 'No combined response generated yet.';
  }

  insightCopy.textContent = buildInsightCopy({ promptTokens, ragTokens, dmlTokens, tokenDelta, avgFidelity: payload.dml?.avg_fidelity, ragCount: payload.rag?.documents?.length || 0, dmlCount: payload.dml?.entries?.length || 0 });
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

function renderDmlEntries(entries) {
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
    parts.push(`Compared to RAG alone, the DML context uses ${nf.format(Math.abs(tokenDelta))} ${direction} tokens, highlighting the compression gains of the lattice.`);
  }
  parts.push(`The user prompt spans approximately ${nf.format(promptTokens)} tokens.`);
  return parts.join(' ');
}


async function loadNimStatus() {
  nimStatus.textContent = 'Checking NVIDIA NIM configuration…';
  if (nimHealthStatus) {
    nimHealthStatus.textContent = '';
  }
  if (testNimHealthButton) {
    testNimHealthButton.disabled = true;
  }
  try {
    const response = await fetch(API.nimOptions);
    if (!response.ok) {
      throw new Error('Failed to load NIM status');
    }
    const payload = await response.json();
    nimConfigured = Boolean(payload.current);
    if (testNimHealthButton) {
      testNimHealthButton.disabled = !nimConfigured;
    }
    if (payload.current) {
      if (nimModelInput) {
        nimModelInput.value = payload.current.model_name || '';
      }
      renderNimSummary(payload.current, 'Using configured NIM connection.');
    } else {
      nimConfigured = false;
      clearNimSummary('Enter your NIM model name and NGC API key to begin.');
    }
  } catch (err) {
    console.error(err);
    nimConfigured = false;
    if (testNimHealthButton) {
      testNimHealthButton.disabled = true;
    }
    nimStatus.textContent = `Error: ${err.message}`;
  }
}

async function configureNimEndpoint() {
  const modelName = nimModelInput ? nimModelInput.value.trim() : '';
  const apiKey = ngcApiKeyInput ? ngcApiKeyInput.value.trim() : '';
  if (!modelName) {
    nimStatus.textContent = 'Enter the NIM model name you are running.';
    return;
  }
  if (!apiKey) {
    nimStatus.textContent = 'Enter your NGC API key to continue.';
    return;
  }
  configureNimButton.disabled = true;
  if (testNimHealthButton) {
    testNimHealthButton.disabled = true;
  }
  nimStatus.textContent = 'Configuring connection…';
  try {
    const response = await fetch(API.nimConfigure, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName, api_key: apiKey }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to configure NIM');
    }
    const payload = await response.json();
    nimConfigured = true;
    if (testNimHealthButton) {
      testNimHealthButton.disabled = false;
    }
    if (nimModelInput) {
      nimModelInput.value = payload.nim?.model_name || modelName;
    }
    if (nimHealthStatus) {
      nimHealthStatus.textContent = '';
    }
    if (ngcApiKeyInput) {
      ngcApiKeyInput.value = '';
    }
    renderNimSummary(payload.nim, payload.message || 'Configured connection to user-managed NIM.');
  } catch (err) {
    console.error(err);
    nimConfigured = false;
    if (testNimHealthButton) {
      testNimHealthButton.disabled = true;
    }
    nimStatus.textContent = `Error: ${err.message}`;
  } finally {
    configureNimButton.disabled = false;
  }
}

function renderNimSummary(nim, message) {
  if (message) {
    nimStatus.textContent = message;
  }
  if (!nim) {
    clearNimSummary(message || 'No NIM configured.');
    return;
  }
  if (nimDetails) {
    nimDetails.classList.remove('hidden');
  }
  const summary = {
    model_name: nim.model_name,
    api_base: nim.api_base,
  };
  if (nimConfigSummary) {
    nimConfigSummary.textContent = JSON.stringify(summary, null, 2);
  }
}

function clearNimSummary(message) {
  if (nimDetails) {
    nimDetails.classList.add('hidden');
  }
  if (nimConfigSummary) {
    nimConfigSummary.textContent = '';
  }
  if (message) {
    nimStatus.textContent = message;
  }
}

async function testNimHealth() {
  if (!nimConfigured) {
    if (nimHealthStatus) {
      nimHealthStatus.textContent = 'Configure a NIM connection before testing health.';
    }
    return;
  }
  if (nimHealthStatus) {
    nimHealthStatus.textContent = 'Testing NIM health…';
  }
  if (testNimHealthButton) {
    testNimHealthButton.disabled = true;
  }
  try {
    const response = await fetch(API.nimHealth, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      const detail = err.detail;
      let message = 'Health check failed. Please check the NIM.';
      if (typeof detail === 'string') {
        message = detail;
      } else if (detail && typeof detail === 'object') {
        if (detail.message) {
          message = detail.message;
        }
        if (Array.isArray(detail.attempts) && detail.attempts.length) {
          const lastAttempt = detail.attempts[detail.attempts.length - 1];
          if (lastAttempt) {
            message += ` (Last error: ${lastAttempt})`;
          }
        }
      }
      throw new Error(message);
    }
    const payload = await response.json();
    if (nimHealthStatus) {
      nimHealthStatus.textContent = payload.message || 'NIM is healthy.';
    }
  } catch (err) {
    console.error(err);
    if (nimHealthStatus) {
      nimHealthStatus.textContent = `Error: ${err.message}`;
    }
  } finally {
    if (testNimHealthButton) {
      testNimHealthButton.disabled = !nimConfigured;
    }
  }
}
