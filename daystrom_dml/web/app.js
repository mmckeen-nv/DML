const uploadForm = document.querySelector('#upload-form');
const fileInput = document.querySelector('#file-input');
const uploadStatus = document.querySelector('#upload-status');
const promptInput = document.querySelector('#prompt');
const topKInput = document.querySelector('#top-k');
const maxTokensInput = document.querySelector('#max-tokens');
const runCompareButton = document.querySelector('#run-compare');
const compareStatus = document.querySelector('#compare-status');
const resultsPanel = document.querySelector('#results');
const promptStats = document.querySelector('#prompt-stats');
const retrievalStats = document.querySelector('#retrieval-stats');
const baseOutput = document.querySelector('#base-output');
const ragOutput = document.querySelector('#rag-output');
const entriesTable = document.querySelector('#entries-table tbody');
const nimImageInput = document.querySelector('#nim-image');
const ngcApiKeyInput = document.querySelector('#ngc-api-key');
const configureNimButton = document.querySelector('#configure-nim');
const nimStatus = document.querySelector('#nim-status');
const nimDetails = document.querySelector('#nim-details');
const nimConfigSummary = document.querySelector('#nim-config-summary');

const API = {
  upload: '/upload',
  compare: '/rag/compare',
  nimOptions: '/nim/options',
  nimConfigure: '/nim/configure',
};

if (nimImageInput && configureNimButton && nimStatus) {
  loadNimStatus();
  configureNimButton.addEventListener('click', configureNimEndpoint);
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
    uploadStatus.textContent = `Ingested ${payload.chunks} chunk(s) (~${payload.tokens} tokens).`;
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
  compareStatus.textContent = 'Generating…';
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
  promptStats.textContent = JSON.stringify(
    {
      prompt_tokens_est: payload.prompt_tokens_est,
      base_usage: payload.base.usage,
      rag_usage: payload.rag.usage,
    },
    null,
    2,
  );
  retrievalStats.textContent = JSON.stringify(
    {
      avg_fidelity: payload.rag.avg_fidelity,
      context_tokens: payload.rag.context_tokens,
      entry_count: payload.rag.entries.length,
    },
    null,
    2,
  );
  baseOutput.textContent = payload.base.response || '';
  ragOutput.textContent = payload.rag.response || '';
  entriesTable.innerHTML = '';
  payload.rag.entries.forEach((entry) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${entry.id}</td>
      <td>L${entry.level}</td>
      <td>${entry.fidelity.toFixed(2)}</td>
      <td>${entry.tokens}</td>
      <td>${escapeHtml(entry.summary)}</td>
    `;
    entriesTable.appendChild(row);
  });
}

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

async function loadNimStatus() {
  nimStatus.textContent = 'Checking NVIDIA NIM configuration…';
  try {
    const response = await fetch(API.nimOptions);
    if (!response.ok) {
      throw new Error('Failed to load NIM status');
    }
    const payload = await response.json();
    if (payload.current) {
      renderNimSummary(payload.current, 'Using previously configured NIM.');
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
}
