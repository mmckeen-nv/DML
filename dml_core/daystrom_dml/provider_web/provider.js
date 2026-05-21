const statusLine = document.querySelector("#status-line");
const metrics = document.querySelector("#metrics");
const results = document.querySelector("#results");

function showJson(value) {
  results.textContent = JSON.stringify(value, null, 2);
}

async function refresh() {
  const response = await fetch("/health");
  const payload = await response.json();
  statusLine.textContent = `${payload.provider} is ${payload.status}`;
  const stats = payload.stats || {};
  metrics.innerHTML = `
    <article class="metric"><span>Memories</span><strong>${stats.count ?? 0}</strong></article>
    <article class="metric"><span>Storage</span><strong>${stats.storage_dir ? "Ready" : "Local"}</strong></article>
    <article class="metric"><span>Uptime</span><strong>${Math.round(payload.uptime_seconds || 0)}s</strong></article>
  `;
}

document.querySelector("#refresh").addEventListener("click", refresh);

document.querySelector("#recall-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    query: document.querySelector("#query").value,
    tenant_id: document.querySelector("#tenant").value || "openclaw",
    session_id: document.querySelector("#session").value || null,
    top_k: Number(document.querySelector("#topk").value || 6),
  };
  const response = await fetch("/api/recall", {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify(payload),
  });
  showJson(await response.json());
});

document.querySelector("#remember-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    text: document.querySelector("#memory-text").value,
    tenant_id: document.querySelector("#tenant").value || "openclaw",
    session_id: document.querySelector("#session").value || null,
    kind: document.querySelector("#kind").value || "note",
    meta: {source: document.querySelector("#source").value || "provider-ui"},
  };
  const response = await fetch("/api/remember", {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify(payload),
  });
  showJson(await response.json());
  await refresh();
});

refresh().catch((error) => {
  statusLine.textContent = `Provider check failed: ${error}`;
});
