const state = {
  baseUrl: window.location.origin,
  tenant: "openclaw",
  session: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function jsonHeaders() {
  return {"content-type": "application/json"};
}

function addMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text || "(empty)";
  $("#chat-log").appendChild(node);
  node.scrollIntoView({block: "end"});
}

function resultItem(result) {
  const node = document.createElement("article");
  node.className = "result-item";
  const title = document.createElement("strong");
  title.textContent = result.title || `memory:${result.id}`;
  const snippet = document.createElement("p");
  snippet.textContent = result.snippet || result.text || "";
  const meta = document.createElement("code");
  meta.textContent = JSON.stringify(result.metadata || result.meta || {}, null, 2);
  node.append(title, snippet, meta);
  return node;
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function refresh() {
  const [health, tags] = await Promise.all([request("/health"), request("/api/tags")]);
  const stats = health.stats || {};
  const model = (tags.models || [])[0] || {};
  $("#status-dot").classList.toggle("ok", health.status === "ok");
  $("#status-line").textContent = health.status === "ok" ? "running" : health.status;
  $("#model-name").textContent = model.name || "daystrom-dml:memory";
  $("#memory-count").textContent = stats.count ?? 0;
  $("#uptime").textContent = `${Math.round(health.uptime_seconds || 0)}s`;
  $("#base-url").textContent = state.baseUrl;
  $("#setup-base-url").textContent = state.baseUrl;
}

function showView(id) {
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === id));
  $$(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.view === id));
  $("#view-title").textContent = id === "chat-view" ? "Chat" : id === "memory-view" ? "Memory" : "Setup";
}

async function sendChat(event) {
  event.preventDefault();
  const prompt = $("#chat-prompt").value.trim();
  if (!prompt) return;
  state.tenant = $("#tenant").value || "openclaw";
  state.session = $("#session").value || null;
  addMessage("user", prompt);
  $("#chat-prompt").value = "";
  const payload = {
    model: "daystrom-dml:memory",
    messages: [{role: "user", content: prompt}],
    tenant_id: state.tenant,
    session_id: state.session,
    stream: false,
  };
  try {
    const result = await request("/api/chat", {
      method: "POST",
      headers: jsonHeaders(),
      body: JSON.stringify(payload),
    });
    addMessage("assistant", result.message?.content || "");
  } catch (error) {
    addMessage("assistant", `Request failed: ${error.message}`);
  }
}

async function remember(event) {
  event.preventDefault();
  const text = $("#memory-text").value.trim();
  if (!text) return;
  const payload = {
    text,
    tenant_id: $("#tenant").value || "openclaw",
    session_id: $("#session").value || null,
    kind: $("#kind").value || "note",
    meta: {source: $("#source").value || "provider-ui"},
  };
  const result = await request("/api/remember", {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  });
  $("#memory-results").replaceChildren(resultItem({title: "stored", snippet: JSON.stringify(result), metadata: payload.meta}));
  $("#memory-text").value = "";
  await refresh();
}

async function search(event) {
  event.preventDefault();
  const query = $("#search-query").value.trim();
  if (!query) return;
  const params = new URLSearchParams({
    q: query,
    tenant_id: $("#tenant").value || "openclaw",
    top_k: $("#topk").value || "6",
  });
  const session = $("#session").value;
  if (session) params.set("session_id", session);
  const result = await request(`/api/search?${params.toString()}`);
  const items = (result.results || []).map(resultItem);
  $("#memory-results").replaceChildren(...(items.length ? items : [resultItem({title: "no results", snippet: "No memories matched.", metadata: {query}})]));
}

async function copyText(text) {
  await navigator.clipboard.writeText(text);
}

function wireEvents() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
  $("#refresh").addEventListener("click", refresh);
  $("#copy-base").addEventListener("click", () => copyText(state.baseUrl));
  $("#chat-form").addEventListener("submit", sendChat);
  $("#remember-form").addEventListener("submit", remember);
  $("#search-form").addEventListener("submit", search);
  $$(".copy").forEach((button) => {
    button.addEventListener("click", () => {
      const targetId = button.dataset.copyTarget;
      const text = targetId ? $(`#${targetId}`).textContent : button.previousElementSibling?.textContent;
      copyText(text || "");
    });
  });
}

wireEvents();
refresh()
  .then(() => addMessage("assistant", "DML is running. Ask a question and I will return retrieved memory context."))
  .catch((error) => {
    $("#status-line").textContent = `offline: ${error.message}`;
    addMessage("assistant", `Provider check failed: ${error.message}`);
  });
