const statusEl = document.querySelector('#visualizer-page-status');
const svgEl = document.querySelector('#full-lattice-svg');
const placeholderEl = document.querySelector('#visualizer-placeholder');
const resetEl = document.querySelector('#reset-lattice-view');

const view = {
  angle: -0.75,
  dragging: false,
  lastX: 0,
  zoom: 1,
};

let entries = [];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function escapeHTML(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function entryId(entry) {
  const raw = entry?.id ?? entry?.meta?.memory_id;
  const numeric = Number(raw);
  return Number.isFinite(numeric) ? numeric : null;
}

function squareEntries() {
  return entries.filter((entry) => {
    const meta = entry.meta || {};
    return meta.synthetic_lattice === 'square'
      && Number.isFinite(Number(meta.lattice_row))
      && Number.isFinite(Number(meta.lattice_col));
  });
}

function nodeMath(entry) {
  const meta = entry.meta || {};
  const tokenWeight = clamp(Number(entry.tokens || 0) / 120, 0, 1);
  const row = Number(meta.lattice_row || 0);
  const col = Number(meta.lattice_col || 0);
  const size = Math.max(1, Number(meta.lattice_size || 1) - 1);
  const positionWeight = clamp((row + col) / Math.max(1, size * 2), 0, 1);
  const salience = Number(entry.salience ?? meta.salience ?? (0.25 + tokenWeight * 0.45 + positionWeight * 0.3));
  const fidelity = Number(entry.fidelity ?? 1);
  const degree = Number(meta.lattice_degree ?? (meta.lattice_neighbors || []).length ?? 0);
  const height = 10
    + clamp(salience, 0, 1) * 46
    + clamp(fidelity, 0, 1) * 18
    + clamp(degree / 4, 0, 1) * 14;
  const radius = 4.8 + clamp(salience, 0, 1) * 4.2;
  return { fidelity: clamp(fidelity, 0, 1), height, radius };
}

function projectPoint(x, y, z, originX, originY) {
  const cos = Math.cos(view.angle);
  const sin = Math.sin(view.angle);
  const rx = x * cos - y * sin;
  const ry = x * sin + y * cos;
  return {
    x: originX + (rx - ry) * 38 * view.zoom,
    y: originY + (rx + ry) * 21 * view.zoom - z * view.zoom,
  };
}

function renderLattice() {
  const square = squareEntries();
  if (!svgEl || !square.length) {
    if (placeholderEl) {
      placeholderEl.hidden = false;
      placeholderEl.textContent = 'No square lattice nodes are available yet.';
    }
    if (statusEl) statusEl.textContent = 'empty';
    return;
  }

  const rows = square.map((entry) => Number(entry.meta.lattice_row));
  const cols = square.map((entry) => Number(entry.meta.lattice_col));
  const maxRow = Math.max(...rows);
  const maxCol = Math.max(...cols);
  const width = 1280;
  const height = 820;
  const originX = width / 2;
  const originY = 220;
  const byId = new Map(square.map((entry) => [entryId(entry), entry]));
  const positions = new Map();
  const lines = [];
  const columns = [];
  const nodes = [];

  for (const entry of square) {
    const id = entryId(entry);
    const meta = entry.meta || {};
    const x = Number(meta.lattice_col) - maxCol / 2;
    const y = Number(meta.lattice_row) - maxRow / 2;
    const math = nodeMath(entry);
    const floor = projectPoint(x, y, 0, originX, originY);
    const top = projectPoint(x, y, math.height, originX, originY);
    positions.set(id, { entry, floor, math, top });
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
      lines.push(`<line x1="${position.top.x.toFixed(2)}" y1="${position.top.y.toFixed(2)}" x2="${neighborPosition.top.x.toFixed(2)}" y2="${neighborPosition.top.y.toFixed(2)}"></line>`);
    }
  }

  for (const [id, position] of positions.entries()) {
    const { entry, floor, math, top } = position;
    const meta = entry.meta || {};
    const source = escapeHTML(meta.source || `node ${id}`);
    const label = escapeHTML(meta.summary || entry.summary || entry.text || source);
    columns.push(`<line x1="${floor.x.toFixed(2)}" y1="${floor.y.toFixed(2)}" x2="${top.x.toFixed(2)}" y2="${top.y.toFixed(2)}"></line>`);
    nodes.push({
      sortY: top.y,
      markup:
        `<g class="lattice-node" tabindex="0" style="--fidelity:${math.fidelity.toFixed(3)}">`
        + `<circle cx="${top.x.toFixed(2)}" cy="${top.y.toFixed(2)}" r="${math.radius.toFixed(2)}"></circle>`
        + `<title>${source}\n${label}</title>`
        + '</g>',
    });
  }
  nodes.sort((a, b) => a.sortY - b.sortY);

  const corners = [
    projectPoint(-maxCol / 2, -maxRow / 2, 0, originX, originY),
    projectPoint(maxCol / 2, -maxRow / 2, 0, originX, originY),
    projectPoint(maxCol / 2, maxRow / 2, 0, originX, originY),
    projectPoint(-maxCol / 2, maxRow / 2, 0, originX, originY),
  ];
  const plane = corners.map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' ');

  svgEl.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svgEl.innerHTML = [
    `<polygon class="lattice-plane" points="${plane}"></polygon>`,
    '<g class="lattice-columns">',
    ...columns,
    '</g>',
    '<g class="lattice-edges">',
    ...lines,
    '</g>',
    '<g class="lattice-nodes">',
    ...nodes.map((node) => node.markup),
    '</g>',
  ].join('');
  if (placeholderEl) placeholderEl.hidden = true;
  if (statusEl) statusEl.textContent = `${square.length} nodes`;
}

function resetView() {
  view.angle = -0.75;
  view.zoom = 1;
  renderLattice();
}

function setupControls() {
  if (!svgEl) return;
  svgEl.addEventListener('pointerdown', (event) => {
    view.dragging = true;
    view.lastX = event.clientX;
    svgEl.setPointerCapture?.(event.pointerId);
  });
  svgEl.addEventListener('pointermove', (event) => {
    if (!view.dragging) return;
    const deltaX = event.clientX - view.lastX;
    view.lastX = event.clientX;
    view.angle += deltaX * 0.008;
    renderLattice();
  });
  svgEl.addEventListener('pointerup', (event) => {
    view.dragging = false;
    svgEl.releasePointerCapture?.(event.pointerId);
  });
  svgEl.addEventListener('pointerleave', () => {
    view.dragging = false;
  });
  svgEl.addEventListener('wheel', (event) => {
    event.preventDefault();
    view.zoom = clamp(view.zoom * (event.deltaY < 0 ? 1.08 : 0.92), 0.55, 2.1);
    renderLattice();
  }, { passive: false });
  svgEl.addEventListener('dblclick', resetView);
  resetEl?.addEventListener('click', resetView);
}

async function initialiseStandaloneVisualizer() {
  if (statusEl) statusEl.textContent = 'loading';
  try {
    const payload = await fetch('/knowledge').then((response) => response.json());
    entries = payload?.dml?.entries || [];
    renderLattice();
  } catch (err) {
    console.error('Lattice load failed:', err);
    if (statusEl) statusEl.textContent = 'unavailable';
    if (placeholderEl) {
      placeholderEl.hidden = false;
      placeholderEl.textContent = err.message;
    }
  }
}

setupControls();
initialiseStandaloneVisualizer();
