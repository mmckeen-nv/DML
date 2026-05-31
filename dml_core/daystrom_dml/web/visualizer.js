const statusEl = document.querySelector('#visualizer-page-status');
const svgEl = document.querySelector('#full-lattice-svg');
const placeholderEl = document.querySelector('#visualizer-placeholder');
const resetEl = document.querySelector('#reset-lattice-view');

const view = {
  angle: -0.75,
  dragMode: 'pan',
  dragging: false,
  lastX: 0,
  lastY: 0,
  panX: 0,
  panY: 0,
  zoom: 1,
};

let entries = [];
let activeNodeIds = new Set();

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
  if (raw === undefined || raw === null || raw === '') return null;
  const numeric = Number(raw);
  return Number.isFinite(numeric) ? String(numeric) : String(raw);
}

function firstFinite(...values) {
  for (const value of values) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return numeric;
  }
  return null;
}

function normalizeIdSet(values) {
  return new Set((values || [])
    .map((value) => {
      if (value === undefined || value === null || value === '') return null;
      const numeric = Number(value);
      return Number.isFinite(numeric) ? String(numeric) : String(value);
    })
    .filter(Boolean));
}

function mergeActivatedEntries(baseEntries, activatedEntries) {
  const merged = [...baseEntries];
  const seen = new Set(merged.map(entryId).filter(Boolean));
  for (const entry of activatedEntries || []) {
    const id = entryId(entry);
    if (!id || seen.has(id)) continue;
    merged.push(entry);
    seen.add(id);
  }
  return merged;
}

function groupKey(entry) {
  const meta = entry.meta || {};
  return String(meta.cluster || meta.topic || meta.source || meta.kind || 'memory');
}

function visualEntries() {
  const gridEntries = entries.filter((entry) => {
    const meta = entry.meta || {};
    return Number.isFinite(Number(meta.lattice_row)) && Number.isFinite(Number(meta.lattice_col));
  });
  const explicitLayers = entries
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
  const fallbackWidth = Math.max(8, maxGridCol + 1, Math.ceil(Math.sqrt(Math.max(1, entries.length))));
  const groups = new Map();
  let fallbackIndex = 0;

  return entries.map((entry) => {
    const meta = entry.meta || {};
    const row = Number(meta.lattice_row);
    const col = Number(meta.lattice_col);
    const hasGridPosition = Number.isFinite(row) && Number.isFinite(col);
    const key = groupKey(entry);
    if (!groups.has(key)) groups.set(key, groups.size);
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
      entry,
      id: entryId(entry),
      row: derivedRow,
      col: derivedCol,
      layer: clamp(layer, 0, 7),
    };
  });
}

function nodeMath(entry, active = false) {
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
    + clamp(degree / 4, 0, 1) * 14
    + (active ? 24 : 0);
  const radius = 4.8 + clamp(salience, 0, 1) * 4.2 + (active ? 1.8 : 0);
  return { fidelity: clamp(fidelity, 0, 1), height, radius };
}

function projectPoint(x, y, z, originX, originY) {
  const cos = Math.cos(view.angle);
  const sin = Math.sin(view.angle);
  const rx = x * cos - y * sin;
  const ry = x * sin + y * cos;
  return {
    x: originX + view.panX + (rx - ry) * 38 * view.zoom,
    y: originY + view.panY + (rx + ry) * 21 * view.zoom - z * view.zoom,
  };
}

function viewBoxForPoints(points, targetWidth, targetHeight, padding = 64) {
  const xs = points.map((point) => point.x).filter(Number.isFinite);
  const ys = points.map((point) => point.y).filter(Number.isFinite);
  if (!xs.length || !ys.length) return `0 0 ${targetWidth} ${targetHeight}`;
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
  return `${minX.toFixed(2)} ${minY.toFixed(2)} ${width.toFixed(2)} ${height.toFixed(2)}`;
}

function renderLattice() {
  const nodesForView = visualEntries();
  if (!svgEl || !nodesForView.length) {
    if (placeholderEl) {
      placeholderEl.hidden = false;
      placeholderEl.textContent = 'No lattice nodes are available yet.';
    }
    if (statusEl) statusEl.textContent = 'empty';
    return;
  }

  const rows = nodesForView.map((node) => node.row);
  const cols = nodesForView.map((node) => node.col);
  const maxRow = Math.max(...rows);
  const minRow = Math.min(...rows);
  const maxCol = Math.max(...cols);
  const minCol = Math.min(...cols);
  const maxLayer = Math.max(...nodesForView.map((node) => node.layer));
  const layerGap = 34;
  const width = 1280;
  const height = 820;
  const originX = width / 2;
  const originY = 220;
  const byId = new Map(nodesForView.map((node) => [node.id, node]));
  const positions = new Map();
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
    layerPlanes.push(`<polygon class="lattice-plane layer-plane" points="${points}"></polygon>`);
  }

  for (const node of nodesForView) {
    const { entry, id, layer } = node;
    const x = node.col - maxCol / 2;
    const y = node.row - maxRow / 2;
    const active = activeNodeIds.has(id);
    const math = nodeMath(entry, active);
    const baseZ = layer * layerGap;
    const floor = projectPoint(x, y, baseZ, originX, originY);
    const top = projectPoint(x, y, baseZ + math.height, originX, originY);
    projectedPoints.push(floor, top);
    positions.set(id, { active, entry, floor, layer, math, top });
  }

  for (const node of nodesForView) {
    const { entry, id } = node;
    const meta = entry.meta || {};
    const position = positions.get(id);
    if (!position) continue;
    for (const neighbor of meta.lattice_neighbors || []) {
      const neighborId = entryId({ id: neighbor });
      const neighborEntry = byId.get(neighborId);
      if (!neighborEntry || String(neighborId) < String(id)) continue;
      const neighborPosition = positions.get(neighborId);
      if (!neighborPosition) continue;
      const active = activeNodeIds.has(id) && activeNodeIds.has(neighborId);
      lines.push(`<line class="${active ? 'active' : ''}" x1="${position.top.x.toFixed(2)}" y1="${position.top.y.toFixed(2)}" x2="${neighborPosition.top.x.toFixed(2)}" y2="${neighborPosition.top.y.toFixed(2)}"></line>`);
    }
  }

  for (const [id, position] of positions.entries()) {
    const { active, entry, floor, layer, math, top } = position;
    const meta = entry.meta || {};
    const source = escapeHTML(meta.source || `node ${id}`);
    const label = escapeHTML(meta.summary || entry.summary || entry.text || source);
    columns.push(`<line class="${active ? 'active' : ''}" x1="${floor.x.toFixed(2)}" y1="${floor.y.toFixed(2)}" x2="${top.x.toFixed(2)}" y2="${top.y.toFixed(2)}"></line>`);
    nodes.push({
      sortY: top.y,
      markup:
        `<g class="lattice-node ${active ? 'active' : ''}" tabindex="0" style="--fidelity:${math.fidelity.toFixed(3)}">`
        + `<circle cx="${top.x.toFixed(2)}" cy="${top.y.toFixed(2)}" r="${math.radius.toFixed(2)}"></circle>`
        + `<title>${source}\nLayer ${layer}\n${label}</title>`
        + '</g>',
    });
  }
  nodes.sort((a, b) => a.sortY - b.sortY);

  const axes = [
    ['x', projectPoint(minCol - maxCol / 2, maxRow - maxRow / 2 + 0.7, 0, originX, originY), projectPoint(maxCol - maxCol / 2, maxRow - maxRow / 2 + 0.7, 0, originX, originY)],
    ['y', projectPoint(maxCol - maxCol / 2 + 0.7, minRow - maxRow / 2, 0, originX, originY), projectPoint(maxCol - maxCol / 2 + 0.7, maxRow - maxRow / 2, 0, originX, originY)],
    ['z', projectPoint(maxCol - maxCol / 2 + 0.9, maxRow - maxRow / 2 + 0.9, 0, originX, originY), projectPoint(maxCol - maxCol / 2 + 0.9, maxRow - maxRow / 2 + 0.9, maxLayer * layerGap + 110, originX, originY)],
  ];
  projectedPoints.push(...axes.flatMap(([, start, end]) => [start, end]));
  const axisMarkup = axes.map(
    ([name, start, end]) =>
      `<line class="axis ${name}" x1="${start.x.toFixed(2)}" y1="${start.y.toFixed(2)}" x2="${end.x.toFixed(2)}" y2="${end.y.toFixed(2)}"></line>`
  );

  svgEl.setAttribute('viewBox', viewBoxForPoints(projectedPoints, width, height));
  svgEl.innerHTML = [
    ...layerPlanes,
    '<g class="lattice-axes">',
    ...axisMarkup,
    '</g>',
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
  if (statusEl) {
    const activeCount = nodesForView.filter((node) => activeNodeIds.has(node.id)).length;
    statusEl.textContent = activeCount
      ? `${activeCount} active / ${nodesForView.length} nodes`
      : `${nodesForView.length} nodes · ${maxLayer + 1} layers`;
  }
}

function resetView() {
  view.angle = -0.75;
  view.panX = 0;
  view.panY = 0;
  view.zoom = 1;
  renderLattice();
}

function setupControls() {
  if (!svgEl) return;

  const pointerDeltaToViewBox = (deltaX, deltaY) => {
    const rect = svgEl.getBoundingClientRect();
    const viewBox = svgEl.viewBox?.baseVal;
    if (!rect.width || !rect.height || !viewBox) {
      return { x: deltaX, y: deltaY };
    }
    return {
      x: deltaX * (viewBox.width / rect.width),
      y: deltaY * (viewBox.height / rect.height),
    };
  };

  const beginDrag = (event) => {
    view.dragging = true;
    view.dragMode = event.shiftKey ? 'rotate' : 'pan';
    view.lastX = event.clientX;
    view.lastY = event.clientY;
    svgEl.classList.add('dragging');
  };
  const moveDrag = (event) => {
    if (!view.dragging) return;
    const deltaX = event.clientX - view.lastX;
    const deltaY = event.clientY - view.lastY;
    view.lastX = event.clientX;
    view.lastY = event.clientY;
    if (view.dragMode === 'rotate') {
      view.angle += deltaX * 0.008;
      renderLattice();
      return;
    }
    const pan = pointerDeltaToViewBox(deltaX, deltaY);
    view.panX += pan.x;
    view.panY += pan.y;
    renderLattice();
  };
  const endDrag = () => {
    view.dragging = false;
    svgEl.classList.remove('dragging');
  };

  svgEl.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    beginDrag(event);
    svgEl.setPointerCapture?.(event.pointerId);
  });
  window.addEventListener('pointermove', moveDrag);
  window.addEventListener('pointerup', endDrag);
  window.addEventListener('pointercancel', endDrag);
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
    const [knowledgePayload, statePayload] = await Promise.all([
      fetch('/knowledge').then((response) => response.json()),
      fetch('/visualizer/state').then((response) => response.json()).catch(() => null),
    ]);
    const metadata = statePayload?.payload?.metadata || {};
    const activatedNodes = metadata.activated_nodes || metadata.activated_entries || [];
    entries = mergeActivatedEntries(knowledgePayload?.dml?.entries || [], activatedNodes);
    activeNodeIds = normalizeIdSet(
      metadata.activated_node_ids
      || metadata.activated_ids
      || metadata.active_node_ids
      || metadata.dml_node_ids
      || activatedNodes.map(entryId)
      || []
    );
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
