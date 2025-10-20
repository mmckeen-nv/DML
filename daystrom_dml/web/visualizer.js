const statusEl = document.querySelector('#visualizer-page-status');
const frameEl = document.querySelector('#visualizer-fullscreen-frame');
const openLinkEl = document.querySelector('#visualizer-open-streamlit');

async function initialiseStandaloneVisualizer() {
  if (!statusEl || !frameEl) {
    return;
  }
  statusEl.textContent = 'Preparing visualizer…';
  try {
    const response = await fetch('/visualizer/launch', { method: 'POST' });
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
    if (payload && typeof payload.url === 'string' && payload.url) {
      frameEl.src = payload.url;
      statusEl.textContent = 'Visualizer ready.';
      if (openLinkEl) {
        openLinkEl.href = payload.url;
      }
    } else {
      statusEl.textContent = 'Visualizer ready (no URL available).';
    }
  } catch (err) {
    console.error('Visualizer launch failed:', err);
    statusEl.textContent = `Visualizer unavailable: ${err.message}`;
    frameEl.removeAttribute('src');
  }
}

initialiseStandaloneVisualizer();
