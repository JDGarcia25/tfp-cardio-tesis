/**
 * CardioAI — Frontend Application
 * API base: http://localhost:8000
 */
'use strict';

const API_BASE = 'http://localhost:8000';

// Estado global
let currentBeatData = null;
let currentBeatLabel = 'normal';
let selectedFile   = null;
let csvFileOk      = false;

// ── Utils ──────────────────────────────────────────────────────────────────────
function toast(msg, type = 'info', duration = 3500) {
  const el = document.createElement('div');
  el.className = `toast toast--${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), duration);
}

function showLoader(id)  { document.getElementById(id).classList.add('loading-bar--visible'); }
function hideLoader(id)  { document.getElementById(id).classList.remove('loading-bar--visible'); }

async function apiFetch(path, options = {}) {
  const resp = await fetch(`${API_BASE}${path}`, options);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(body.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  fetchModelInfo();
  setupManualPanel();
  setupCsvPanel();
  setupMonitor();
  setInterval(checkHealth, 30_000);
});

// ── Health ─────────────────────────────────────────────────────────────────────
async function checkHealth() {
  const dot   = document.getElementById('statusDot');
  const label = document.getElementById('statusLabel');
  dot.className = 'status-dot status-dot--loading';
  label.textContent = 'Conectando...';
  try {
    const data = await apiFetch('/health');
    dot.className   = 'status-dot status-dot--ok';
    label.textContent = `En línea · ${data.model_name}`;
  } catch {
    dot.className   = 'status-dot status-dot--error';
    label.textContent = 'Sin conexión';
  }
}

// ── Model info ─────────────────────────────────────────────────────────────────
async function fetchModelInfo() {
  try {
    const info = await apiFetch('/model-info');

    document.getElementById('badgeModel').innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
        <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
      </svg>
      ${info.model_name}`;

    document.getElementById('badgeType').textContent  = `Tipo: ${info.model_type}`;
    document.getElementById('badgeThreshold').textContent =
      info.threshold != null
        ? `Umbral: ${info.threshold.toFixed(5)}`
        : `Representación: ${info.representation}`;

    // Marcar el mejor modelo con ★ en los selects ya poblados
    ['modelSelectManual', 'modelSelect'].forEach(id => {
      const sel = document.getElementById(id);
      Array.from(sel.options).forEach(opt => {
        if (opt.value === info.model_name) {
          opt.textContent = opt.textContent.replace(' ★', '') + ' ★';
        }
      });
    });

    // Footer
    document.getElementById('footerModelInfo').innerHTML = [
      `<li><code>GET /health</code></li>`,
      `<li><code>GET /model-info</code></li>`,
      `<li><code>POST /predict</code></li>`,
      `<li><code>POST /predict-csv</code></li>`,
    ].join('');

    // Monitor threshold display
    document.getElementById('monitorThreshold').textContent =
      info.threshold != null ? info.threshold.toFixed(4) : '—';

  } catch (err) {
    console.warn('Model info unavailable:', err.message);
  }
}

// ── Panel manual ───────────────────────────────────────────────────────────────
function setupManualPanel() {
  const textarea   = document.getElementById('beatInput');
  const countHint  = document.getElementById('beatCountHint');
  const btnPredict = document.getElementById('btnPredict');
  const btnSample  = document.getElementById('btnSample');

  textarea.addEventListener('input', () => {
    const values = parseBeats(textarea.value);
    const n = values.length;
    countHint.textContent = `${n} / 200 valores`;
    countHint.className = 'field__hint' + (
      n === 0   ? '' :
      n === 200 ? ' field__hint--ok' : ' field__hint--error'
    );
    btnPredict.disabled = n !== 200;
    if (n === 200) currentBeatData = new Float32Array(values);
  });

  btnSample.addEventListener('click', () => {
    const beat = generateSyntheticBeat('normal');
    currentBeatData = beat;
    currentBeatLabel = 'normal';
    textarea.value = JSON.stringify(Array.from(beat));
    textarea.dispatchEvent(new Event('input'));
    drawEcgCanvas(beat, 'normal');
    toast('Beat normal sintético cargado', 'info');
  });

  btnPredict.addEventListener('click', predictManual);
}

async function predictManual() {
  const textarea  = document.getElementById('beatInput');
  const values    = parseBeats(textarea.value);
  if (values.length !== 200) {
    toast('Se requieren exactamente 200 valores float', 'error');
    return;
  }

  const preprocessed = getPreprocessedValue();
  const modelName    = document.getElementById('modelSelectManual').value;
  const url          = modelName
    ? `/predict?model_name=${encodeURIComponent(modelName)}`
    : '/predict';

  showLoader('spinnerManual');
  document.getElementById('resultManual').innerHTML = '';

  try {
    const result = await apiFetch(url, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ beat: values, preprocessed }),
    });
    renderManualResult(result);
    if (currentBeatData) drawEcgCanvas(currentBeatData, result.label);
    updateMonitorReadings(result);
  } catch (err) {
    toast(`Error: ${err.message}`, 'error');
  } finally {
    hideLoader('spinnerManual');
  }
}

function renderManualResult(r) {
  const isAnomaly = r.prediction === 1;
  const icon      = isAnomaly ? '⚠' : '♥';
  const alertInfo = getAlertLevel(r);

  let scorePercent = 0;
  if (r.reconstruction_error != null && r.threshold != null) {
    scorePercent = Math.min(100, Math.round((r.reconstruction_error / (r.threshold * 1.5)) * 100));
  }

  const alertRatio = r.reconstruction_error != null && r.threshold != null && r.threshold > 0
    ? (r.reconstruction_error / r.threshold).toFixed(2) : '—';

  document.getElementById('resultManual').innerHTML = `
    <div class="result-block">
      <div class="result-block__top result-block__top--${isAnomaly ? 'anomaly' : 'normal'}">
        <div class="result-block__icon">${icon}</div>
        <div class="result-block__verdict">
          <span class="result-block__label">
            ${isAnomaly ? 'Anomalía Cardíaca Detectada' : 'Ritmo Cardíaco Normal'}
          </span>
          <span class="result-block__sub">
            Modelo: ${r.model_name} · Normalización: ${r.normalization_applied ? 'aplicada' : 'no aplicada'}
          </span>
        </div>
      </div>
      <div class="result-block__metrics">
        <div class="result-metric">
          <span class="result-metric__val">${r.prediction}</span>
          <span class="result-metric__lbl">Predicción</span>
        </div>
        <div class="result-metric">
          <span class="result-metric__val">${r.reconstruction_error != null ? r.reconstruction_error.toFixed(5) : '—'}</span>
          <span class="result-metric__lbl">Error MSE</span>
        </div>
        <div class="result-metric">
          <span class="result-metric__val">${r.threshold != null ? r.threshold.toFixed(5) : '—'}</span>
          <span class="result-metric__lbl">Umbral</span>
        </div>
        <div class="result-metric">
          <span class="result-metric__val">${alertInfo.icon}</span>
          <span class="result-metric__lbl">${alertInfo.label}</span>
        </div>
      </div>
      <div class="alert-bar alert-bar--${alertInfo.level}">
        <span class="alert-bar__light alert-bar__light--${alertInfo.level}">${alertInfo.icon === '🟢' ? '✓' : '!'}</span>
        <span class="alert-bar__label alert-bar__label--${alertInfo.level}">${alertInfo.label}</span>
        <span class="alert-bar__desc">
          ${alertRatio !== '—' ? `Error × <span class="alert-bar__ratio">${alertRatio}</span> umbral` : ''}
        </span>
      </div>
      ${r.reconstruction_error != null ? `
      <div class="mse-bar">
        <div class="mse-bar__header">
          <span>Error de reconstrucción</span>
          <span>${r.threshold ? ((r.reconstruction_error / r.threshold) * 100).toFixed(1) : 0}% del umbral</span>
        </div>
        <div class="mse-bar__track">
          <div class="mse-bar__fill mse-bar__fill--${isAnomaly ? 'bad' : 'ok'}"
               style="width:${scorePercent}%"></div>
        </div>
      </div>` : ''}
    </div>
  `;
}

// ── Panel CSV ──────────────────────────────────────────────────────────────────
function setupCsvPanel() {
  const dropzone    = document.getElementById('dropzone');
  const fileInput   = document.getElementById('csvFileInput');
  const btnUpload   = document.getElementById('btnCsvUpload');
  const hint        = document.getElementById('dropzoneHint');
  const slider      = document.getElementById('sensitivityRange');
  const sliderLabel = document.getElementById('sensitivityValue');

  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    sliderLabel.textContent = `${v.toFixed(1)}×`;
    sliderLabel.style.background = v > 1.5 ? '#ffcdd2' : v > 1.0 ? '#fff9c4' : '';
    sliderLabel.style.color      = v > 1.5 ? '#c62828'  : v > 1.0 ? '#f57f17'  : '';
  });

  // Toggle adaptativo: mostrar/ocultar slider de sensibilidad
  const adaptiveToggle = document.getElementById('adaptiveThreshold');
  const sensitivityRow = document.getElementById('sensitivityRow');
  const adaptiveLabel  = document.getElementById('adaptiveLabel');
  const adaptiveHint   = document.getElementById('adaptiveHint');
  function updateAdaptiveUI() {
    const on = adaptiveToggle.checked;
    sensitivityRow.style.display = on ? 'none' : 'block';
    adaptiveLabel.textContent    = on ? 'Adaptativo (p85)' : 'Fijo (entrenamiento)';
    adaptiveHint.textContent     = on
      ? 'Umbral calculado desde el propio registro'
      : 'Usa el umbral del modelo entrenado (puede no detectar nada)';
  }
  adaptiveToggle.addEventListener('change', updateAdaptiveUI);
  updateAdaptiveUI();

  dropzone.addEventListener('click',   () => fileInput.click());
  dropzone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dropzone--active'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dropzone--active'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dropzone--active');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });
  btnUpload.addEventListener('click', analyzeCsv);

  function handleFile(file) {
    if (!file.name.endsWith('.csv')) { toast('Solo archivos .csv', 'error'); return; }
    selectedFile = file; csvFileOk = true;
    dropzone.classList.add('dropzone--has-file');
    hint.textContent = `✓ ${file.name}  (${(file.size/1024).toFixed(1)} KB)`;
    btnUpload.disabled = false;
  }
}

async function analyzeCsv() {
  if (!csvFileOk || !selectedFile) { toast('Selecciona un CSV primero', 'error'); return; }

  const model    = document.getElementById('modelSelect').value;
  const adaptive = document.getElementById('adaptiveThreshold').checked;
  const sensitivity = document.getElementById('sensitivityRange').value;

  const params = new URLSearchParams({ adaptive_threshold: adaptive });
  if (!adaptive) params.set('sensitivity', sensitivity);
  if (model) params.set('model_name', model);

  const formData = new FormData();
  formData.append('file', selectedFile);

  showLoader('spinnerCsv');
  document.getElementById('resultCsvSummary').innerHTML = '';
  document.getElementById('resultMseStats').style.display = 'none';
  document.getElementById('resultCsvPlot').style.display  = 'none';

  try {
    const resp = await fetch(`${API_BASE}/predict-csv?${params}`, { method: 'POST', body: formData });
    if (!resp.ok) {
      const b = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(b.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderCsvResult(data);
    if (data.mse_stats) renderMseStats(data);
    toast(`${data.total_beats} latidos · ${data.anomalias} anomalías · ${data.r_peaks_detected} picos R`, 'success');
  } catch (err) {
    toast(`Error: ${err.message}`, 'error');
  } finally {
    hideLoader('spinnerCsv');
  }
}

function renderCsvResult(data) {
  const pct = data.total_beats > 0
    ? ((data.anomalias / data.total_beats) * 100).toFixed(1) : '0.0';

  let csvAlert;
  const pctNum = parseFloat(pct);
  if (pctNum >= 15)      csvAlert = { level: 'critico', label: 'Riesgo Alto', icon: '🔴' };
  else if (pctNum >= 8)  csvAlert = { level: 'moderado', label: 'Riesgo Moderado', icon: '🟠' };
  else if (pctNum >= 3)  csvAlert = { level: 'leve', label: 'Riesgo Leve', icon: '🟡' };
  else                   csvAlert = { level: 'normal', label: 'Sin Riesgo', icon: '🟢' };

  document.getElementById('resultCsvSummary').innerHTML = `
    <div class="csv-summary">
      <div class="csv-stat csv-stat--total">
        <span class="csv-stat__num csv-stat__num--blue">${data.total_beats}</span>
        <span class="csv-stat__label">Total latidos</span>
      </div>
      <div class="csv-stat csv-stat--normal">
        <span class="csv-stat__num csv-stat__num--green">${data.normales}</span>
        <span class="csv-stat__label">Normales</span>
      </div>
      <div class="csv-stat csv-stat--anomaly">
        <span class="csv-stat__num csv-stat__num--red">${data.anomalias}</span>
        <span class="csv-stat__label">Anomalías</span>
      </div>
      <div class="csv-stat csv-stat--model">
        <span class="csv-stat__num csv-stat__num--gray">${pct}%</span>
        <span class="csv-stat__label">${data.model_used}</span>
      </div>
    </div>
    <div class="alert-bar alert-bar--${csvAlert.level}" style="margin-top:10px;">
      <span class="alert-bar__light alert-bar__light--${csvAlert.level}">${csvAlert.icon === '🟢' ? '✓' : '!'}</span>
      <span class="alert-bar__label alert-bar__label--${csvAlert.level}">
        ${csvAlert.label} · ${pct}% anomalías
      </span>
      <span class="alert-bar__desc">
        ${data.total_beats} latidos · ${data.normales} normales · ${data.anomalias} anomalías
      </span>
    </div>
  `;

  if (data.csv_plot_b64) {
    const src = `data:image/png;base64,${data.csv_plot_b64}`;
    document.getElementById('csvPlotImg').src = src;
    document.getElementById('csvPlotDownload').href = src;
    document.getElementById('resultCsvPlot').style.display = 'block';
  }
}

function renderMseStats(data) {
  const s      = data.mse_stats;
  const thr    = data.threshold_used;
  const thrFix = data.threshold_training;
  const isAdaptive = data.adaptive_threshold;
  const method = data.threshold_method || '';

  const barMax  = Math.max(s.max, thr || s.max, thrFix || 0) * 1.1;
  const fillPct = Math.min(100, Math.round((s.p90 / barMax) * 100));
  const thrPct  = thr     ? Math.min(100, Math.round((thr    / barMax) * 100)) : null;
  const fixPct  = thrFix  ? Math.min(100, Math.round((thrFix / barMax) * 100)) : null;

  let hint;
  if (isAdaptive) {
    const thrLabel = method.replace('adaptativo_', '');
    hint = `✓ Umbral adaptativo (${thrLabel}): los ${data.anomalias} latidos con mayor error de reconstrucción fueron marcados como anomalías.`;
  } else {
    const noAnom = data.anomalias === 0 && thr != null;
    if (noAnom) {
      const sug = Math.ceil((s.max / thr + 0.1) * 10) / 10;
      hint = `⚠ Ningún latido superó el umbral fijo (${thr.toFixed(5)}). Máx. MSE: ${s.max.toFixed(5)}. Activa "Umbral adaptativo" o prueba sensibilidad ≥ ${sug}.`;
    } else {
      hint = `✓ ${data.anomalias} latidos superaron el umbral fijo (${thr?.toFixed(5) ?? '—'}).`;
    }
  }

  const methodBadge = isAdaptive
    ? `<span class="method-badge method-badge--green">Adaptativo · ${method.replace('adaptativo_', 'P')}</span>`
    : `<span class="method-badge method-badge--gray">Fijo · entrenamiento · ${data.sensitivity}×</span>`;

  document.getElementById('resultMseStats').innerHTML = `
    <div class="mse-panel">
      <div class="mse-panel__title">
        📊 Diagnóstico MSE — Error de reconstrucción
        ${methodBadge}
      </div>
      <div class="mse-grid">
        <div class="mse-cell"><span class="mse-cell__val">${s.min.toFixed(5)}</span><span class="mse-cell__lbl">Mínimo</span></div>
        <div class="mse-cell"><span class="mse-cell__val">${s.mean.toFixed(5)}</span><span class="mse-cell__lbl">Media</span></div>
        <div class="mse-cell"><span class="mse-cell__val">${s.p90.toFixed(5)}</span><span class="mse-cell__lbl">P90</span></div>
        <div class="mse-cell"><span class="mse-cell__val">${s.max.toFixed(5)}</span><span class="mse-cell__lbl">Máximo</span></div>
      </div>
      ${thr != null ? `
      <div class="mse-threshold-bar">
        <div class="mse-threshold-bar__label">
          <span>Distribución MSE (P90 vs umbral aplicado)</span>
          <span>Umbral aplicado: <strong>${thr.toFixed(5)}</strong>${thrFix && !isAdaptive ? ` · Entrenamiento: ${thrFix.toFixed(5)}` : ''}</span>
        </div>
        <div class="mse-threshold-bar__track">
          <div class="mse-threshold-bar__fill" style="width:${fillPct}%"></div>
          ${thrPct != null ? `<div class="mse-threshold-bar__marker" style="left:${thrPct}%"
            title="Umbral aplicado: ${thr.toFixed(5)}"></div>` : ''}
          ${fixPct != null && isAdaptive ? `<div class="mse-threshold-bar__marker mse-threshold-bar__marker--gray" style="left:${fixPct}%"
            title="Umbral entrenamiento: ${thrFix.toFixed(5)}"></div>` : ''}
        </div>
        ${isAdaptive && thrFix ? `<div class="mse-compare">
          Umbral entrenamiento: <code>${thrFix.toFixed(5)}</code> — Umbral adaptativo: <code>${thr.toFixed(5)}</code>
          (${thr < thrFix ? '▼ más sensible' : '▲ menos sensible'} que el umbral de entrenamiento)
        </div>` : ''}
      </div>` : ''}
      <p class="mse-hint">${hint}</p>
    </div>
  `;
  document.getElementById('resultMseStats').style.display = 'block';
}

// ── Monitor ECG ────────────────────────────────────────────────────────────────
function setupMonitor() {
  document.getElementById('btnGenNormal').addEventListener('click', () => {
    const b = generateSyntheticBeat('normal');
    currentBeatData  = b;
    currentBeatLabel = 'normal';
    drawEcgCanvas(b, 'normal');
    loadBeatToTextarea(b);
    toast('Ritmo normal cargado', 'info');
  });

  document.getElementById('btnGenAnomaly').addEventListener('click', () => {
    const b = generateSyntheticBeat('anomaly');
    currentBeatData  = b;
    currentBeatLabel = 'anomaly';
    drawEcgCanvas(b, 'anomaly');
    loadBeatToTextarea(b);
    toast('Fibrilación ventricular cargada', 'info');
  });

  document.getElementById('btnSendToAnalyze').addEventListener('click', () => {
    if (!currentBeatData) { toast('Genera un beat primero', 'info'); return; }
    loadBeatToTextarea(currentBeatData);
    document.getElementById('panelManual').scrollIntoView({ behavior: 'smooth' });
    document.getElementById('btnPredict').click();
  });

  // Hero canvas animation
  setupHeroCanvas();

  // Inicializar con beat normal
  const init = generateSyntheticBeat('normal');
  currentBeatData  = init;
  currentBeatLabel = 'normal';
  drawEcgCanvas(init, 'normal');
  loadBeatToTextarea(init);
}

function loadBeatToTextarea(beat) {
  const ta = document.getElementById('beatInput');
  ta.value = JSON.stringify(Array.from(beat));
  ta.dispatchEvent(new Event('input'));
}

function updateMonitorReadings(result) {
  const lbl = document.getElementById('monitorLabel');
  const mse = document.getElementById('monitorMse');
  const statusEl = document.getElementById('monitorStatus');
  const icon = statusEl.querySelector('.monitor-reading__icon');

  lbl.textContent = result.label === 'normal' ? 'Normal' : 'Anomalía';
  mse.textContent = result.reconstruction_error != null
    ? result.reconstruction_error.toFixed(5) : '—';

  if (result.label === 'normal') {
    icon.className = 'monitor-reading__icon monitor-reading__icon--green';
    icon.textContent = '♥';
  } else {
    icon.className = 'monitor-reading__icon monitor-reading__icon--red';
    icon.textContent = '⚠';
  }

  updateMonitorAlert(result);
}

// ── ECG Canvas ─────────────────────────────────────────────────────────────────
function drawEcgCanvas(beat, label) {
  const canvas = document.getElementById('ecgCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = rect.width, H = rect.height;
  const pad = { top: 16, bottom: 16, left: 12, right: 12 };
  const dW = W - pad.left - pad.right;
  const dH = H - pad.top  - pad.bottom;

  // Fondo
  ctx.fillStyle = '#080e18';
  ctx.fillRect(0, 0, W, H);

  // Grilla
  ctx.strokeStyle = 'rgba(255,255,255,.04)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 8; i++) {
    const x = pad.left + (i / 8) * dW;
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, H - pad.bottom); ctx.stroke();
  }
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * dH;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
  }

  // Línea cero
  ctx.strokeStyle = 'rgba(255,255,255,.08)';
  const zeroY = pad.top + dH * 0.5;
  ctx.beginPath(); ctx.moveTo(pad.left, zeroY); ctx.lineTo(W - pad.right, zeroY); ctx.stroke();

  // Normalizar beat
  const min = Math.min(...beat), max = Math.max(...beat), range = max - min || 1;

  const color = label === 'anomaly' ? '#ef5350' : '#66bb6a';

  // Sombra glow
  ctx.shadowColor = color;
  ctx.shadowBlur  = 8;

  // Señal ECG
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.lineJoin    = 'round';
  ctx.lineCap     = 'round';

  beat.forEach((v, i) => {
    const x = pad.left + (i / (beat.length - 1)) * dW;
    const y = pad.top  + (1 - (v - min) / range) * dH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Pico R
  const rX = pad.left + (90 / (beat.length - 1)) * dW;
  const rY = pad.top  + (1 - (beat[90] - min) / range) * dH;
  ctx.beginPath();
  ctx.arc(rX, rY, 4, 0, Math.PI * 2);
  ctx.fillStyle = '#ef5350';
  ctx.shadowColor = '#ef5350';
  ctx.shadowBlur = 6;
  ctx.fill();
  ctx.shadowBlur = 0;

  // Label top-right
  ctx.font = `600 11px 'Inter', sans-serif`;
  ctx.textAlign = 'right';
  ctx.fillStyle = label === 'anomaly' ? '#ef9a9a' : '#a5d6a7';
  ctx.fillText(label === 'anomaly' ? '⚠ ANOMALÍA' : '✓ NORMAL', W - pad.right, pad.top + 14);
}

// ── Hero mini canvas (animación continua) ──────────────────────────────────────
function setupHeroCanvas() {
  const canvas = document.getElementById('heroCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let offset = 0;
  let bpm    = 72;

  function frame() {
    const dpr  = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width  = rect.width  * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width, H = rect.height;

    ctx.fillStyle = '#0a0f1a';
    ctx.fillRect(0, 0, W, H);

    // Grid tenue
    ctx.strokeStyle = 'rgba(255,255,255,.03)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 8; i++) {
      ctx.beginPath();
      ctx.moveTo((i / 8) * W, 0);
      ctx.lineTo((i / 8) * W, H);
      ctx.stroke();
    }

    // ECG scrolling
    const pts = 200;
    ctx.beginPath();
    ctx.strokeStyle = '#4caf50';
    ctx.lineWidth = 1.5;
    ctx.shadowColor = '#4caf50';
    ctx.shadowBlur = 6;
    ctx.lineJoin = 'round';

    for (let i = 0; i < pts; i++) {
      const t   = (i + offset) % pts;
      const v   = ecgWaveform(t, pts);
      const x   = (i / (pts - 1)) * W;
      const y   = H * 0.5 - v * H * 0.38;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    offset = (offset + 1) % pts;

    // Calcular BPM simulado (fluctúa ligeramente)
    bpm = 72 + Math.sin(offset / 30) * 3;
    document.getElementById('heroBpm').textContent = Math.round(bpm);

    requestAnimationFrame(frame);
  }
  frame();
}

function ecgWaveform(t, total) {
  const norm = t / total;
  const cycle = norm % 1;
  // PQRST simplificado
  const p  = 0.12 * Math.exp(-Math.pow((cycle - 0.20) / 0.04, 2));
  const q  = -0.08 * Math.exp(-Math.pow((cycle - 0.38) / 0.015, 2));
  const r  = 0.9  * Math.exp(-Math.pow((cycle - 0.42) / 0.015, 2));
  const s  = -0.15 * Math.exp(-Math.pow((cycle - 0.46) / 0.015, 2));
  const tt = 0.18 * Math.exp(-Math.pow((cycle - 0.60) / 0.06, 2));
  return p + q + r + s + tt;
}

// ── Signal synthesis ───────────────────────────────────────────────────────────
function generateSyntheticBeat(type = 'normal') {
  const N = 200, out = new Float32Array(N);

  if (type === 'normal') {
    for (let t = 0; t < N; t++) {
      out[t] = (
        0.25 * Math.exp(-Math.pow(t - 60,  2) / (2 * 64))  +
       -0.30 * Math.exp(-Math.pow(t - 85,  2) / (2 * 9))   +
        1.50 * Math.exp(-Math.pow(t - 90,  2) / (2 * 16))  +
       -0.40 * Math.exp(-Math.pow(t - 97,  2) / (2 * 9))   +
        0.40 * Math.exp(-Math.pow(t - 140, 2) / (2 * 225)) +
        (Math.random() - 0.5) * 0.04
      );
    }
  } else {
    for (let t = 0; t < N; t++) {
      out[t] = (
        0.5 * Math.sin(2 * Math.PI * 5  * t / N) +
        0.3 * Math.sin(2 * Math.PI * 13 * t / N) +
        0.4 * Math.sin(2 * Math.PI * 7  * t / N) +
        (Math.random() - 0.5) * 0.45
      );
    }
  }

  // Z-score
  const mean = out.reduce((a, b) => a + b, 0) / N;
  const std  = Math.sqrt(out.reduce((a, b) => a + (b - mean) ** 2, 0) / N) || 1;
  for (let i = 0; i < N; i++) out[i] = (out[i] - mean) / std;
  return out;
}

// ── Clinical Alert Levels ──────────────────────────────────────────────────────
function getAlertLevel(result) {
  const isAnomaly = result.prediction === 1;
  if (!isAnomaly) return { level: 'normal', label: 'Normal', icon: '🟢' };

  const error = result.reconstruction_error;
  const threshold = result.threshold;
  if (error == null || threshold == null || threshold === 0) return { level: 'leve', label: 'Alerta Leve', icon: '🟡' };

  const ratio = error / threshold;
  if (ratio >= 3.0) return { level: 'critico', label: 'Crítico', icon: '🔴' };
  if (ratio >= 2.0) return { level: 'moderado', label: 'Alerta Moderada', icon: '🟠' };
  return { level: 'leve', label: 'Alerta Leve', icon: '🟡' };
}

function renderTrafficLight(level) {
  const levels = ['normal', 'leve', 'moderado', 'critico'];
  return levels.map(l => {
    const active = l === level ? ' traffic-light__lamp--active' : '';
    return `<span class="traffic-light__lamp${active} traffic-light__lamp--${l}"></span>`;
  }).join('');
}

function updateMonitorAlert(result) {
  const alertInfo = getAlertLevel(result);
  const alertEl = document.getElementById('monitorAlert');
  const trafficEl = document.getElementById('trafficLight');
  if (alertEl) alertEl.textContent = alertInfo.icon + ' ' + alertInfo.label;
  if (trafficEl) trafficEl.innerHTML = renderTrafficLight(alertInfo.level);
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function parseBeats(raw) {
  try {
    const str = raw.trim();
    const arr = JSON.parse(str.startsWith('[') ? str : `[${str}]`);
    if (!Array.isArray(arr)) return [];
    return arr.map(Number).filter(n => !isNaN(n));
  } catch { return []; }
}

function getPreprocessedValue() {
  const sel = document.querySelector('input[name="preprocessed"]:checked');
  if (!sel) return null;
  if (sel.value === 'true')  return true;
  if (sel.value === 'false') return false;
  return null;
}

window.addEventListener('resize', () => {
  if (currentBeatData) drawEcgCanvas(currentBeatData, currentBeatLabel);
});
