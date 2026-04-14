/**
 * Main entry point: wire UI elements to the inference worker and Mol* viewer.
 */

import './style.css';
import { initViewer, loadStructure } from './molstar';
import type { InitMessage, RunMessage, WorkerResponse } from './inference-worker';

// Example FASTA for 3owz_A (L=86)
const EXAMPLE_FASTA = `>3owzA
GGCUCUGGAGAGAACCGUUUAAUCGGUCGCCGAAGGAGCAAGCUCUGCGGAAACGCAGAGUGAAACUCUCAGGCAAAAGGACAGAGUC`;

const MODELS_BASE_URL = 'https://r2.brighthong.com/v0';
const MODEL_CACHE_NAME = 'rhofold-models-v0';

const MODEL_DEFS = [
  { name: 'RNA-FM',          file: 'rna_fm.onnx',          size: 378 },
  { name: 'Embedder',        file: 'embedder.onnx',        size: 5 },
  { name: 'E2Eformer',       file: 'e2eformer.onnx',       size: 86 },
  { name: 'Structure+Heads', file: 'structure_heads.onnx',  size: 11 },
  { name: 'RefineNet',       file: 'refinenet.onnx',        size: 5 },
];

// DOM elements
const fastaInput = document.getElementById('fasta-input') as HTMLTextAreaElement;
const msaInput = document.getElementById('msa-input') as HTMLInputElement;
const msaFilename = document.getElementById('msa-filename') as HTMLSpanElement;
const recyclesSlider = document.getElementById('recycles-slider') as HTMLInputElement;
const recyclesValue = document.getElementById('recycles-value') as HTMLElement;
const msaDepthSelect = document.getElementById('msa-depth') as HTMLSelectElement;
const backendSelect = document.getElementById('backend-select') as HTMLSelectElement;
const btnExample = document.getElementById('btn-example') as HTMLButtonElement;
const btnPredict = document.getElementById('btn-predict') as HTMLButtonElement;
const statusText = document.getElementById('status-text') as HTMLSpanElement;
const statusBar = document.getElementById('status-bar') as HTMLDivElement;
const progressSection = document.getElementById('progress-section') as HTMLDivElement;
const progressBar = document.getElementById('progress-bar') as HTMLDivElement;
const progressText = document.getElementById('progress-text') as HTMLSpanElement;
const plddtSection = document.getElementById('plddt-section') as HTMLDivElement;
const plddtChart = document.getElementById('plddt-chart') as HTMLCanvasElement;
const plddtGlobalEl = document.getElementById('plddt-global') as HTMLParagraphElement;
const downloadSection = document.getElementById('download-section') as HTMLDivElement;
const btnDownloadPdb = document.getElementById('btn-download-pdb') as HTMLButtonElement;
const btnDownloadPlddt = document.getElementById('btn-download-plddt') as HTMLButtonElement;
const viewerPlaceholder = document.getElementById('viewer-placeholder') as HTMLDivElement;
const viewerContainer = document.getElementById('molstar-viewer') as HTMLDivElement;
const modelsBadge = document.getElementById('models-badge') as HTMLSpanElement;
const btnDownloadModels = document.getElementById('btn-download-models') as HTMLButtonElement;
const btnClearCache = document.getElementById('btn-clear-cache') as HTMLButtonElement;

// State
let worker: Worker | null = null;
let modelsReady = false;
let running = false;
let resultPdb = '';
let resultPlddt: number[] = [];
let resultSequence = '';
let msaFileContent: string | null = null;
let modelsDownloading = false;

// --- WebGPU detection ---
function hasWebGPU(): boolean {
  return !!(navigator as any).gpu;
}

// --- Models status (compact) ---
async function checkModelCache() {
  let cachedCount = 0;
  try {
    const cache = await caches.open(MODEL_CACHE_NAME);
    for (const m of MODEL_DEFS) {
      if (await cache.match(`${MODELS_BASE_URL}/${m.file}`)) cachedCount++;
    }
  } catch { /* no cache API */ }

  if (cachedCount === MODEL_DEFS.length) {
    modelsBadge.textContent = '(cached)';
    modelsBadge.className = 'models-badge cached';
    btnDownloadModels.style.display = 'none';
    btnClearCache.style.display = '';
  } else {
    modelsBadge.textContent = '(not cached)';
    modelsBadge.className = 'models-badge';
    btnDownloadModels.style.display = '';
    btnClearCache.style.display = 'none';
  }
}

async function predownloadModels() {
  if (modelsDownloading) return;
  modelsDownloading = true;
  btnDownloadModels.disabled = true;
  btnDownloadModels.textContent = 'Downloading...';
  modelsBadge.textContent = '';
  modelsBadge.className = 'models-badge downloading';

  const w = ensureWorker();
  w.postMessage({ type: 'init', backend: backendSelect.value } as InitMessage);
}

async function clearModelCache() {
  await caches.delete(MODEL_CACHE_NAME);
  modelsReady = false;
  checkModelCache();
}

// --- Init ---
function init() {
  // Auto-detect WebGPU
  if (!hasWebGPU()) {
    backendSelect.value = 'wasm';
    const webgpuOpt = backendSelect.querySelector('option[value="webgpu"]');
    if (webgpuOpt) {
      webgpuOpt.textContent = 'WebGPU (not available)';
      (webgpuOpt as HTMLOptionElement).disabled = true;
    }
  }

  fastaInput.addEventListener('input', updatePredictButton);

  recyclesSlider.addEventListener('input', () => {
    recyclesValue.textContent = recyclesSlider.value;
  });

  msaInput.addEventListener('change', async () => {
    const file = msaInput.files?.[0];
    if (file) {
      msaFilename.textContent = file.name;
      msaFileContent = await file.text();
    } else {
      msaFilename.textContent = 'No file selected';
      msaFileContent = null;
    }
  });

  btnExample.addEventListener('click', loadExample);
  btnPredict.addEventListener('click', startPrediction);
  btnDownloadModels.addEventListener('click', predownloadModels);
  btnClearCache.addEventListener('click', clearModelCache);
  btnDownloadPdb.addEventListener('click', downloadPdb);
  btnDownloadPlddt.addEventListener('click', downloadPlddtCsv);

  updatePredictButton();
  checkModelCache();
}

function updatePredictButton() {
  btnPredict.disabled = running || !fastaInput.value.trim();
}

async function loadExample() {
  fastaInput.value = EXAMPLE_FASTA.trim();
  msaFileContent = null;
  msaFilename.textContent = 'No file selected';
  msaInput.value = '';

  try {
    const resp = await fetch('/example/3owz_A.afa');
    if (resp.ok) {
      msaFileContent = await resp.text();
      msaFilename.textContent = '3owz_A.afa (example)';
    }
  } catch {
    // MSA is optional
  }

  updatePredictButton();
  setStatus('Example loaded. Click "Predict" to start.');
}

// --- Worker ---
function ensureWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./inference-worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = handleWorkerMessage;
    worker.onerror = (e) => {
      setStatus(`Worker error: ${e.message}`, 'error');
      running = false;
      updatePredictButton();
    };
  }
  return worker;
}

function handleWorkerMessage(event: MessageEvent<WorkerResponse>) {
  const msg = event.data;

  switch (msg.type) {
    case 'status':
      setStatus(msg.message, msg.ready ? 'ready' : undefined);
      if (msg.ready) {
        modelsReady = true;
        modelsDownloading = false;
        modelsBadge.textContent = '(cached)';
        modelsBadge.className = 'models-badge cached';
        btnDownloadModels.style.display = 'none';
        btnClearCache.style.display = '';
      }
      break;

    case 'progress':
      showProgress(msg.stage, msg.recycle, msg.totalRecycles);
      break;

    case 'download':
      if (msg.allCached || (msg.loaded === 0 && msg.total === 0)) {
        // All done or all from cache
      } else if (msg.total > 0) {
        const pct = ((msg.loaded / msg.total) * 100).toFixed(0);
        const mb = (msg.loaded / 1024 / 1024).toFixed(0);
        const totalMb = (msg.total / 1024 / 1024).toFixed(0);
        setStatus(`Downloading models: ${mb} / ${totalMb} MB (${pct}%)`);
        btnDownloadModels.textContent = `${pct}%`;
        progressSection.hidden = false;
        progressBar.style.width = `${pct}%`;
        progressText.textContent = `${mb} / ${totalMb} MB`;
      }
      break;

    case 'result':
      handleResult(msg.pdb, msg.plddt, msg.plddtGlobal, msg.sequence);
      break;

    case 'error':
      setStatus(`Error: ${msg.message}`, 'error');
      running = false;
      progressSection.hidden = true;
      updatePredictButton();
      break;
  }
}

async function startPrediction() {
  if (running) return;
  running = true;
  updatePredictButton();

  resultPdb = '';
  resultPlddt = [];
  plddtSection.hidden = true;
  downloadSection.hidden = true;
  progressSection.hidden = false;
  progressBar.style.width = '0%';

  const w = ensureWorker();

  if (!modelsReady) {
    setStatus('Initializing models...');
    const initMsg: InitMessage = {
      type: 'init',
      backend: backendSelect.value as 'webgpu' | 'wasm',
    };
    w.postMessage(initMsg);

    await new Promise<void>((resolve) => {
      const orig = w.onmessage as (e: MessageEvent) => void;
      w.onmessage = (e: MessageEvent<WorkerResponse>) => {
        orig?.call(w, e);
        if (e.data.type === 'status' && (e.data as any).ready) {
          resolve();
        } else if (e.data.type === 'error') {
          resolve();
        }
      };
    });

    if (!modelsReady) {
      running = false;
      updatePredictButton();
      return;
    }
  }

  const runMsg: RunMessage = {
    type: 'run',
    fastaContent: fastaInput.value,
    msaContent: msaFileContent || undefined,
    nRecycles: parseInt(recyclesSlider.value, 10),
    msaDepth: parseInt(msaDepthSelect.value, 10),
  };
  w.postMessage(runMsg);
}

// --- Progress ---
function showProgress(stage: string, recycle?: number, total?: number) {
  progressSection.hidden = false;
  if (recycle && total) {
    const pct = (recycle / total) * 100;
    progressBar.style.width = `${pct}%`;
    progressText.textContent = `${stage} (${recycle}/${total})`;
  } else {
    progressText.textContent = stage;
  }
  setStatus(stage);
}

// --- Results ---
async function handleResult(pdb: string, plddt: number[], plddtGlobal: number, sequence: string) {
  running = false;
  updatePredictButton();
  progressSection.hidden = true;

  resultPdb = pdb;
  resultPlddt = plddt;
  resultSequence = sequence;

  setStatus(`Done! Mean pLDDT: ${plddtGlobal.toFixed(3)}`, 'ready');

  plddtGlobalEl.textContent = `Mean pLDDT: ${plddtGlobal.toFixed(4)}`;
  plddtSection.hidden = false;
  downloadSection.hidden = false;
  // Draw after unhiding so the canvas has a real width
  requestAnimationFrame(() => drawPlddtChart(plddt));

  try {
    viewerPlaceholder.classList.add('hidden');
    await initViewer(viewerContainer);
    await loadStructure(pdb);
  } catch (e: any) {
    console.error('Mol* error:', e);
    setStatus(`3D viewer error: ${e.message}. Structure available for download.`, 'error');
  }
}

// --- pLDDT bar chart ---
function drawPlddtChart(plddt: number[]) {
  const canvas = plddtChart;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = 100 * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = 100;
  const barW = Math.max(1, w / plddt.length);

  ctx.clearRect(0, 0, w, h);

  for (let i = 0; i < plddt.length; i++) {
    const v = plddt[i];
    const barH = v * (h - 10);
    const x = i * barW;
    const y = h - barH;
    ctx.fillStyle = plddtColor(v);
    ctx.fillRect(x, y, Math.max(1, barW - 0.5), barH);
  }

  ctx.strokeStyle = '#666';
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  const threshY = h - 0.7 * (h - 10);
  ctx.beginPath();
  ctx.moveTo(0, threshY);
  ctx.lineTo(w, threshY);
  ctx.stroke();
  ctx.setLineDash([]);
}

function plddtColor(v: number): string {
  if (v > 0.9) return '#0053d6';
  if (v > 0.7) return '#65cbf3';
  if (v > 0.5) return '#ffdb13';
  return '#ff7d45';
}

// --- Status ---
function setStatus(text: string, type?: 'error' | 'ready') {
  statusText.textContent = text;
  statusBar.className = 'status-bar';
  if (type) statusBar.classList.add(type);
}

// --- Downloads ---
function downloadPdb() {
  if (!resultPdb) return;
  downloadBlob(resultPdb, 'rhofold_prediction.pdb', 'chemical/x-pdb');
}

function downloadPlddtCsv() {
  if (!resultPlddt.length) return;
  const header = 'residue,nucleotide,plddt\n';
  const rows = resultPlddt.map((v, i) =>
    `${i + 1},${resultSequence[i] || '?'},${v.toFixed(4)}`
  ).join('\n');
  downloadBlob(header + rows, 'plddt.csv', 'text/csv');
}

function downloadBlob(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// --- Start ---
init();
