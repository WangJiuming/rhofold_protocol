/**
 * Inference Web Worker: loads 5 ONNX sessions and runs the full
 * RhoFold pipeline with recycling.
 */

import * as ort from 'onnxruntime-web';
import { tokenizeMSA, tokenizeRNAFM, MSA_PADDING_IDX } from './tokenizer';
import { parseMSA, parseFASTA } from './msa-parser';
import { buildCoords } from './build-coords';
import { writePDB } from './pdb-writer';
import { ATOM_NUM_MAX } from './constants';

const MODELS_BASE_URL = 'https://r2.brighthong.com/v0';
const MODEL_CACHE_NAME = 'rhofold-models-v0';

// Message types
export interface InitMessage {
  type: 'init';
  backend: 'webgpu' | 'wasm';
}

export interface RunMessage {
  type: 'run';
  fastaContent: string;
  msaContent?: string;
  nRecycles: number;
  msaDepth: number;
}

export type WorkerMessage = InitMessage | RunMessage;

export interface ProgressMessage {
  type: 'progress';
  stage: string;
  recycle?: number;
  totalRecycles?: number;
}

export interface ResultMessage {
  type: 'result';
  pdb: string;
  plddt: number[];
  plddtGlobal: number;
  sequence: string;
}

export interface DownloadProgressMessage {
  type: 'download';
  loaded: number;
  total: number;
  allCached: boolean;
}

export interface ErrorMessage {
  type: 'error';
  message: string;
}

export interface StatusMessage {
  type: 'status';
  message: string;
  ready?: boolean;
}

export type WorkerResponse = ProgressMessage | ResultMessage | ErrorMessage | StatusMessage | DownloadProgressMessage;

// Sessions
let sessRnaFm: ort.InferenceSession | null = null;
let sessEmbedder: ort.InferenceSession | null = null;
let sessE2eformer: ort.InferenceSession | null = null;
let sessStructure: ort.InferenceSession | null = null;
let sessRefinenet: ort.InferenceSession | null = null;

function post(msg: WorkerResponse) {
  self.postMessage(msg);
}

// Aggregate download progress across all parallel fetches
const dlProgress: Record<string, { loaded: number; total: number }> = {};
let dlTotalKnown = 0;

function postAggregateProgress() {
  let loaded = 0;
  let total = 0;
  for (const p of Object.values(dlProgress)) {
    loaded += p.loaded;
    total += p.total;
  }
  post({ type: 'download', loaded, total, allCached: false });
}

/**
 * Fetch a model file with Cache API caching and download progress reporting.
 */
async function fetchModelCached(name: string, file: string): Promise<ArrayBuffer> {
  const url = `${MODELS_BASE_URL}/${file}`;
  const cache = await caches.open(MODEL_CACHE_NAME);

  // Check cache first
  const cached = await cache.match(url);
  if (cached) {
    return cached.arrayBuffer();
  }

  // Fetch with progress tracking
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to download ${name}: ${resp.status} ${resp.statusText}`);

  const contentLength = parseInt(resp.headers.get('Content-Length') || '0', 10);
  dlProgress[name] = { loaded: 0, total: contentLength };

  const reader = resp.body!.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    dlProgress[name].loaded = loaded;
    postAggregateProgress();
  }

  // Reassemble into a single ArrayBuffer
  const buf = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buf.set(chunk, offset);
    offset += chunk.length;
  }

  // Store in cache for next time
  await cache.put(url, new Response(buf.buffer, {
    headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(loaded) },
  }));

  return buf.buffer;
}

async function loadSessions(backend: 'webgpu' | 'wasm') {
  const ep = backend === 'webgpu' ? 'webgpu' : 'wasm';
  const opts: ort.InferenceSession.SessionOptions = {
    executionProviders: [ep],
  };

  const models = [
    { name: 'RNA-FM', file: 'rna_fm.onnx' },
    { name: 'Embedder', file: 'embedder.onnx' },
    { name: 'E2Eformer', file: 'e2eformer.onnx' },
    { name: 'Structure+Heads', file: 'structure_heads.onnx' },
    { name: 'RefineNet', file: 'refinenet.onnx' },
  ];

  // Download all models in parallel
  post({ type: 'status', message: 'Downloading models...' });
  const buffers = await Promise.all(
    models.map(m => fetchModelCached(m.name, m.file))
  );
  // Signal download complete (allCached if nothing was actually downloaded)
  const anythingDownloaded = Object.keys(dlProgress).length > 0;
  post({ type: 'download', loaded: 0, total: 0, allCached: !anythingDownloaded });

  // Create sessions sequentially (GPU session creation can't be parallelized)
  const sessions: ort.InferenceSession[] = [];
  for (let i = 0; i < models.length; i++) {
    const m = models[i];
    post({ type: 'status', message: `Loading ${m.name}...` });
    try {
      const sess = await ort.InferenceSession.create(buffers[i], opts);
      sessions.push(sess);
    } catch (e) {
      if (backend === 'webgpu') {
        post({ type: 'status', message: `WebGPU failed for ${m.name}, falling back to WASM...` });
        const wasmOpts: ort.InferenceSession.SessionOptions = {
          executionProviders: ['wasm'],
        };
        const sess = await ort.InferenceSession.create(buffers[i], wasmOpts);
        sessions.push(sess);
      } else {
        throw e;
      }
    }
  }

  [sessRnaFm, sessEmbedder, sessE2eformer, sessStructure, sessRefinenet] = sessions;
}

async function runPipeline(msg: RunMessage) {
  if (!sessRnaFm || !sessEmbedder || !sessE2eformer || !sessStructure || !sessRefinenet) {
    post({ type: 'error', message: 'Models not loaded. Call init first.' });
    return;
  }

  try {
    // Parse input
    const { name, sequence: seq } = parseFASTA(msg.fastaContent);
    const L = seq.length;

    // Parse MSA (or use single sequence)
    let msaSequences: string[];
    if (msg.msaContent) {
      msaSequences = parseMSA(msg.msaContent, msg.msaDepth);
    } else {
      msaSequences = [seq];
    }
    const K = Math.min(msaSequences.length, msg.msaDepth);

    post({ type: 'progress', stage: 'Tokenizing...' });

    // Tokenize
    const { tokens: msaTokens } = tokenizeMSA(msaSequences, msg.msaDepth);
    const fmTokens = tokenizeRNAFM(seq);

    // RNA-FM
    post({ type: 'progress', stage: 'Running RNA-FM...' });
    const rnaFmInput = {
      tokens: new ort.Tensor('int64', fmTokens, [1, L]),
    };
    const rnaFmOut = await sessRnaFm.run(rnaFmInput);
    const rnaFmRepr = rnaFmOut['representations'];

    // Initialize recycling state
    let recycleSingle = new ort.Tensor('float32', new Float32Array(L * 256), [1, L, 256]);
    let recyclePair = new ort.Tensor('float32', new Float32Array(L * L * 128), [1, L, L, 128]);
    let recycleC1 = new ort.Tensor('float32', new Float32Array(L * 3), [1, L, 3]);
    let recycleMask = new ort.Tensor('float32', new Float32Array([0.0]), [1]);

    // MSA mask (non-padding positions)
    const msaMaskData = new Float32Array(K * L);
    for (let i = 0; i < K * L; i++) {
      msaMaskData[i] = Number(msaTokens[i]) !== MSA_PADDING_IDX ? 1.0 : 0.0;
    }

    let frames: Float32Array = new Float32Array(0);
    let angles: Float32Array = new Float32Array(0);
    let plddtLocal: Float32Array = new Float32Array(0);
    let plddtGlobal = 0;
    let ssLogits: Float32Array = new Float32Array(0);
    let coordsMask: Uint8Array = new Uint8Array(0);
    let refinedCoords: Float32Array = new Float32Array(0);

    const msaTokensTensor = new ort.Tensor('int64', msaTokens, [1, K, L]);
    const msaMaskTensor = new ort.Tensor('float32', msaMaskData, [1, K, L]);

    for (let r = 0; r < msg.nRecycles; r++) {
      post({
        type: 'progress',
        stage: `Recycle ${r + 1}/${msg.nRecycles}`,
        recycle: r + 1,
        totalRecycles: msg.nRecycles,
      });

      // Embedder
      const embOut = await sessEmbedder.run({
        msa_tokens: msaTokensTensor,
        rna_fm_repr: rnaFmRepr,
        recycle_single: recycleSingle,
        recycle_pair: recyclePair,
        recycle_c1: recycleC1,
        recycle_mask: recycleMask,
      });
      const msaFea = embOut['msa_fea'];
      const pairFea = embOut['pair_fea'];

      // E2Eformer
      const e2eOut = await sessE2eformer.run({
        msa_fea: msaFea,
        pair_fea: pairFea,
        msa_mask: msaMaskTensor,
      });
      const msaFeaOut = e2eOut['out_msa_fea'];
      const pairFeaOut = e2eOut['out_pair_fea'];
      const singleFea = e2eOut['single_fea'];

      // Structure + Heads
      const structOut = await sessStructure.run({
        single_fea: singleFea,
        pair_fea: pairFeaOut,
      });

      frames = structOut['frames'].data as Float32Array;
      angles = structOut['angles'].data as Float32Array;
      plddtLocal = structOut['plddt_local'].data as Float32Array;
      plddtGlobal = (structOut['plddt_global'].data as Float32Array)[0];
      ssLogits = structOut['ss_logits'].data as Float32Array;

      // build_cords (JS)
      const buildResult = buildCoords(seq, frames, angles);
      coordsMask = buildResult.mask;
      const coordsFlat = buildResult.coords;

      // RefineNet
      const firstMsaRow = new BigInt64Array(L);
      for (let i = 0; i < L; i++) firstMsaRow[i] = msaTokens[i]; // row 0
      const refineOut = await sessRefinenet.run({
        first_msa_row: new ort.Tensor('int64', firstMsaRow, [1, L]),
        coords: new ort.Tensor('float32', coordsFlat, [1, L * ATOM_NUM_MAX, 3]),
      });
      refinedCoords = refineOut['refined_coords'].data as Float32Array;

      // Update recycling state
      const msaFeaOutData = msaFeaOut.data as Float32Array;
      const singleForRecycle = new Float32Array(L * 256);
      // Extract first MSA row: msaFeaOut[0, 0, :, :] → [L, 256]
      for (let i = 0; i < L * 256; i++) {
        singleForRecycle[i] = msaFeaOutData[i]; // row 0 is first L*256 elements
      }

      recycleSingle = new ort.Tensor('float32', singleForRecycle, [1, L, 256]);
      recyclePair = new ort.Tensor('float32', pairFeaOut.data as Float32Array, [1, L, L, 128]);

      // C1' coords from build_cords
      recycleC1 = new ort.Tensor('float32', buildResult.c1Coords, [1, L, 3]);
      recycleMask = new ort.Tensor('float32', new Float32Array([1.0]), [1]);
    }

    // Generate PDB
    post({ type: 'progress', stage: 'Generating PDB...' });
    const pdb = writePDB(seq, refinedCoords, coordsMask, plddtLocal);

    post({
      type: 'result',
      pdb,
      plddt: Array.from(plddtLocal),
      plddtGlobal,
      sequence: seq,
    });
  } catch (e: any) {
    post({ type: 'error', message: e.message || String(e) });
  }
}

// Worker message handler
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const msg = event.data;
  switch (msg.type) {
    case 'init':
      try {
        await loadSessions(msg.backend);
        post({ type: 'status', message: 'Ready', ready: true });
      } catch (e: any) {
        post({ type: 'error', message: `Failed to load models: ${e.message}` });
      }
      break;
    case 'run':
      await runPipeline(msg);
      break;
  }
};
