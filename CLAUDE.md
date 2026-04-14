# CLAUDE.md — RhoFold WebGPU Port

## Project Overview

RhoFold+ is a deep learning system for predicting RNA 3D structure from sequence and MSA (Multiple Sequence Alignment). This branch (`dev/webgpu`) ports the PyTorch inference pipeline to run in-browser via WebGPU.

The original model is described in "A language-model-based deep learning platform for predicting RNA 3D structures" (Nature Methods, Shen et al., 2024).

## Branch Goal

Port RhoFold+ inference to WebGPU so that RNA 3D structure prediction can run entirely client-side in a web browser, with no server-side GPU required.

## Architecture Summary

The PyTorch model has five major components that must be ported:

1. **MSAEmbedder** — tokenizes MSA + query, builds pair representation, integrates frozen RNA-FM (ESM1b-style, 12-layer, dim=640) features
2. **RecyclingEmbedder** — feeds back single/pair features and C1' coordinates across 10 recycle iterations
3. **E2EformerStack** — 12 transformer blocks with MSA row/column attention, outer product mean, triangle multiplicative updates, and triangle attention (AlphaFold2-style Evoformer adapted for RNA)
4. **StructureModule** — 8 IPA (Invariant Point Attention) blocks + AngleResnet + RefineNet (E(n)-equivariant GNN, 4 layers)
5. **Prediction Heads** — DistHead (pairwise distances), SSHead (secondary structure), pLDDTHead (confidence)

### Key dimensions

| Parameter | Value |
|-----------|-------|
| c_m (MSA) | 256 |
| c_z (pair) | 128 |
| c_s (single) | 384 |
| MSA depth | 128 |
| E2Eformer blocks | 12 |
| Structure module blocks | 8 |
| Recycles | 10 |
| Torsion angles/residue | 6 |
| Max atoms/residue | 23 |

## Repository Layout

```
rhofold_protocol/
├── rhofold/
│   ├── inference.py              # CLI entry point for structure prediction
│   └── rhofold/
│       ├── rhofold.py            # Top-level nn.Module (RhoFold)
│       ├── config.py             # Model hyperparameters (ml_collections)
│       ├── model/
│       │   ├── embedders.py      # MSAEmbedder, RecyclingEmbedder
│       │   ├── e2eformer.py      # E2EformerStack (12 blocks)
│       │   ├── msa.py            # MSA row/column attention
│       │   ├── pair.py           # Pair embeddings and transitions
│       │   ├── outer_product_mean.py
│       │   ├── triangular_attention.py
│       │   ├── triangular_update.py
│       │   ├── structure_module.py  # IPA, AngleResnet, RefineNet
│       │   ├── heads.py          # Dist, SS, pLDDT heads
│       │   ├── primitives.py     # Linear, LayerNorm, Attention (SDPA)
│       │   └── rna_fm/           # Frozen RNA-FM language model
│       ├── relax/                # Amber relaxation (NOT ported — server-side only)
│       ├── utils/
│       │   ├── alphabet.py       # Tokenization, feature construction
│       │   ├── constants.py      # RNA geometry constants
│       │   ├── converter.py      # Frames+angles -> 3D coordinates + PDB
│       │   ├── rigid_utils.py    # Quaternion rigid body math
│       │   ├── chunk_utils.py    # Chunked forward passes
│       │   └── ss_utils.py       # Secondary structure utilities
│       └── data/
│           └── balstn.py
├── checkpoints/
│   └── rhofold_pretrained_params.pt  # Pre-trained weights (~350MB)
├── data/                         # Example inputs (FASTA, MSA, PDB)
├── scripts/                      # Analysis/evaluation scripts
├── environment.yml               # Conda env (Linux/CUDA)
├── environment_mac.yml           # Conda env (macOS)
└── webgpu/                       # WebGPU port
    ├── export/                   # Python ONNX export scripts
    │   ├── common.py             # Shared utilities (load model, validate)
    │   ├── export_rna_fm.py      # RNA-FM export (378 MB)
    │   ├── export_embedder.py    # Embedder export (5 MB)
    │   ├── export_e2eformer.py   # E2Eformer export (86 MB)
    │   ├── export_structure.py   # Structure+Heads export (11 MB)
    │   ├── export_refinenet.py   # RefineNet export (5 MB)
    │   └── validate_e2e.py       # End-to-end validation (10 recycles, RMSD 0.016 A)
    ├── models/                   # ONNX files (gitignored, hosted on R2)
    ├── src/
    │   ├── main.ts               # UI entry point, worker coordination
    │   ├── style.css             # Page layout and component styles
    │   ├── molstar.ts            # Mol* 3D viewer, pLDDT coloring
    │   ├── inference-worker.ts   # Web Worker: ORT sessions + pipeline
    │   ├── tokenizer.ts          # MSA + RNA-FM tokenization
    │   ├── msa-parser.ts         # Parse .afa aligned FASTA
    │   ├── build-coords.ts       # Frames+angles -> all-atom coordinates
    │   ├── pdb-writer.ts         # Coordinates -> PDB string
    │   ├── constants.ts          # RNA atom defs, rigid groups, local coords
    │   └── rigid.ts              # Quaternion->rotmat, frame composition
    ├── index.html                # Main page
    ├── package.json              # onnxruntime-web, molstar, vite
    ├── vite.config.ts            # COOP/COEP headers, model serving
    └── tsconfig.json
```

## Development Commands

```bash
# Activate environment
conda activate rhofold_protocol

# Run PyTorch inference (reference baseline)
python rhofold/inference.py \
  --fasta ./data/rhofold/3owz_A/3owz_A.fasta \
  --msa ./data/rhofold/3owz_A/3owz_A.afa \
  --output-dir ./results/rhofold/3owz_A \
  --device cpu

# ONNX export (run from repo root with conda env active)
python webgpu/export/export_rna_fm.py
python webgpu/export/export_embedder.py
python webgpu/export/export_e2eformer.py
python webgpu/export/export_structure.py
python webgpu/export/export_refinenet.py

# End-to-end ONNX validation (10 recycles)
python webgpu/export/validate_e2e.py \
  --fasta data/rhofold/3owz_A/3owz_A.fasta \
  --msa data/rhofold/3owz_A/3owz_A.afa

# Web UI development
cd webgpu && npm install && npm run dev
# Opens at http://localhost:5173/
```

## Tech Stack

### Runtime: ONNX Runtime Web (WebGPU backend)

ONNX Runtime Web is the most mature WebGPU ML runtime for browsers. It handles GPU dispatch, memory management, shader compilation, and supports the full ONNX opset we need (Einsum, etc.).

### Export Strategy: 5 Modular ONNX Files

| Module | Size | Input -> Output |
|--------|------|-----------------|
| **RNA-FM** | 378 MB | `tokens [1,L]` -> `repr [1,L,640]` (run once) |
| **Embedder** | 5 MB | `msa_tokens + rna_fm_repr + recycling_state` -> `msa_fea + pair_fea` |
| **E2Eformer** | 86 MB | `msa_fea + pair_fea + msa_mask` -> `msa_fea' + pair_fea' + single_fea` |
| **Structure+Heads** | 11 MB | `single_fea + pair_fea` -> `frames + angles + plddt + ss_logits` |
| **RefineNet** | 5 MB | `first_msa_row + coords` -> `refined_coords` |

Recycling loop (1-10 iterations of Embedder -> E2Eformer -> Structure -> build_cords -> RefineNet) orchestrated in TypeScript.

### Model Hosting

ONNX models are hosted on Cloudflare R2 at `https://r2.brighthong.com/v0/`.
- R2 bucket: `rhofold` (account `8c3ded8031b5fc24914f3205bdf61093`)
- CORS enabled for all origins
- Browser caches models via Cache API (`rhofold-models-v0`)
- All 5 models downloaded in parallel on first use (~486 MB total)

### Browser Pipeline

```
Main Thread                          Web Worker
┌──────────────────────┐       ┌──────────────────────────┐
│  UI (index.html)     │       │  inference-worker.ts     │
│  - FASTA textarea    │  msg  │  - Download from R2      │
│  - MSA file upload   │<----->│  - Cache API caching     │
│  - Options panel     │       │  - Load 5 ONNX sessions  │
│  - Mol* 3D viewer    │       │  - RNA-FM (once)         │
│  - pLDDT chart       │       │  - Recycle loop x N:     │
│  - Progress bar      │       │    Embed->E2E->Struct    │
│  - Download buttons  │       │    build_cords (JS)      │
│                      │       │    RefineNet             │
└──────────────────────┘       │  - Generate PDB string   │
                               └──────────────────────────┘
```

### Precision Strategy

**Phase 1 (current)**: fp32 everywhere — match PyTorch reference exactly, eliminate precision as a variable. ~486 MB total weight size.

**Phase 2 (optimization)**: fp16 storage + mixed f16/f32 compute. Pin IPA distance, OPM einsum, and quaternion ops to f32. Target ~254MB. Validate against bf16-mixed CUDA baseline.

**Phase 3 (stretch)**: int8 for frozen RNA-FM -> ~154MB total.

### Target Sequence Length

L <= 200 (covers tRNAs, riboswitches, aptamers, most structured RNAs). Memory at L=200 fp32: pair_fea ~20MB, OPM peak ~160MB — comfortable for any modern GPU.

### Scope

- **In scope**: Full forward-pass inference, weight loading, coordinate generation, PDB export
- **Out of scope**: Amber relaxation (OpenMM), MSA search (BLAST/Infernal databases), training

### Key Non-Standard Ops Handled in ONNX Export

1. **Rigid class** -> decompose to explicit rotation matrices `[*,3,3]` + translations `[*,3]`
2. **IPA 3D pairwise distance** -> Einsum + Sub + Mul + ReduceSum
3. **Outer product mean** `einsum('...bac,...dae->...bdce')` -> decompose to batched MatMul
4. **Quaternion <-> rotation** -> precomputed coefficient tensors + reshape-based contraction
5. **Triangle multiplicative updates** -> channel-as-batch MatMul with sigmoid gating
6. **Multi-bias attention** -> pre-sum biases before SDPA
7. **EGNN coordinate update** -> weighted coordinate aggregation einsum
8. **CoorsNorm** -> LayerNorm(1) replaced with learned bias multiply (ORT rejects dim=1 LayerNorm)
9. **Frame chaining loop** -> unroll sequential rigid body composition
10. **build_cords** -> NOT in ONNX; ported to TypeScript (`build-coords.ts` + `rigid.ts`)

### Milestones

1. ~~Export RNA-FM to ONNX (fp32), validate numerical equivalence in Python~~
2. ~~Export Embedder + E2Eformer — rewrite Rigid abstractions, decompose custom ops~~
3. ~~Export StructureModule + Heads — rewrite IPA and EGNN for ONNX~~
4. ~~Export RefineNet — CoorsNorm workaround~~
5. ~~End-to-end validation: 10 recycles, RMSD 0.016 A vs PyTorch~~
6. ~~Web UI: FASTA input, Mol* 3D viewer, progress indicators, model caching~~
7. Browser testing and debugging
8. Optimization: fp16 weights, memory profiling

### Frontend

- **3D viewer**: Mol* (Molstar) with uncertainty color theme (pLDDT)
- **UI**: TypeScript + Vite, two-panel layout
- **Threading**: ONNX Runtime Web runs in a Web Worker
- **Options**: Recycles (1-10), MSA depth (4/16/64/128), Backend (WebGPU/WASM)
- **Caching**: Models cached via Cache API with download/clear UI

## Conventions

- Keep the original Python code untouched as the reference implementation
- All WebGPU code goes under `webgpu/`
- Phase 1 uses fp32 — validate numerical equivalence (max |diff| < 1e-4 per module)
- Document any operator that required special decomposition for ONNX export
- ONNX weights hosted on R2 (gitignored locally under `webgpu/models/`)
- Two tokenization schemes: MSA alphabet (A=4,U=5,G=6,C=7) and RNA-FM ESM-1b alphabet (A=4,C=5,G=6,U=7) — different orderings
