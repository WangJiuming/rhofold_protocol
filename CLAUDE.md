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
└── webgpu/                       # [NEW] WebGPU port lives here
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

# With profiling
python rhofold/inference.py \
  --fasta ./data/rhofold/3owz_A/3owz_A.fasta \
  --msa ./data/rhofold/3owz_A/3owz_A.afa \
  --output-dir ./results/rhofold/3owz_A \
  --device cpu --profile
```

## Tech Stack

### Runtime: ONNX Runtime Web (WebGPU backend)

ONNX Runtime Web is the most mature WebGPU ML runtime for browsers. It handles GPU dispatch, memory management, shader compilation, and supports the full ONNX opset we need (Einsum, etc.).

### Export Strategy: 4 Modular ONNX Files

| Module | Params | Input → Output |
|--------|--------|---------------|
| **RNA-FM** | ~100M | `tokens [1,L]` → `repr [1,L,640]` (run once) |
| **Embedder** | ~1.3M | `msa_tokens + rna_fm_repr + recycling_state` → `msa_fea + pair_fea` |
| **E2Eformer** | ~25M | `msa_fea + pair_fea` → `msa_fea' + pair_fea' + single_fea` |
| **StructureModule+Heads** | ~1M | `single_fea + pair_fea + seq` → `coords + plddt + ss + dist` |

Recycling loop (10 iterations of Embedder→E2Eformer→StructureModule) orchestrated in TypeScript.

### Precision Strategy

**Phase 1 (current)**: fp32 everywhere — match PyTorch reference exactly, eliminate precision as a variable. ~508MB total weight size.

**Phase 2 (optimization)**: fp16 storage + mixed f16/f32 compute. Pin IPA distance, OPM einsum, and quaternion ops to f32. Target ~254MB. Validate against bf16-mixed CUDA baseline.

**Phase 3 (stretch)**: int8 for frozen RNA-FM → ~154MB total.

### Target Sequence Length

L ≤ 200 (covers tRNAs, riboswitches, aptamers, most structured RNAs). Memory at L=200 fp32: pair_fea ~20MB, OPM peak ~160MB — comfortable for any modern GPU.

### Scope

- **In scope**: Full forward-pass inference, weight loading, coordinate generation, PDB export
- **Out of scope**: Amber relaxation (OpenMM), MSA search (BLAST/Infernal databases), training

### Key Non-Standard Ops to Handle for ONNX Export

1. **Rigid class** → decompose to explicit rotation matrices `[*,3,3]` + translations `[*,3]`
2. **IPA 3D pairwise distance** → Einsum + Sub + Mul + ReduceSum
3. **Outer product mean** `einsum('...bac,...dae->...bdce')` → decompose to batched MatMul
4. **Quaternion ↔ rotation** → precomputed coefficient tensors + reshape-based contraction
5. **Triangle multiplicative updates** → channel-as-batch MatMul with sigmoid gating
6. **Multi-bias attention** → pre-sum biases before SDPA
7. **EGNN coordinate update** → weighted coordinate aggregation einsum
8. **Frame chaining loop** → unroll sequential rigid body composition

### Milestones

1. Export RNA-FM to ONNX (fp32), validate numerical equivalence in Python
2. Run RNA-FM in browser via ORT-Web WebGPU EP
3. Export Embedder + E2Eformer — rewrite Rigid abstractions, decompose custom ops
4. Export StructureModule + Heads — rewrite IPA and EGNN for ONNX
5. End-to-end browser inference on 3owz_A (L=68)
6. Optimization: fp16 weights, memory profiling, streaming weights
7. Web UI: FASTA input + Mol* 3D viewer + progress indicators

### Frontend

- **3D viewer**: Mol* (Molstar)
- **UI**: Lightweight TypeScript
- **Threading**: ONNX Runtime Web runs in a Web Worker

## Conventions

- Keep the original Python code untouched as the reference implementation
- All WebGPU code goes under `webgpu/`
- Phase 1 uses fp32 — validate numerical equivalence (max |diff| < 1e-4 per module)
- Document any operator that required special decomposition for ONNX export
- Weights stored in ONNX external data format under `webgpu/models/` (gitignored)
