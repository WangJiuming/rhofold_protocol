# RhoFold+ WebGPU

**Client-side RNA 3D structure prediction, directly in your browser.** No GPU server, no Python environment, no installation — open a page and predict. Powered by WebGPU via ONNX Runtime Web.

This is the `dev/webgpu` branch of the [RhoFold+ protocol](https://github.com/WangJiuming/rhofold_protocol). For the original Linux/CUDA PyTorch protocol that produced these weights, see the [`main` branch](https://github.com/WangJiuming/rhofold_protocol/tree/main).

## Quick start

```bash
git clone -b dev/webgpu https://github.com/WangJiuming/rhofold_protocol.git
cd rhofold_protocol/webgpu
npm install
npm run dev
```

Open <http://localhost:5173/> in a WebGPU-capable browser (Chrome/Edge 113+, Safari 18.2+, or any modern Chromium).

## Usage

1. **Paste a FASTA sequence** (RNA, A/U/G/C, ≤ 200 nt recommended) — or click **Load example** for the 3owz_A tRNA, which also auto-loads its MSA.
2. **(Optional) Upload an MSA** — aligned FASTA (`.afa`). Deeper MSAs improve accuracy. If omitted, the query alone is used.
3. **Adjust options** (optional):
   - **Recycles** (1–10) — more iterations = slightly better quality, linear time cost.
   - **MSA depth** (4 / 16 / 64 / 128) — deeper = better accuracy, slightly more memory.
   - **Backend** — WebGPU (default, fast) or WASM (fallback, ~10× slower).
4. Click **Predict**. On first run, the 5 ONNX models (~485 MB) download from Cloudflare R2 and are cached in the browser's Cache API.
5. View the predicted 3D structure, colored by per-residue pLDDT, and download the `.pdb`.

## Performance

Benchmarked on an Electron-based Chromium browser (macOS, WebGPU), 3owz_A (L=88, MSA depth 128):

| Phase | Cold start | Warm (cached) |
|---|---|---|
| Download models (~485 MB) | ~53 s | 0 |
| Session init / shader compile | ~8 s | ~8 s |
| Inference, 10 recycles | ~85 s | ~85 s |
| **Total click → PDB** | **~2.5 min** | **~1.5 min** |

- Per-recycle cost: **~8–10 s**, dominated by **E2Eformer (~53 %)** and **RefineNet (~43 %)**.
- With **recycles = 3**, inference drops to **~35 s** with only a minor pLDDT difference (0.819 vs 0.824 on 3owz_A).

## Browser requirements

- **WebGPU enabled.** Verify at `chrome://gpu` — WebGPU should be "Hardware accelerated."
- **≥ 2 GB free GPU memory** for sequences up to L = 200 at fp32.
- **~500 MB Cache API quota.** If the 378 MB RNA-FM model fails to cache (some browsers reject single entries this large), prediction still works but the model re-downloads each reload.

Without WebGPU, the app falls back to WASM automatically — correct but significantly slower.

## Repository layout

```
webgpu/
├── src/                    # TypeScript: UI, worker, pipeline
│   ├── main.ts             # UI entry point
│   ├── inference-worker.ts # Web Worker: ORT sessions + recycle loop
│   ├── molstar.ts          # Mol* 3D viewer (pLDDT coloring)
│   ├── tokenizer.ts        # MSA + RNA-FM tokenization
│   ├── build-coords.ts     # Frames + angles → all-atom coordinates
│   └── pdb-writer.ts       # Coordinates → PDB string
├── export/                 # Python scripts that exported the 5 ONNX models
├── index.html              # Main page
└── vite.config.ts          # Dev server with COOP/COEP headers
```

The PyTorch reference implementation lives under `rhofold/` at the repo root.

## Limitations

- Sequence length ≤ 200 nt recommended.
- MSA search is not included — generate the `.afa` input with rMSA2, Infernal, or similar.

## Citation

If you use this work, please cite the original paper:

> Shen, Tao, *et al.* "Accurate RNA 3D structure prediction using a language model-based deep learning approach." *Nature Methods* (2024).

## Contact

Open an issue, or reach the author at <jmwang@link.cuhk.edu.hk>.
