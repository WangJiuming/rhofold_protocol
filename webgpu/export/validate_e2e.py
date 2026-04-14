"""
End-to-end validation: run the full RhoFold pipeline in both PyTorch and
ONNX Runtime (chaining all 5 modular ONNX files + Python build_cords),
then compare final outputs.

This validates that the modular ONNX export preserves end-to-end accuracy
across all 10 recycling iterations.

Usage:
    cd <repo_root>
    python webgpu/export/validate_e2e.py
    python webgpu/export/validate_e2e.py --fasta data/rhofold/3owz_A/3owz_A.fasta --msa data/rhofold/3owz_A/3owz_A.afa
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rhofold'))

from common import patch_permute_final_dims, load_rhofold
patch_permute_final_dims()

from rhofold.config import rhofold_config
from rhofold.utils.converter import RNAConverter


def run_pytorch(model, tokens, rna_fm_tokens, seq, n_recycles=10):
    """Run full PyTorch pipeline and return final outputs."""
    model.eval()
    # Override the model's recycle count
    orig_recycles = model.config.model.recycling_embedder.recycles
    model.config.model.recycling_embedder.recycles = n_recycles
    with torch.no_grad():
        outputs = model.forward(tokens, rna_fm_tokens, seq)
    model.config.model.recycling_embedder.recycles = orig_recycles
    return outputs[-1]


def run_onnx_pipeline(
    sess_rna_fm, sess_embedder, sess_e2eformer, sess_structure, sess_refinenet,
    msa_tokens_np, rna_fm_tokens_np, seq, n_recycles=10, pad_idx=1,
):
    """
    Run the full pipeline using ONNX Runtime sessions, chaining modules
    with Python build_cords for the recycling loop.
    """
    converter = RNAConverter()
    L = msa_tokens_np.shape[2]

    # Step 1: RNA-FM (run once)
    print("  RNA-FM...", end=" ", flush=True)
    t0 = time.time()
    rna_fm_out = sess_rna_fm.run(None, {'tokens': rna_fm_tokens_np})
    rna_fm_repr = rna_fm_out[0]  # [1, L, 640]
    print(f"{time.time() - t0:.2f}s")

    # Initialize recycling state (zeros on first iteration)
    recycle_single = np.zeros((1, L, 256), dtype=np.float32)
    recycle_pair = np.zeros((1, L, L, 128), dtype=np.float32)
    recycle_c1 = np.zeros((1, L, 3), dtype=np.float32)
    recycle_mask = np.array([0.0], dtype=np.float32)

    # MSA mask: non-padding positions
    msa_mask = (msa_tokens_np != pad_idx).astype(np.float32)

    for r in range(n_recycles):
        print(f"  Recycle {r+1}/{n_recycles}...", end=" ", flush=True)
        t0 = time.time()

        # Embedder
        emb_out = sess_embedder.run(None, {
            'msa_tokens': msa_tokens_np,
            'rna_fm_repr': rna_fm_repr,
            'recycle_single': recycle_single,
            'recycle_pair': recycle_pair,
            'recycle_c1': recycle_c1,
            'recycle_mask': recycle_mask,
        })
        msa_fea, pair_fea = emb_out

        # E2Eformer
        e2e_out = sess_e2eformer.run(None, {
            'msa_fea': msa_fea,
            'pair_fea': pair_fea,
            'msa_mask': msa_mask,
        })
        msa_fea_out, pair_fea_out, single_fea = e2e_out

        # Structure + Heads
        struct_out = sess_structure.run(None, {
            'single_fea': single_fea,
            'pair_fea': pair_fea_out,
        })
        frames, angles = struct_out[0], struct_out[1]
        plddt_local, plddt_global = struct_out[2], struct_out[3]
        ss_logits = struct_out[4]
        dist_p, dist_c4, dist_n = struct_out[5], struct_out[6], struct_out[7]

        # build_cords (Python — not in ONNX)
        frames_t = torch.from_numpy(frames)
        angles_t = torch.from_numpy(angles)
        cords, cmsk = converter.build_cords(seq, frames_t, angles_t, rtn_cmsk=True)
        # cords shape: [L, 23, 3]
        coords_flat = cords.unsqueeze(0).reshape(1, -1, 3).numpy()

        # RefineNet
        first_msa_row = msa_tokens_np[:, 0, :]  # [1, L]
        refine_out = sess_refinenet.run(None, {
            'first_msa_row': first_msa_row,
            'coords': coords_flat.astype(np.float32),
        })
        refined_coords = refine_out[0]  # [1, L*23, 3]

        # Extract recycling state for next iteration
        recycle_single = msa_fea_out[:, 0, :, :]  # [1, L, 256]
        recycle_pair = pair_fea_out                 # [1, L, L, 128]
        recycle_c1 = cords[:, 1, :].unsqueeze(0).numpy()  # [1, L, 3] — C1' atom
        recycle_mask = np.array([1.0], dtype=np.float32)

        print(f"{time.time() - t0:.2f}s")

    return {
        'frames': frames,
        'angles': angles,
        'plddt_local': plddt_local,
        'plddt_global': plddt_global,
        'ss_logits': ss_logits,
        'dist_p': dist_p,
        'dist_c4': dist_c4,
        'dist_n': dist_n,
        'refined_coords': refined_coords,
        'single_fea': single_fea,
    }


def compare_outputs(pt_output, ort_output):
    """Compare PyTorch and ONNX outputs after full recycling."""
    print("\n=== End-to-End Comparison (after all recycles) ===\n")
    all_pass = True

    # pLDDT
    pt_plddt = pt_output['plddt'][0].numpy()  # (L,)
    ort_plddt = ort_output['plddt_local'].squeeze()
    diff = np.max(np.abs(pt_plddt - ort_plddt))
    status = "PASS" if diff < 0.05 else ("WARN" if diff < 0.2 else "FAIL")
    print(f"  pLDDT (local):    max_diff={diff:.4f}  {status}")
    if status == "FAIL":
        all_pass = False

    # SS logits
    pt_ss = pt_output['ss']
    if isinstance(pt_ss, torch.Tensor):
        pt_ss = pt_ss.numpy()
    ort_ss = ort_output['ss_logits']
    diff = np.max(np.abs(pt_ss.reshape(ort_ss.shape) - ort_ss))
    status = "PASS" if diff < 0.1 else ("WARN" if diff < 1.0 else "FAIL")
    print(f"  SS logits:        max_diff={diff:.4f}  {status}")
    if status == "FAIL":
        all_pass = False

    # Distance heads
    for name_pt, name_ort in [('p', 'dist_p'), ('c4_', 'dist_c4'), ('n', 'dist_n')]:
        pt_d = pt_output[name_pt]
        if isinstance(pt_d, torch.Tensor):
            pt_d = pt_d.numpy()
        ort_d = ort_output[name_ort]
        diff = np.max(np.abs(pt_d.reshape(ort_d.shape) - ort_d))
        status = "PASS" if diff < 0.1 else ("WARN" if diff < 1.0 else "FAIL")
        print(f"  dist_{name_pt:4s}:         max_diff={diff:.4f}  {status}")
        if status == "FAIL":
            all_pass = False

    # Refined coordinates — RMSD
    pt_cords = pt_output['cord_tns_pred'][0].numpy()  # [1, L*23, 3]
    ort_cords = ort_output['refined_coords']  # [1, L*23, 3]
    pt_flat = pt_cords.reshape(-1, 3)
    ort_flat = ort_cords.reshape(-1, 3)
    # Only compare non-zero coords (masked atoms)
    mask = np.any(pt_flat != 0, axis=-1) & np.any(ort_flat != 0, axis=-1)
    if mask.sum() > 0:
        d = pt_flat[mask] - ort_flat[mask]
        rmsd = np.sqrt(np.mean(np.sum(d ** 2, axis=-1)))
        max_dist = np.max(np.sqrt(np.sum(d ** 2, axis=-1)))
        status = "PASS" if rmsd < 0.5 else ("WARN" if rmsd < 2.0 else "FAIL")
        print(f"  Coords RMSD:      {rmsd:.4f} A  (max_dist={max_dist:.4f} A)  {status}")
        if status == "FAIL":
            all_pass = False

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description='End-to-end ONNX vs PyTorch validation')
    parser.add_argument('--ckpt', default='./checkpoints/rhofold_pretrained_params.pt')
    parser.add_argument('--models-dir', default='./webgpu/models')
    parser.add_argument('--fasta', default='./data/rhofold/3owz_A/3owz_A.fasta')
    parser.add_argument('--msa', default='./data/rhofold/3owz_A/3owz_A.afa')
    parser.add_argument('--recycles', type=int, default=10)
    args = parser.parse_args()

    import onnxruntime as ort

    # Load ONNX sessions
    print("Loading ONNX sessions...")
    providers = ['CPUExecutionProvider']
    sess_rna_fm = ort.InferenceSession(os.path.join(args.models_dir, 'rna_fm.onnx'), providers=providers)
    sess_embedder = ort.InferenceSession(os.path.join(args.models_dir, 'embedder.onnx'), providers=providers)
    sess_e2eformer = ort.InferenceSession(os.path.join(args.models_dir, 'e2eformer.onnx'), providers=providers)
    sess_structure = ort.InferenceSession(os.path.join(args.models_dir, 'structure_heads.onnx'), providers=providers)
    sess_refinenet = ort.InferenceSession(os.path.join(args.models_dir, 'refinenet.onnx'), providers=providers)
    print("  All sessions loaded.")

    # Prepare input from FASTA + MSA
    from rhofold.utils.alphabet import get_features
    print(f"\nPreparing features from {args.fasta}...")
    data_dict = get_features(args.fasta, args.msa)
    seq = data_dict['seq']
    msa_tokens = data_dict['tokens']          # [1, K, L]
    rna_fm_tokens = data_dict['rna_fm_tokens'] # [1, L]
    print(f"  Sequence: {seq[:40]}{'...' if len(seq) > 40 else ''} (L={len(seq)})")
    print(f"  MSA tokens: {msa_tokens.shape}, RNA-FM tokens: {rna_fm_tokens.shape}")

    # Numpy for ONNX — truncate MSA to model's msa_depth
    msa_depth = rhofold_config.globals.msa_depth  # 128
    msa_tokens_np = msa_tokens[:, :msa_depth].numpy().astype(np.int64)
    rna_fm_tokens_np = rna_fm_tokens.numpy().astype(np.int64)

    # Padding index
    from rhofold.model.rna_fm.data import Alphabet
    pad_idx = Alphabet.from_architecture("ESM-1", theme="rna").padding_idx

    # Run ONNX pipeline
    print(f"\n--- ONNX Pipeline ({args.recycles} recycles) ---")
    t0 = time.time()
    ort_output = run_onnx_pipeline(
        sess_rna_fm, sess_embedder, sess_e2eformer, sess_structure, sess_refinenet,
        msa_tokens_np, rna_fm_tokens_np, seq,
        n_recycles=args.recycles, pad_idx=pad_idx,
    )
    print(f"  Total ONNX: {time.time() - t0:.2f}s")

    # Run PyTorch pipeline
    print(f"\n--- PyTorch Pipeline ({args.recycles} recycles) ---")
    model, cfg = load_rhofold(args.ckpt)
    t0 = time.time()
    pt_output = run_pytorch(model, msa_tokens, rna_fm_tokens, seq, n_recycles=args.recycles)
    print(f"  Total PyTorch: {time.time() - t0:.2f}s")

    # Compare
    compare_outputs(pt_output, ort_output)


if __name__ == '__main__':
    main()
