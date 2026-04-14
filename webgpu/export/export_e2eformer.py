"""
Export E2EformerStack to ONNX.

Input:  msa_fea [1,K,L,256], pair_fea [1,L,L,128], msa_mask [1,K,L]
Output: msa_fea [1,K,L,256], pair_fea [1,L,L,128], single_fea [1,L,384]

Usage:
    cd <repo_root>
    python webgpu/export/export_e2eformer.py --validate
"""

import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rhofold'))

from common import patch_permute_final_dims, load_rhofold, validate_outputs
patch_permute_final_dims()


class E2EformerONNX(nn.Module):
    def __init__(self, rhofold):
        super().__init__()
        self.e2eformer = rhofold.e2eformer

    def forward(self, msa_fea, pair_fea, msa_mask):
        pair_mask = pair_fea.new_ones(pair_fea.shape[:3])
        msa_fea, pair_fea, single_fea = self.e2eformer(
            m=msa_fea, z=pair_fea,
            msa_mask=msa_mask, pair_mask=pair_mask,
            chunk_size=None,
        )
        return msa_fea, pair_fea, single_fea


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='./checkpoints/rhofold_pretrained_params.pt')
    parser.add_argument('--output', default='./webgpu/models/e2eformer.onnx')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--L', type=int, default=68)
    args = parser.parse_args()

    model, cfg = load_rhofold(args.ckpt)
    wrapper = E2EformerONNX(model)
    wrapper.eval()

    K, L = args.K, args.L
    msa_fea = torch.randn(1, K, L, 256)
    pair_fea = torch.randn(1, L, L, 128)
    msa_mask = torch.ones(1, K, L)

    print(f"Test forward (K={K}, L={L})...")
    with torch.no_grad():
        m, z, s = wrapper(msa_fea, pair_fea, msa_mask)
    print(f"  msa_fea: {m.shape}, pair_fea: {z.shape}, single: {s.shape}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print("Exporting to ONNX...")
    torch.onnx.export(
        wrapper,
        (msa_fea, pair_fea, msa_mask),
        args.output,
        input_names=['msa_fea', 'pair_fea', 'msa_mask'],
        output_names=['out_msa_fea', 'out_pair_fea', 'single_fea'],
        dynamic_axes={
            'msa_fea': {1: 'K', 2: 'L'}, 'pair_fea': {1: 'L', 2: 'L'},
            'msa_mask': {1: 'K', 2: 'L'},
            'out_msa_fea': {1: 'K', 2: 'L'}, 'out_pair_fea': {1: 'L', 2: 'L'},
            'single_fea': {1: 'L'},
        },
        opset_version=17, do_constant_folding=True,
    )
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Exported: {args.output} ({size_mb:.1f} MB)")

    if args.validate:
        import onnxruntime as ort
        torch.manual_seed(42)
        msa_fea = torch.randn(1, K, L, 256)
        pair_fea = torch.randn(1, L, L, 128)
        msa_mask = torch.ones(1, K, L)

        with torch.no_grad():
            pt_out = wrapper(msa_fea, pair_fea, msa_mask)

        sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
        ort_out = sess.run(None, {
            'msa_fea': msa_fea.numpy(),
            'pair_fea': pair_fea.numpy(),
            'msa_mask': msa_mask.numpy(),
        })
        print(f"\nValidation:")
        validate_outputs(pt_out, ort_out, ['out_msa_fea', 'out_pair_fea', 'single_fea'])


if __name__ == '__main__':
    main()
