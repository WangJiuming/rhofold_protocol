"""
Export MSAEmbedder (sans RNA-FM) + RecyclingEmbedder to ONNX.

Input:  msa_tokens [1,K,L], rna_fm_repr [1,L,640],
        recycle_single [1,L,256], recycle_pair [1,L,L,128],
        recycle_c1 [1,L,3], recycle_mask [1]
Output: msa_fea [1,K,L,256], pair_fea [1,L,L,128]

Usage:
    cd <repo_root>
    python webgpu/export/export_embedder.py --validate
"""

import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rhofold'))

from common import patch_permute_final_dims, load_rhofold, validate_outputs
patch_permute_final_dims()


class EmbedderONNX(nn.Module):
    def __init__(self, rhofold):
        super().__init__()
        self.msa_emb = rhofold.msa_embedder.msa_emb
        self.pair_emb = rhofold.msa_embedder.pair_emb
        self.rna_fm_reduction = rhofold.msa_embedder.rna_fm_reduction
        self.recycle_embnet = rhofold.recycle_embnet
        self.msa_depth = rhofold.config.globals.msa_depth

    def forward(self, msa_tokens, rna_fm_repr, recycle_single, recycle_pair, recycle_c1, recycle_mask):
        msa_tokens = msa_tokens[:, :self.msa_depth]
        B, K, L = msa_tokens.shape

        msa_fea = self.msa_emb(msa_tokens)
        token_repr = rna_fm_repr.unsqueeze(1).expand(-1, K, -1, -1)
        msa_fea = self.rna_fm_reduction(torch.cat([token_repr, msa_fea], dim=-1))
        pair_fea = self.pair_emb(msa_tokens, t1ds=None, t2ds=None)

        m_update, z_update = self.recycle_embnet(recycle_single, recycle_pair, recycle_c1)
        rm = recycle_mask.view(-1, 1, 1)
        msa_fea[..., 0, :, :] = msa_fea[..., 0, :, :] + m_update * rm
        pair_fea = pair_fea + z_update * rm.unsqueeze(-1)

        return msa_fea, pair_fea


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='./checkpoints/rhofold_pretrained_params.pt')
    parser.add_argument('--output', default='./webgpu/models/embedder.onnx')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--L', type=int, default=68)
    args = parser.parse_args()

    model, cfg = load_rhofold(args.ckpt)
    wrapper = EmbedderONNX(model)
    wrapper.eval()

    K, L = args.K, args.L
    msa_tokens = torch.randint(0, 9, (1, K, L), dtype=torch.long)
    rna_fm_repr = torch.randn(1, L, 640)
    recycle_single = torch.zeros(1, L, 256)
    recycle_pair = torch.zeros(1, L, L, 128)
    recycle_c1 = torch.zeros(1, L, 3)
    recycle_mask = torch.tensor([0.0])

    print(f"Test forward (K={K}, L={L})...")
    with torch.no_grad():
        msa_fea, pair_fea = wrapper(msa_tokens, rna_fm_repr, recycle_single, recycle_pair, recycle_c1, recycle_mask)
    print(f"  msa_fea: {msa_fea.shape}, pair_fea: {pair_fea.shape}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print("Exporting to ONNX...")
    torch.onnx.export(
        wrapper,
        (msa_tokens, rna_fm_repr, recycle_single, recycle_pair, recycle_c1, recycle_mask),
        args.output,
        input_names=['msa_tokens', 'rna_fm_repr', 'recycle_single', 'recycle_pair', 'recycle_c1', 'recycle_mask'],
        output_names=['msa_fea', 'pair_fea'],
        dynamic_axes={
            'msa_tokens': {1: 'K', 2: 'L'}, 'rna_fm_repr': {1: 'L'},
            'recycle_single': {1: 'L'}, 'recycle_pair': {1: 'L', 2: 'L'},
            'recycle_c1': {1: 'L'},
            'msa_fea': {1: 'K', 2: 'L'}, 'pair_fea': {1: 'L', 2: 'L'},
        },
        opset_version=17, do_constant_folding=True,
    )
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Exported: {args.output} ({size_mb:.1f} MB)")

    if args.validate:
        import onnxruntime as ort
        torch.manual_seed(42)
        msa_tokens = torch.randint(0, 9, (1, K, L), dtype=torch.long)
        rna_fm_repr = torch.randn(1, L, 640)
        recycle_single = torch.zeros(1, L, 256)
        recycle_pair = torch.zeros(1, L, L, 128)
        recycle_c1 = torch.zeros(1, L, 3)
        recycle_mask = torch.tensor([0.0])

        with torch.no_grad():
            pt_out = wrapper(msa_tokens, rna_fm_repr, recycle_single, recycle_pair, recycle_c1, recycle_mask)

        sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
        ort_out = sess.run(None, {
            'msa_tokens': msa_tokens.numpy(), 'rna_fm_repr': rna_fm_repr.numpy(),
            'recycle_single': recycle_single.numpy(), 'recycle_pair': recycle_pair.numpy(),
            'recycle_c1': recycle_c1.numpy(), 'recycle_mask': recycle_mask.numpy(),
        })
        print(f"\nValidation:")
        validate_outputs(pt_out, ort_out, ['msa_fea', 'pair_fea'])


if __name__ == '__main__':
    main()
