"""
Export the RhoFold trunk (everything except RNA-FM and coordinate construction)
to ONNX format as a single-cycle model.

The JS orchestrator runs RNA-FM once, then loops this model for 10 recycles,
doing coordinate construction in JS between cycles.

Usage:
    conda activate rhofold_protocol
    cd <repo_root>
    python webgpu/export/export_trunk.py \
        --ckpt ./checkpoints/rhofold_pretrained_params.pt \
        --output ./webgpu/models/rhofold_trunk.onnx \
        --validate
"""

import argparse
import os
import sys
import math
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rhofold'))

from rhofold.config import rhofold_config
from rhofold.rhofold import RhoFold
from rhofold.model.embedders import MSAEmbedder, RecyclingEmbedder
from rhofold.model.e2eformer import E2EformerStack
from rhofold.model.structure_module import StructureModule
from rhofold.model.heads import DistHead, SSHead, pLDDTHead
from rhofold.utils.tensor_utils import add


class RhoFoldCycleONNX(nn.Module):
    """
    ONNX-friendly wrapper for one RhoFold recycling cycle.

    Covers: MSAEmbedder (sans RNA-FM) + RecyclingEmbedder + E2Eformer
            + StructureModule (IPA + AngleResnet, no build_cords/RefineNet)
            + Heads (pLDDT, Dist, SS)

    Inputs:
        msa_tokens:     [1, K, L]        MSA token IDs
        rna_fm_repr:    [1, L, 640]      Pre-computed RNA-FM representations
        recycle_single: [1, L, 256]      Single fea from previous cycle (zeros for cycle 0)
        recycle_pair:   [1, L, L, 128]   Pair fea from previous cycle (zeros for cycle 0)
        recycle_c1:     [1, L, 3]        C1' coords from previous cycle (zeros for cycle 0)
        recycle_mask:   [1]              0.0 for cycle 0, 1.0 for cycle 1+

    Outputs:
        frames:         [1, L, 7]        Final block's quaternion frames (for coordinate construction)
        angles:         [1, L, 6, 2]     Final block's torsion angles (sin/cos)
        plddt_local:    [1, L]           Per-residue confidence
        plddt_global:   [1]              Global confidence
        ss_logits:      [1, 1, L, L]     Secondary structure logits
        dist_p:         [1, 40, L, L]    Distance logits (P atom)
        dist_c4:        [1, 40, L, L]    Distance logits (C4' atom)
        dist_n:         [1, 40, L, L]    Distance logits (N atom)
        out_single:     [1, L, 256]      Single fea for next cycle's recycling
        out_pair:       [1, L, L, 128]   Pair fea for next cycle's recycling
    """

    def __init__(self, rhofold: RhoFold):
        super().__init__()

        # Copy sub-modules from the full model
        self.msa_emb = rhofold.msa_embedder.msa_emb
        self.pair_emb = rhofold.msa_embedder.pair_emb
        self.rna_fm_reduction = rhofold.msa_embedder.rna_fm_reduction

        self.recycle_embnet = rhofold.recycle_embnet
        self.e2eformer = rhofold.e2eformer
        self.structure_module = rhofold.structure_module
        self.dist_head = rhofold.dist_head
        self.ss_head = rhofold.ss_head
        self.plddt_head = rhofold.plddt_head

        # MSA depth from config
        self.msa_depth = rhofold.config.globals.msa_depth  # 128
        self.pad_idx = rhofold.msa_embedder.alphabet.padding_idx

    def forward(
        self,
        msa_tokens: torch.Tensor,
        rna_fm_repr: torch.Tensor,
        recycle_single: torch.Tensor,
        recycle_pair: torch.Tensor,
        recycle_c1: torch.Tensor,
        recycle_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # --- Clip MSA depth ---
        msa_tokens_clip = msa_tokens[:, :self.msa_depth]
        B, K, L = msa_tokens_clip.shape

        # --- MSA embedding (without internal RNA-FM call) ---
        msa_fea = self.msa_emb(msa_tokens_clip)  # [B, K, L, 256]

        # Integrate pre-computed RNA-FM features
        token_repr = rna_fm_repr.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, L, 640]
        msa_fea = self.rna_fm_reduction(
            torch.cat([token_repr, msa_fea], dim=-1)
        )  # [B, K, L, 256]

        pair_fea = self.pair_emb(msa_tokens_clip, t1ds=None, t2ds=None)  # [B, L, L, 128]

        # --- Recycling (masked: no-op for cycle 0) ---
        m_update, z_update = self.recycle_embnet(
            recycle_single, recycle_pair, recycle_c1
        )
        # Apply recycling update, masked by recycle_mask
        rm = recycle_mask.view(-1, 1, 1)  # [B, 1, 1] for broadcasting
        msa_fea[..., 0, :, :] = msa_fea[..., 0, :, :] + m_update * rm
        pair_fea = pair_fea + z_update * rm.unsqueeze(-1)  # [B, 1, 1, 1]

        # --- E2Eformer ---
        msa_mask = (msa_tokens_clip != self.pad_idx).to(dtype=msa_fea.dtype)
        pair_mask = pair_fea.new_ones(pair_fea.shape[:3])

        msa_fea, pair_fea, single_fea = self.e2eformer(
            m=msa_fea,
            z=pair_fea,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            chunk_size=None,
        )

        # --- Structure Module (IPA loop → frames + angles) ---
        # We call the structure module but stop before build_cords
        output = self._structure_forward(single_fea, pair_fea, msa_tokens_clip)

        frames = output['frames']     # [1, L, 7]
        angles = output['angles']     # [1, L, 6, 2]

        # --- Prediction Heads ---
        plddt_local, plddt_global = self.plddt_head(output['single'])
        ss_logits = self.ss_head(pair_fea.float())
        dist_p, dist_c4, dist_n = self.dist_head(pair_fea.float())

        # --- Recycling outputs ---
        out_single = msa_fea[..., 0, :, :]  # [B, L, 256]
        out_pair = pair_fea                   # [B, L, L, 128]

        return (
            frames, angles,
            plddt_local, plddt_global,
            ss_logits, dist_p, dist_c4, dist_n,
            out_single, out_pair,
        )

    def _structure_forward(self, single_fea, pair_fea, msa_tokens):
        """
        Run the StructureModule IPA loop, output frames and angles
        from the LAST block only (matching the original model which uses
        outputs[-1] for coordinate construction).
        """
        sm = self.structure_module

        s = sm.layer_norm_s(single_fea)
        z = sm.layer_norm_z(pair_fea)

        s_initial = s
        s = sm.linear_in(s)

        mask = s.new_ones(s.shape[:-1])

        # Initialize identity rigid (quaternion format)
        from rhofold.utils.rigid_utils import Rigid
        rigids = Rigid.identity(
            s.shape[:-1],
            s.dtype,
            s.device,
            False,  # requires_grad=False for inference
            fmt="quat",
        )

        # Run IPA blocks
        for i in range(sm.no_blocks):
            s = s + sm.ipa(s, z, rigids, mask)
            s = sm.layer_norm_ipa(s)
            s = sm.transition(s)
            rigids = rigids.compose_q_update_vec(sm.bb_update(s))
            unnormalized_angles, angles = sm.angle_resnet(s, s_initial)
            if i != sm.no_blocks - 1:
                rigids = rigids.stop_rot_gradient()

        # Only return the LAST block's outputs
        scaled_rigids = rigids.scale_translation(sm.trans_scale_factor)
        frames = scaled_rigids.to_tensor_7()

        return {
            'frames': frames,
            'angles': angles,
            'single': s,
        }


def export_trunk(ckpt_path: str, output_path: str, K: int = 4, L: int = 68):
    """Export the RhoFold trunk wrapper to ONNX."""
    print("Loading full RhoFold model...")
    model = RhoFold(rhofold_config)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("Creating ONNX wrapper...")
    wrapper = RhoFoldCycleONNX(model)
    wrapper.eval()

    # Dummy inputs
    msa_tokens = torch.randint(0, 9, (1, K, L), dtype=torch.long)
    rna_fm_repr = torch.randn(1, L, 640)
    recycle_single = torch.zeros(1, L, 256)
    recycle_pair = torch.zeros(1, L, L, 128)
    recycle_c1 = torch.zeros(1, L, 3)
    recycle_mask = torch.tensor([0.0])

    print(f"Test forward pass (K={K}, L={L})...")
    with torch.no_grad():
        outputs = wrapper(msa_tokens, rna_fm_repr, recycle_single, recycle_pair, recycle_c1, recycle_mask)
    print(f"  Output shapes: {[o.shape for o in outputs]}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    print(f"Exporting to ONNX (opset 17, fp32)...")
    try:
        torch.onnx.export(
            wrapper,
            (msa_tokens, rna_fm_repr, recycle_single, recycle_pair, recycle_c1, recycle_mask),
            output_path,
            input_names=[
                'msa_tokens', 'rna_fm_repr',
                'recycle_single', 'recycle_pair', 'recycle_c1', 'recycle_mask',
            ],
            output_names=[
                'frames', 'angles',
                'plddt_local', 'plddt_global',
                'ss_logits', 'dist_p', 'dist_c4', 'dist_n',
                'out_single', 'out_pair',
            ],
            dynamic_axes={
                'msa_tokens': {1: 'K', 2: 'L'},
                'rna_fm_repr': {1: 'L'},
                'recycle_single': {1: 'L'},
                'recycle_pair': {1: 'L', 2: 'L'},
                'recycle_c1': {1: 'L'},
                'frames': {1: 'L'},
                'angles': {1: 'L'},
                'plddt_local': {1: 'L'},
                'ss_logits': {2: 'L', 3: 'L'},
                'dist_p': {2: 'L', 3: 'L'},
                'dist_c4': {2: 'L', 3: 'L'},
                'dist_n': {2: 'L', 3: 'L'},
                'out_single': {1: 'L'},
                'out_pair': {1: 'L', 2: 'L'},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Exported: {output_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_trunk(ckpt_path: str, onnx_path: str, K: int = 4, L: int = 68):
    """Validate ONNX output matches PyTorch output."""
    import onnxruntime as ort

    print("\nLoading PyTorch model for validation...")
    model = RhoFold(rhofold_config)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    wrapper = RhoFoldCycleONNX(model)
    wrapper.eval()

    torch.manual_seed(42)
    msa_tokens = torch.randint(0, 9, (1, K, L), dtype=torch.long)
    rna_fm_repr = torch.randn(1, L, 640)
    recycle_single = torch.zeros(1, L, 256)
    recycle_pair = torch.zeros(1, L, L, 128)
    recycle_c1 = torch.zeros(1, L, 3)
    recycle_mask = torch.tensor([0.0])

    print("Running PyTorch forward...")
    with torch.no_grad():
        pt_outputs = wrapper(msa_tokens, rna_fm_repr, recycle_single, recycle_pair, recycle_c1, recycle_mask)

    print("Running ONNX Runtime forward...")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {
        'msa_tokens': msa_tokens.numpy(),
        'rna_fm_repr': rna_fm_repr.numpy(),
        'recycle_single': recycle_single.numpy(),
        'recycle_pair': recycle_pair.numpy(),
        'recycle_c1': recycle_c1.numpy(),
        'recycle_mask': recycle_mask.numpy(),
    }
    ort_outputs = sess.run(None, ort_inputs)

    output_names = [
        'frames', 'angles', 'plddt_local', 'plddt_global',
        'ss_logits', 'dist_p', 'dist_c4', 'dist_n',
        'out_single', 'out_pair',
    ]

    print(f"\nValidation (K={K}, L={L}):")
    all_pass = True
    for name, pt_out, ort_out in zip(output_names, pt_outputs, ort_outputs):
        pt_np = pt_out.numpy()
        diff = np.max(np.abs(pt_np - ort_out))
        status = "PASS" if diff < 1e-3 else ("WARN" if diff < 1e-2 else "FAIL")
        if status == "FAIL":
            all_pass = False
        print(f"  {name:15s} shape={str(pt_np.shape):25s} max_diff={diff:.2e}  {status}")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description='Export RhoFold trunk to ONNX')
    parser.add_argument('--ckpt', default='./checkpoints/rhofold_pretrained_params.pt')
    parser.add_argument('--output', default='./webgpu/models/rhofold_trunk.onnx')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--K', type=int, default=4, help='MSA depth for dummy export input')
    parser.add_argument('--L', type=int, default=68, help='Sequence length for dummy export input')
    args = parser.parse_args()

    success = export_trunk(args.ckpt, args.output, K=args.K, L=args.L)

    if success and args.validate:
        validate_trunk(args.ckpt, args.output, K=args.K, L=args.L)


if __name__ == '__main__':
    main()
