"""
Export StructureModule (IPA + AngleResnet) + prediction heads to ONNX.

This module outputs frames and angles (NOT coordinates — coordinate
construction is done in JS). Also outputs pLDDT, SS, and distance heads.

Input:  single_fea [1,L,384], pair_fea [1,L,L,128]
Output: frames [1,L,7], angles [1,L,6,2],
        plddt_local [1,L], plddt_global [1],
        ss_logits [1,1,L,L],
        dist_p/c4/n [1,40,L,L] each

Usage:
    cd <repo_root>
    python webgpu/export/export_structure.py --validate
"""

import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rhofold'))

from common import patch_permute_final_dims, load_rhofold, validate_outputs
patch_permute_final_dims()

from rhofold.utils.rigid_utils import Rigid


class StructureHeadsONNX(nn.Module):
    def __init__(self, rhofold):
        super().__init__()
        sm = rhofold.structure_module
        self.layer_norm_s = sm.layer_norm_s
        self.layer_norm_z = sm.layer_norm_z
        self.linear_in = sm.linear_in
        self.ipa = sm.ipa
        self.layer_norm_ipa = sm.layer_norm_ipa
        self.transition = sm.transition
        self.bb_update = sm.bb_update
        self.angle_resnet = sm.angle_resnet
        self.no_blocks = sm.no_blocks
        self.trans_scale_factor = sm.trans_scale_factor

        self.dist_head = rhofold.dist_head
        self.ss_head = rhofold.ss_head
        self.plddt_head = rhofold.plddt_head

    def forward(self, single_fea, pair_fea):
        s = self.layer_norm_s(single_fea)
        z = self.layer_norm_z(pair_fea)

        s_initial = s
        s = self.linear_in(s)

        mask = s.new_ones(s.shape[:-1])

        rigids = Rigid.identity(
            s.shape[:-1], s.dtype, s.device, False, fmt="quat",
        )

        for i in range(self.no_blocks):
            s = s + self.ipa(s, z, rigids, mask)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)
            rigids = rigids.compose_q_update_vec(self.bb_update(s))
            _unnorm, angles = self.angle_resnet(s, s_initial)
            if i != self.no_blocks - 1:
                rigids = rigids.stop_rot_gradient()

        scaled_rigids = rigids.scale_translation(self.trans_scale_factor)
        frames = scaled_rigids.to_tensor_7()

        # Heads
        plddt_local, plddt_global = self.plddt_head(s)
        ss_logits = self.ss_head(pair_fea.float())
        dist_p, dist_c4, dist_n = self.dist_head(pair_fea.float())

        return frames, angles, plddt_local, plddt_global, ss_logits, dist_p, dist_c4, dist_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='./checkpoints/rhofold_pretrained_params.pt')
    parser.add_argument('--output', default='./webgpu/models/structure_heads.onnx')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--L', type=int, default=68)
    args = parser.parse_args()

    model, cfg = load_rhofold(args.ckpt)
    wrapper = StructureHeadsONNX(model)
    wrapper.eval()

    L = args.L
    single_fea = torch.randn(1, L, 384)
    pair_fea = torch.randn(1, L, L, 128)

    print(f"Test forward (L={L})...")
    with torch.no_grad():
        outputs = wrapper(single_fea, pair_fea)
    print(f"  Output shapes: {[o.shape for o in outputs]}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print("Exporting to ONNX...")
    try:
        torch.onnx.export(
            wrapper,
            (single_fea, pair_fea),
            args.output,
            input_names=['single_fea', 'pair_fea'],
            output_names=['frames', 'angles', 'plddt_local', 'plddt_global',
                         'ss_logits', 'dist_p', 'dist_c4', 'dist_n'],
            dynamic_axes={
                'single_fea': {1: 'L'}, 'pair_fea': {1: 'L', 2: 'L'},
                'frames': {1: 'L'}, 'angles': {1: 'L'},
                'plddt_local': {1: 'L'},
                'ss_logits': {2: 'L', 3: 'L'},
                'dist_p': {2: 'L', 3: 'L'},
                'dist_c4': {2: 'L', 3: 'L'},
                'dist_n': {2: 'L', 3: 'L'},
            },
            opset_version=17, do_constant_folding=True,
        )
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"Exported: {args.output} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"Export FAILED: {e}")
        import traceback; traceback.print_exc()
        return

    if args.validate:
        import onnxruntime as ort
        torch.manual_seed(42)
        single_fea = torch.randn(1, L, 384)
        pair_fea = torch.randn(1, L, L, 128)

        with torch.no_grad():
            pt_out = wrapper(single_fea, pair_fea)

        sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
        ort_out = sess.run(None, {
            'single_fea': single_fea.numpy(),
            'pair_fea': pair_fea.numpy(),
        })
        print(f"\nValidation:")
        names = ['frames', 'angles', 'plddt_local', 'plddt_global',
                 'ss_logits', 'dist_p', 'dist_c4', 'dist_n']
        validate_outputs(pt_out, ort_out, names)


if __name__ == '__main__':
    main()
