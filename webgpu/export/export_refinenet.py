"""
Export RefineNet (E(n)-equivariant GNN for coordinate refinement) to ONNX.

Usage:
    conda activate rhofold_protocol
    cd <repo_root>
    python webgpu/export/export_refinenet.py \
        --ckpt ./checkpoints/rhofold_pretrained_params.pt \
        --output ./webgpu/models/refinenet.onnx \
        --validate
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rhofold'))

from rhofold.config import rhofold_config
from rhofold.rhofold import RhoFold
from rhofold.model.structure_module import CoorsNorm


class CoorsNormONNX(nn.Module):
    """
    Drop-in replacement for CoorsNorm that avoids LayerNorm(1).

    The original CoorsNorm computes:
        norm = ||coors||_2 (keepdim)
        normed = coors / max(norm, eps)
        phase = LayerNorm(1)(norm)   # weight=1, bias=b → output is just b
        return phase * normed

    LayerNorm(1) on a scalar always returns weight * 0 + bias = bias.
    We replace it with a simple multiply by the learned bias parameter.
    """
    def __init__(self, orig: CoorsNorm):
        super().__init__()
        self.eps = orig.eps
        self.bias = nn.Parameter(orig.fn.bias.clone())

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return self.bias * normed_coors


def _patch_coors_norm(refinenet):
    """Replace all CoorsNorm instances with ONNX-friendly version."""
    for name, module in refinenet.named_modules():
        if isinstance(module, CoorsNorm):
            replacement = CoorsNormONNX(module)
            # Navigate to parent and replace
            parts = name.split('.')
            parent = refinenet
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], replacement)


class RefineNetONNX(nn.Module):
    """
    ONNX wrapper for RefineNet.

    Input:
        first_msa_row: [1, L]  Token IDs from the first MSA row
        coords:        [1, L*23, 3]  All-atom coordinates from build_cords

    Output:
        refined_coords: [1, L*23, 3]  Refined coordinates
    """

    def __init__(self, refinenet):
        super().__init__()
        self.refinenet = refinenet

    def forward(self, first_msa_row: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # RefineNet.forward expects tokens as [B, K, L] and does tokens[:, 0, :]
        # We pass the first row directly and reshape to match
        tokens = first_msa_row.unsqueeze(1)  # [1, 1, L]
        return self.refinenet(tokens, coords)


def export_refinenet(ckpt_path: str, output_path: str, L: int = 68):
    """Export RefineNet to ONNX."""
    print("Loading full RhoFold model...")
    model = RhoFold(rhofold_config)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if model.structure_module.refinenet is None:
        print("RefineNet is disabled in config. Nothing to export.")
        return False

    refinenet = model.structure_module.refinenet
    _patch_coors_norm(refinenet)
    wrapper = RefineNetONNX(refinenet)
    wrapper.eval()

    # Dummy inputs
    first_msa_row = torch.randint(0, 9, (1, L), dtype=torch.long)
    coords = torch.randn(1, L * 23, 3)

    print(f"Test forward pass (L={L})...")
    with torch.no_grad():
        out = wrapper(first_msa_row, coords)
    print(f"  Output shape: {out.shape}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    print(f"Exporting to ONNX (opset 17, fp32)...")
    try:
        torch.onnx.export(
            wrapper,
            (first_msa_row, coords),
            output_path,
            input_names=['first_msa_row', 'coords'],
            output_names=['refined_coords'],
            dynamic_axes={
                'first_msa_row': {1: 'L'},
                'coords': {1: 'L_times_23'},
                'refined_coords': {1: 'L_times_23'},
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


def validate_refinenet(ckpt_path: str, onnx_path: str, L: int = 68):
    """Validate ONNX output matches PyTorch output."""
    import onnxruntime as ort

    model = RhoFold(rhofold_config)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    wrapper = RefineNetONNX(model.structure_module.refinenet)
    _patch_coors_norm(wrapper)
    wrapper.eval()

    torch.manual_seed(42)
    first_msa_row = torch.randint(0, 9, (1, L), dtype=torch.long)
    coords = torch.randn(1, L * 23, 3)

    with torch.no_grad():
        pt_out = wrapper(first_msa_row, coords).numpy()

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {
        'first_msa_row': first_msa_row.numpy(),
        'coords': coords.numpy(),
    })[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    mean_diff = np.mean(np.abs(pt_out - ort_out))
    print(f"\nRefineNet validation (L={L}):")
    print(f"  Output shape: {pt_out.shape}")
    print(f"  Max diff:  {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    status = "PASS" if max_diff < 1e-3 else ("WARN" if max_diff < 1e-2 else "FAIL")
    print(f"  {status}")
    return max_diff < 1e-2


def main():
    parser = argparse.ArgumentParser(description='Export RefineNet to ONNX')
    parser.add_argument('--ckpt', default='./checkpoints/rhofold_pretrained_params.pt')
    parser.add_argument('--output', default='./webgpu/models/refinenet.onnx')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--L', type=int, default=68)
    args = parser.parse_args()

    success = export_refinenet(args.ckpt, args.output, L=args.L)
    if success and args.validate:
        validate_refinenet(args.ckpt, args.output, L=args.L)


if __name__ == '__main__':
    main()
