"""
Export RNA-FM (frozen ESM-1b RNA transformer) to ONNX format.

Usage:
    conda activate rhofold_protocol
    cd <repo_root>
    python webgpu/export/export_rna_fm.py \
        --ckpt ./checkpoints/rhofold_pretrained_params.pt \
        --output ./webgpu/models/rna_fm.onnx \
        --validate
"""

import argparse
import sys
import os

import numpy as np
import torch
import torch.nn as nn

# Add project root to path so rhofold imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rhofold'))

from rhofold.model.rna_fm.pretrained import esm1b_rna_t12
from rhofold.model.rna_fm.data import Alphabet


class RNAFM_ONNX(nn.Module):
    """
    Thin wrapper around ProteinBertModel that returns only the layer-12
    representation as a single tensor, making it ONNX-exportable.

    Input:  tokens [B, L] — RNA-FM token IDs (no CLS/EOS)
    Output: repr   [B, L, 640] — per-residue embeddings from final layer
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # --- Inline a simplified ProteinBertModel.forward ---
        # We skip: token_dropout, attn_weights, contacts, lm_head logits.
        # We only need representations at the final layer.

        m = self.model

        padding_mask = tokens.eq(m.padding_idx)  # [B, L]

        x = m.embed_scale * m.embed_tokens(tokens)
        x = x + m.embed_positions(tokens)

        # ESM-1b path: pre-LN + mask padding
        x = m.emb_layer_norm_before(x)
        x = x * (1 - padding_mask.unsqueeze(-1).to(x.dtype))

        # (B, L, E) -> (L, B, E) for transformer layers
        x = x.transpose(0, 1)

        # Always pass padding_mask (avoid data-dependent `if None` branch)
        for layer in m.layers:
            x, _attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=False,
            )

        # Post-LN
        x = m.emb_layer_norm_after(x)

        # (L, B, E) -> (B, L, E)
        x = x.transpose(0, 1)

        return x


def extract_rna_fm_weights(ckpt_path: str) -> dict:
    """Extract RNA-FM state dict from the full RhoFold checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model']

    prefix = 'msa_embedder.rna_fm.'
    rna_fm_state = {}
    for k, v in state.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            rna_fm_state[new_key] = v

    print(f"Extracted {len(rna_fm_state)} RNA-FM tensors from checkpoint")
    return rna_fm_state


def export_onnx(wrapper: nn.Module, output_path: str, seq_len: int = 68):
    """Export the RNA-FM wrapper to ONNX."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Dummy input: RNA-FM tokens without CLS/EOS
    dummy_tokens = torch.randint(4, 20, (1, seq_len), dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy_tokens,),
        output_path,
        input_names=['tokens'],
        output_names=['representations'],
        dynamic_axes={
            'tokens': {0: 'batch', 1: 'seq_len'},
            'representations': {0: 'batch', 1: 'seq_len'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {output_path}")

    # Print file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX file size: {size_mb:.1f} MB")


def validate(wrapper: nn.Module, onnx_path: str, seq_len: int = 68):
    """Validate ONNX output matches PyTorch output."""
    import onnxruntime as ort

    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    tokens = torch.randint(4, 20, (1, seq_len), dtype=torch.long)

    # PyTorch reference
    wrapper.eval()
    with torch.no_grad():
        pt_out = wrapper(tokens).numpy()

    # ONNX Runtime
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {'tokens': tokens.numpy()})[0]

    # Compare
    max_diff = np.max(np.abs(pt_out - ort_out))
    mean_diff = np.mean(np.abs(pt_out - ort_out))
    print(f"\nValidation (seq_len={seq_len}):")
    print(f"  PyTorch output shape: {pt_out.shape}")
    print(f"  ONNX output shape:    {ort_out.shape}")
    print(f"  Max absolute diff:    {max_diff:.2e}")
    print(f"  Mean absolute diff:   {mean_diff:.2e}")

    if max_diff < 1e-4:
        print("  PASS: max diff < 1e-4")
    elif max_diff < 1e-3:
        print("  WARN: max diff < 1e-3 (acceptable for fp32)")
    else:
        print("  FAIL: max diff >= 1e-3")

    return max_diff


def validate_with_real_data(wrapper: nn.Module, onnx_path: str):
    """Validate using the 3owz_A example data."""
    import onnxruntime as ort
    from rhofold.utils.alphabet import get_features

    fasta = './data/rhofold/3owz_A/3owz_A.fasta'
    msa = './data/rhofold/3owz_A/3owz_A.afa'

    if not os.path.exists(fasta):
        print("\nSkipping real-data validation (example data not found)")
        return None

    data = get_features(fasta, msa)
    rna_fm_tokens = data['rna_fm_tokens']  # [1, L]

    print(f"\nReal-data validation (3owz_A, L={rna_fm_tokens.shape[1]}):")

    # PyTorch
    wrapper.eval()
    with torch.no_grad():
        pt_out = wrapper(rna_fm_tokens).numpy()

    # ONNX
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {'tokens': rna_fm_tokens.numpy()})[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    mean_diff = np.mean(np.abs(pt_out - ort_out))
    print(f"  Output shape: {pt_out.shape}")
    print(f"  Max absolute diff:  {max_diff:.2e}")
    print(f"  Mean absolute diff: {mean_diff:.2e}")

    if max_diff < 1e-4:
        print("  PASS")
    elif max_diff < 1e-3:
        print("  WARN (acceptable)")
    else:
        print("  FAIL")

    return max_diff


def main():
    parser = argparse.ArgumentParser(description='Export RNA-FM to ONNX')
    parser.add_argument('--ckpt', default='./checkpoints/rhofold_pretrained_params.pt',
                        help='Path to RhoFold checkpoint')
    parser.add_argument('--output', default='./webgpu/models/rna_fm.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--validate', action='store_true',
                        help='Run numerical validation after export')
    parser.add_argument('--seq-len', type=int, default=68,
                        help='Sequence length for dummy input during export')
    args = parser.parse_args()

    # 1. Create RNA-FM model and load weights
    print("Creating RNA-FM model...")
    model, alphabet = esm1b_rna_t12()
    model.eval()

    print("Loading weights from checkpoint...")
    rna_fm_state = extract_rna_fm_weights(args.ckpt)
    model.load_state_dict(rna_fm_state, strict=True)

    for param in model.parameters():
        param.detach_()

    # 2. Wrap for ONNX
    wrapper = RNAFM_ONNX(model)
    wrapper.eval()

    # 3. Export
    print(f"\nExporting to ONNX (opset 17, fp32)...")
    export_onnx(wrapper, args.output, seq_len=args.seq_len)

    # 4. Validate
    if args.validate:
        validate(wrapper, args.output, seq_len=args.seq_len)
        validate_with_real_data(wrapper, args.output)


if __name__ == '__main__':
    main()
