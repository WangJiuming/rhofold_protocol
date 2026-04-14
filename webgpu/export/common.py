"""
Common utilities for ONNX export of RhoFold modules.

Patches problematic functions that produce ONNX-unfriendly output
(e.g., negative Transpose perm indices).
"""

import torch
import torch.nn as nn
import numpy as np


def patch_permute_final_dims():
    """
    Monkey-patch permute_final_dims to use only non-negative indices.
    ONNX Runtime rejects negative perm values in Transpose nodes.
    """
    import rhofold.utils.tensor_utils as tu

    _original = tu.permute_final_dims

    def _safe_permute_final_dims(tensor: torch.Tensor, inds):
        ndim = len(tensor.shape)
        n_inds = len(inds)
        first_inds = list(range(ndim - n_inds))
        perm = first_inds + [ndim - n_inds + i for i in inds]
        return tensor.permute(perm)

    tu.permute_final_dims = _safe_permute_final_dims

    # Also patch the import in modules that import it directly
    import rhofold.model.structure_module as sm
    if hasattr(sm, 'permute_final_dims'):
        sm.permute_final_dims = _safe_permute_final_dims

    import rhofold.model.primitives as pr
    if hasattr(pr, 'permute_final_dims'):
        pr.permute_final_dims = _safe_permute_final_dims

    return _original


def load_rhofold(ckpt_path: str):
    """Load full RhoFold model from checkpoint."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rhofold'))

    from rhofold.config import rhofold_config
    from rhofold.rhofold import RhoFold

    model = RhoFold(rhofold_config)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, rhofold_config


def validate_outputs(pt_outputs, ort_outputs, names, threshold=1e-3):
    """Compare PyTorch and ONNX Runtime outputs."""
    all_pass = True
    for name, pt_out, ort_out in zip(names, pt_outputs, ort_outputs):
        pt_np = pt_out.numpy() if isinstance(pt_out, torch.Tensor) else pt_out
        diff = np.max(np.abs(pt_np - ort_out))
        status = "PASS" if diff < threshold else ("WARN" if diff < threshold * 10 else "FAIL")
        if status == "FAIL":
            all_pass = False
        print(f"  {name:20s} shape={str(pt_np.shape):25s} max_diff={diff:.2e}  {status}")
    return all_pass
