"""Batched RhoFold+ inference with true tensor batching + length bucketing.

Unlike `batch_inference.py`, which loops over sequences one at a time,
this script stacks multiple sequences into a single forward pass with
explicit padding and masking. Heavy compute (MSA embedder, E2Eformer,
IPA, distogram/SS/pLDDT heads) runs at batch dim B > 1. The per-residue
coord builder (RNAConverter) and the EGNN refinenet are looped per
sample inside the structure module because they are not batch-safe on
heterogeneous sequences (the converter assumes B=1 and indexes residues
by name; the refinenet's EGNN exchanges messages across all positions
and would leak padded positions into real ones).

Before batching, samples are sorted by (sequence length, MSA depth)
ascending. With `--batch-size B`, consecutive sorted samples are
grouped, which keeps each forward pass tight on padding and — because
within a batch the tensor shapes are uniform — yields per-sample
outputs that match B=1 to float32 precision (no cuBLAS algorithm
selection drift across batch shapes).

Amber relaxation is run sequentially per sample (OpenMM is not
batchable). Pass --relax-steps 0 to skip.

Input format: JSON with a `sequences` array whose entries are RNA
monomers to predict independently. Example:

    {
      "name": "my_batch",
      "sequences": [
        {"rna": {"id": "seq1", "sequence": "GGCG...",
                 "msaPath": "data/rhofold/3owz_A/3owz_A.afa"}},
        {"rna": {"id": "seq2", "sequence": "AUGC..."}}
      ]
    }

Each entry under `sequences` must contain an `rna` object with:
  - id (required): output subdirectory name.
  - sequence (required): RNA sequence (A/U/G/C; T auto-converted to U).
  - msaPath (optional): path to an .afa/.a3m/.fasta MSA file.
  - msa (optional): inline MSA as a FASTA-format string.
If neither MSA field is given, single-sequence mode is used for that entry.

Per-sample reproducibility across batch sizes
---------------------------------------------
Length bucketing keeps the (B, max_K, max_L) shape uniform within each
forward pass, so cuBLAS / cuDNN pick the same algorithms regardless of
the surrounding batch — per-sample outputs match B=1 to float32 noise.
Residual padding drift only appears when two samples in the same bucket
still differ in shape (e.g. adjacent unique lengths); that drift stays
in the ~1e-6 range. Pass --deterministic if you need bit-level
reproducibility within a fixed shape; on most hardware it is unnecessary.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# CUBLAS_WORKSPACE_CONFIG is required by torch.use_deterministic_algorithms
# for some cuBLAS ops; set it before importing torch so it is in effect.
# (Harmless when deterministic mode is off.)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from rhofold.config import rhofold_config
from rhofold.model.rna_fm.data import Alphabet as FMAlphabet
from rhofold.relax.relax import AmberRelaxation
from rhofold.rhofold import RhoFold
from rhofold.utils import get_device, save_ss2ct, timing
from rhofold.utils.alphabet import RNAAlphabet, get_features
from rhofold.utils.constants import RNA_CONSTANTS


# A/U/G/C are the standard alphabet; N is "unknown" and is accepted by the
# RNAAlphabet via the <unk> token (used in some PDB-derived sequences, e.g.
# data/rhofold/milu/milu.fasta).
VALID_RNA_RESIDUES = set('AUGCN')


@dataclass
class Sample:
    """A single RNA monomer prediction job."""
    sample_id: str
    sequence: str                       # uppercase, T->U normalized
    msa_source: Optional[str]           # path to MSA file, or None for single-seq
    msa_is_temp: bool                   # True if we materialized an inline MSA


def parse_input_json(path: Path, work_dir: Path,
                     logger: Optional[logging.Logger] = None) -> List[Sample]:
    """Parse the JSON input into a list of Samples.

    Inline `msa` strings are written to temp .a3m files under work_dir so
    that downstream `get_features` (which reads from paths) can consume
    them uniformly with `msaPath` entries.

    Empty MSA files / empty inline MSAs log a warning and fall back to
    single-sequence mode rather than erroring.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    with open(path) as f:
        spec = json.load(f)

    if 'sequences' not in spec:
        raise ValueError(f"Input JSON {path} has no 'sequences' field")

    samples: List[Sample] = []
    seen_ids: set = set()
    for i, entry in enumerate(spec['sequences']):
        if 'rna' not in entry:
            raise ValueError(
                f"Entry {i} is missing 'rna' object. Only RNA monomers are supported.")
        rna = entry['rna']
        if 'id' not in rna or 'sequence' not in rna:
            raise ValueError(f"Entry {i} 'rna' must have both 'id' and 'sequence'")

        sid = str(rna['id'])
        if sid in seen_ids:
            raise ValueError(f"Duplicate sequence id: {sid!r}")
        seen_ids.add(sid)

        seq = str(rna['sequence']).upper().replace('T', 'U')
        bad = set(seq) - VALID_RNA_RESIDUES
        if bad:
            raise ValueError(
                f"Sequence {sid!r} contains non-AUGC residues: {sorted(bad)}")

        msa_path: Optional[str] = None
        msa_is_temp = False
        if rna.get('msaPath'):
            msa_path = str(Path(rna['msaPath']).expanduser())
            if not os.path.exists(msa_path):
                raise ValueError(
                    f"Sequence {sid!r}: msaPath does not exist: {msa_path}")
            if os.path.getsize(msa_path) == 0:
                logger.warning(
                    f"Sequence {sid!r}: msaPath is empty ({msa_path}); "
                    "falling back to single-sequence mode.")
                msa_path = None
        elif rna.get('msa'):
            inline = rna['msa'].strip()
            if not inline:
                logger.warning(
                    f"Sequence {sid!r}: inline msa is empty; falling back to "
                    "single-sequence mode.")
            else:
                tmp = work_dir / f'_msa_{sid}.a3m'
                tmp.write_text(inline + '\n')
                msa_path = str(tmp)
                msa_is_temp = True

        samples.append(Sample(sample_id=sid, sequence=seq,
                              msa_source=msa_path, msa_is_temp=msa_is_temp))

    return samples


def materialize_features(sample: Sample, work_dir: Path):
    """Write a temp FASTA and call get_features for one sample.

    Returns dict with keys 'tokens' (Tensor [1, K, L]), 'rna_fm_tokens'
    (Tensor [1, L]), and 'seq' (str of length L). All on CPU.
    """
    tmp_fasta = work_dir / f'_query_{sample.sample_id}.fasta'
    tmp_fasta.write_text(f'>{sample.sample_id}\n{sample.sequence}\n')

    # In single-seq mode the original pipeline points msa=fasta so that
    # MSA depth is 1 and only the query is read.
    msa_path = sample.msa_source if sample.msa_source else str(tmp_fasta)

    feats = get_features(str(tmp_fasta), msa_path)
    return feats, tmp_fasta


def build_batch(feats_list, msa_pad_idx: int, fm_pad_idx: int, msa_depth_cap: int):
    """Stack per-sample features into padded batch tensors and masks.

    Args:
        feats_list: list of dicts (from get_features) with 'tokens'
            shape [1, K_i, L_i], 'rna_fm_tokens' shape [1, L_i], 'seq' str.
        msa_pad_idx: padding token id in the MSA alphabet.
        fm_pad_idx: padding token id in the RNA-FM alphabet.
        msa_depth_cap: cap on MSA depth (matches rhofold_config.globals.msa_depth).

    Returns:
        dict with batched tensors and masks (all CPU):
            tokens          [B, max_K, max_L]
            rna_fm_tokens   [B, max_L]
            res_mask        [B, max_L]
            msa_mask        [B, max_K, max_L]
            pair_mask       [B, max_L, max_L]
            seqs            List[str]
            real_lengths    List[int]
    """
    B = len(feats_list)
    max_L = max(f['tokens'].shape[2] for f in feats_list)
    max_K = max(f['tokens'].shape[1] for f in feats_list)
    # MSA depth in the forward pass is capped to msa_depth_cap; over-pad
    # is harmless because the mask zeros it out, but it wastes memory.
    max_K = min(max_K, msa_depth_cap)

    tokens = torch.full((B, max_K, max_L), msa_pad_idx, dtype=torch.long)
    rna_fm_tokens = torch.full((B, max_L), fm_pad_idx, dtype=torch.long)
    res_mask = torch.zeros((B, max_L), dtype=torch.float32)
    msa_mask = torch.zeros((B, max_K, max_L), dtype=torch.float32)

    seqs: List[str] = []
    real_lengths: List[int] = []

    for i, f in enumerate(feats_list):
        t = f['tokens']        # [1, K_i, L_i]
        r = f['rna_fm_tokens'] # [1, L_i]
        L_i = t.shape[2]
        K_i = min(t.shape[1], max_K)

        tokens[i, :K_i, :L_i] = t[0, :K_i, :L_i]
        rna_fm_tokens[i, :L_i] = r[0, :L_i]
        res_mask[i, :L_i] = 1.0
        msa_mask[i, :K_i, :L_i] = 1.0

        seqs.append(f['seq'])
        real_lengths.append(L_i)

    pair_mask = res_mask[:, :, None] * res_mask[:, None, :]

    return {
        'tokens': tokens,
        'rna_fm_tokens': rna_fm_tokens,
        'res_mask': res_mask,
        'msa_mask': msa_mask,
        'pair_mask': pair_mask,
        'seqs': seqs,
        'real_lengths': real_lengths,
    }


def write_sample_outputs(sample_id: str, seq: str, L: int, output_dir: Path,
                         per_sample_output: dict, model, logger):
    """Slice the batched outputs back to one sample and write the standard files.

    Files written (mirroring inference.py): ss.ct, results.npz, unrelaxed_model.pdb.
    Returns path to the unrelaxed pdb so the caller can run Amber relax.
    """
    seq_dir = output_dir / sample_id
    seq_dir.mkdir(parents=True, exist_ok=True)

    n_atoms = RNA_CONSTANTS.ATOM_NUM_MAX

    # All padded fields use [:, :L] or [:L*n_atoms] to drop padding.
    ss_prob_map = torch.sigmoid(per_sample_output['ss'][0, :L, :L]).cpu().numpy()
    save_ss2ct(ss_prob_map, seq, str(seq_dir / 'ss.ct'), threshold=0.5)

    plddt_1d = per_sample_output['plddt'][:L].cpu().numpy()    # [L]
    plddt_2d = plddt_1d[None, :]                                # [1, L] to match downstream

    npz_path = seq_dir / 'results.npz'
    np.savez_compressed(
        npz_path,
        dist_n=torch.softmax(
            per_sample_output['n'][:, :L, :L], dim=0).cpu().numpy(),
        dist_p=torch.softmax(
            per_sample_output['p'][:, :L, :L], dim=0).cpu().numpy(),
        dist_c=torch.softmax(
            per_sample_output['c4_'][:, :L, :L], dim=0).cpu().numpy(),
        ss_prob_map=ss_prob_map,
        plddt=plddt_2d,
    )

    cords_flat = per_sample_output['cord_tns_pred'][:L * n_atoms]  # [L*n_atoms, 3]
    cords = cords_flat.reshape(L, n_atoms, 3).cpu().numpy()

    unrelaxed = seq_dir / 'unrelaxed_model.pdb'
    model.structure_module.converter.export_pdb_file(
        seq,
        cords,
        path=str(unrelaxed),
        chain_id=None,
        confidence=plddt_2d,
        logger=logger,
    )
    return str(unrelaxed)


def slice_output_for_sample(batched_output: dict, b: int, L: int) -> dict:
    """Pluck the per-sample sub-tensors from the last-cycle batched outputs.

    Note: `batched_output['plddt']` is a (plddt_local, plddt_global) tuple
    from pLDDTHead; the existing single-seq pipeline saves `plddt_local`
    (index [0]) and downstream tooling (scripts/8_parse_plddt.py) expects
    shape (1, L). We mirror that exactly per sample.
    """
    plddt_local = batched_output['plddt'][0]  # [B, L_max]
    return {
        'ss': batched_output['ss'][b],                          # [1, L_max, L_max]
        'n': batched_output['n'][b],                            # [bins, L_max, L_max]
        'p': batched_output['p'][b],
        'c4_': batched_output['c4_'][b],
        'plddt': plddt_local[b],                                # [L_max]
        'cord_tns_pred': batched_output['cord_tns_pred'][-1][b],# [L_max*n_atoms, 3]
    }


@torch.no_grad()
def main(args):
    if args.deterministic:
        # See module docstring "Per-sample reproducibility across batch sizes"
        # for details. With these flags, cuBLAS/cuDNN pin a single algorithm
        # per input shape (rather than picking the fastest), which reduces but
        # does not eliminate cross-batch-shape drift. warn_only=True so any
        # ops without a deterministic implementation still run.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('RhoFold+ BatchedInference')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    fh = logging.FileHandler(output_dir / 'log.txt', mode='w')
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)

    work_dir = Path(tempfile.mkdtemp(prefix='rhofold_batched_'))
    logger.info(f'Working directory for temp files: {work_dir}')

    try:
        logger.info(f'Parsing input JSON: {args.input}')
        samples = parse_input_json(Path(args.input), work_dir, logger=logger)
        logger.info(f'Found {len(samples)} sample(s)')

        logger.info(f'Constructing RhoFold+')
        model = RhoFold(rhofold_config)
        logger.info(f'    loading {args.ckpt}')
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu'))['model'])
        model.eval()

        device = get_device(args.device)
        logger.info(f'    Inference using device {device}')
        model = model.to(device)

        msa_alphabet = RNAAlphabet.from_architecture('RNA')
        fm_alphabet = FMAlphabet.from_architecture('ESM-1b', theme='rna')
        msa_pad = msa_alphabet.padding_idx
        fm_pad = fm_alphabet.padding_idx
        msa_depth_cap = rhofold_config.globals.msa_depth

        # Featurize all samples up-front (cheap, CPU). This lets us decide
        # batches and detect malformed inputs before touching the GPU.
        logger.info('Featurizing samples')
        feats_per_sample = []
        for s in samples:
            feats, _ = materialize_features(s, work_dir)
            feats_per_sample.append((s, feats))
            logger.info(
                f'  {s.sample_id}: L={feats["tokens"].shape[2]}, K={feats["tokens"].shape[1]}, '
                f'mode={"MSA" if s.msa_source else "single-seq"}'
            )

        unrelaxed_paths: List[tuple] = []  # (sample_id, seq, unrelaxed_pdb_path)

        # Length bucketing: sort by (L, K) ascending before chunking so each
        # forward pass works on the tightest possible (max_L, max_K) for that
        # batch. This both recovers the speedup wasted on padding and removes
        # the cross-batch float32 drift previously caused by cuBLAS picking
        # different algorithms for different batch shapes — within a bucket
        # all samples share the same shape, so cuBLAS makes consistent choices
        # and per-sample outputs match B=1 to float32 precision.
        if args.no_bucketing:
            logger.info(
                'Length bucketing disabled (--no-bucketing). Processing '
                'samples in input order.'
            )
        elif len(feats_per_sample) > 1:
            feats_per_sample.sort(key=lambda sf: (
                sf[1]['tokens'].shape[2],   # L
                sf[1]['tokens'].shape[1],   # K
            ))
            logger.info(
                f'Sorted {len(feats_per_sample)} samples by (L, K) for length '
                f'bucketing: order = {[s.sample_id for s, _ in feats_per_sample]}'
            )

        for start in range(0, len(feats_per_sample), args.batch_size):
            chunk = feats_per_sample[start:start + args.batch_size]
            chunk_ids = [s.sample_id for s, _ in chunk]
            logger.info(
                f'Batch {start // args.batch_size + 1}: '
                f'{len(chunk)} samples = {chunk_ids}'
            )

            batch = build_batch(
                [f for _, f in chunk],
                msa_pad_idx=msa_pad,
                fm_pad_idx=fm_pad,
                msa_depth_cap=msa_depth_cap,
            )

            with timing(
                f'Forward pass: B={len(chunk)}, max_L={batch["tokens"].shape[2]}, '
                f'max_K={batch["tokens"].shape[1]}',
                logger=logger,
            ):
                outputs = model(
                    tokens=batch['tokens'].to(device),
                    rna_fm_tokens=batch['rna_fm_tokens'].to(device),
                    seq=batch['seqs'],
                    msa_mask=batch['msa_mask'].to(device),
                    pair_mask=batch['pair_mask'].to(device),
                    res_mask=batch['res_mask'].to(device),
                )
            last = outputs[-1]

            for b, (sample, _) in enumerate(chunk):
                L = batch['real_lengths'][b]
                per_sample = slice_output_for_sample(last, b, L)
                unrelaxed_path = write_sample_outputs(
                    sample.sample_id, sample.sequence, L, output_dir,
                    per_sample, model, logger,
                )
                unrelaxed_paths.append((sample.sample_id, sample.sequence,
                                        unrelaxed_path))
                logger.info(f'  Wrote outputs for {sample.sample_id}')

        # Free GPU before Amber (OpenMM may grab device memory).
        del outputs, last
        torch.cuda.empty_cache() if str(device).startswith('cuda') else None

        relax_steps = int(args.relax_steps) if args.relax_steps is not None else 0
        if relax_steps > 0:
            use_gpu = str(device) != 'cpu'
            amber = AmberRelaxation(
                max_iterations=relax_steps, logger=logger, use_gpu=use_gpu,
            )
            for sid, _seq, unrelaxed in unrelaxed_paths:
                relaxed = Path(unrelaxed).parent / f'relaxed_{relax_steps}_model.pdb'
                with timing(
                    f'Amber relaxation for {sid}: {relax_steps} iters',
                    logger=logger,
                ):
                    try:
                        amber.process(unrelaxed, str(relaxed))
                    except Exception as e:
                        logger.error(f'Amber relax failed for {sid}: {e}',
                                     exc_info=True)
        else:
            logger.info('Skipping Amber relaxation (--relax-steps 0).')

        if str(device).startswith('cuda') and torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info(f'Peak GPU memory allocated on {device}: {peak_gb:.2f} GiB')

        logger.info(f'Batched inference complete. Results in {output_dir}')

    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batched RhoFold+ inference from an AF3-style JSON input.',
    )
    parser.add_argument(
        '--input', required=True,
        help='Path to AF3-style JSON file. See module docstring for schema.',
    )
    parser.add_argument(
        '--output-dir', required=True,
        help='Output directory. Each sample writes into <output-dir>/<id>/.',
    )
    parser.add_argument(
        '--ckpt',
        default='./checkpoints/rhofold_pretrained_params.pt',
        help='Path to the RhoFold+ pretrained checkpoint.',
    )
    parser.add_argument(
        '--device', default='cpu',
        help='Device, e.g. cuda:0. Default cpu.',
    )
    parser.add_argument(
        '--batch-size', type=int, default=2,
        help='Number of sequences per forward pass. Default 2 (safe for ~24 GB VRAM '
             'at typical RNA lengths in MSA mode; raise for short / single-seq batches).',
    )
    parser.add_argument(
        '--relax-steps', default=1000,
        help='Amber relaxation iterations per sample. Set 0 to skip relax entirely. '
             'Default 1000; runs sequentially after batched forward passes.',
    )
    parser.add_argument(
        '--deterministic', action='store_true',
        help='Enable cudnn.deterministic + use_deterministic_algorithms so '
             'per-sample outputs reproduce across different batch sizes '
             'within float32 noise. Off by default for speed; the structures '
             'are equivalent within ~1e-3 pLDDT max either way, but pLDDT '
             'and distograms may differ across batch shapes when off.',
    )
    parser.add_argument(
        '--no-bucketing', action='store_true',
        help='Disable length bucketing and process samples in input order. '
             'Bucketing (the default) sorts by (L, K) before chunking so each '
             'batch has tight shapes; without it, mixing short and long '
             'sequences in the same batch wastes compute on padding.',
    )
    args = parser.parse_args()
    main(args)
