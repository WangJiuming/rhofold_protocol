"""
Lightning-based batched/parallel inference entry for RhoFold.

This script mirrors rhofold/inference.py but runs via Lightning's
Trainer.predict to enable multi-GPU or multi-process parallel prediction
and convenient batching.

Key points:
- CLI largely mirrors rhofold/inference.py; adds directory mode with
  separate --fasta-dir and --msa-dir, and --num-devices for CUDA.
- Single-sample mode: pass --fasta/--msa (or --use-single-seq).
- Directory mode: scan --fasta-dir; for each FASTA, search a matching MSA
  in --msa-dir with the same stem. If missing, log and fall back to
  single-sequence for that sample (handled in the DataModule).
- Devices: choose accelerator via --device (cpu|cuda|mps). When using
  CUDA, set --num-devices to control parallel workers. No explicit index
  parsing like cuda:0.
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import contextlib
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch

# Use Lightning only (no pytorch_lightning fallback)
from lightning.pytorch import Trainer
import lightning.pytorch as pl

from torch.utils.data import Dataset, DataLoader

from rhofold.config import rhofold_config
from rhofold.relax.relax import AmberRelaxation
from rhofold.rhofold import RhoFold
from rhofold.utils import timing, save_ss2ct
from rhofold.utils.alphabet import get_features


# ------------------------
# IO discovery utilities
# ------------------------

FA_EXTS = {".fa", ".fasta", ".fna"}
MSA_EXTS = [".afa", ".a3m", ".aln", ".sto", ".msa", ".fa", ".fasta"]


def _stem(path: str) -> Tuple[str, str]:
    base = os.path.basename(path)
    root, ext = os.path.splitext(base)
    return root, ext.lower()


def _is_fasta(path: str) -> bool:
    return _stem(path)[1] in FA_EXTS


@dataclass
class Sample:
    fasta: str
    msa: Optional[str]
    sample_id: str


# ------------------------
# Dataset
# ------------------------

class RhoFoldInferenceDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        msa_path = s.fasta if s.msa is None else s.msa
        data_dict = get_features(s.fasta, msa_path)
        # Attach meta for saving
        data_dict["sample_id"] = s.sample_id
        data_dict["fasta_path"] = s.fasta
        data_dict["msa_path"] = msa_path
        return data_dict


def _list_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Variable L across samples; keep as a list for predict_step to iterate
    return batch


# ------------------------
# LightningModule
# ------------------------

class LightningRhoFold(pl.LightningModule):
    def __init__(self,
                 ckpt_path: str,
                 recycles: Optional[int] = None,
                 relax_steps: Optional[int] = 1000,
                 profile: bool = False,
                 amp: bool = False,
                 output_root: Optional[str] = None,
                 ):
        super().__init__()
        # Save hyperparams for checkpointing (optional)
        self.save_hyperparameters({
            "ckpt_path": ckpt_path,
            "recycles": recycles,
            "relax_steps": relax_steps,
            "profile": profile,
            "amp": amp,
            "output_root": output_root,
        })
        if recycles is not None:
            rhofold_config.model.recycling_embedder.recycles = int(recycles)

        self.model = RhoFold(rhofold_config)
        state = torch.load(ckpt_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state["model"])
        self.model.eval()

        # Avoid Lightning adding grads on predict
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, tokens, rna_fm_tokens, seq, profile: bool, logger: Optional[logging.Logger]):
        return self.model(tokens=tokens, rna_fm_tokens=rna_fm_tokens, seq=seq, profile=profile, logger=logger)

    def _build_logger(self, out_dir: str) -> logging.Logger:
        os.makedirs(out_dir, exist_ok=True)
        logger = logging.getLogger(f"RhoFold+Predict:{out_dir}")
        logger.setLevel(logging.DEBUG)
        # Clear duplicate handlers when reusing in workers
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        fh = logging.FileHandler(os.path.join(out_dir, 'log.txt'), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        # Batch may be a list of samples due to custom collate
        samples: List[Dict[str, Any]] = batch if isinstance(batch, list) else [batch]

        device = self.device  # Lightning's assigned device for this process

        for sample in samples:
            sample_id: str = sample["sample_id"]
            # Decide output directory
            if self.hparams.output_root:
                out_dir = os.path.join(self.hparams.output_root, sample_id)
            else:
                # Fallback: use working directory per sample id
                out_dir = os.path.abspath(sample_id)

            logger = self._build_logger(out_dir)

            with timing(f"RhoFold+ Inference [{sample_id}]", logger=logger):
                with timing('Select/Move Device', logger=logger):
                    # Tensors are currently on CPU; move to module device
                    tokens = sample['tokens'].to(device)
                    rna_fm_tokens = sample['rna_fm_tokens'].to(device)
                    seq = sample['seq']

                amp_enabled = bool(self.hparams.amp)
                amp_device_type = None
                if amp_enabled:
                    if device.type == 'cuda':
                        amp_device_type = 'cuda'
                        try:
                            torch.backends.cuda.matmul.allow_tf32 = True
                            torch.backends.cudnn.allow_tf32 = True
                            if hasattr(torch, 'set_float32_matmul_precision'):
                                torch.set_float32_matmul_precision('high')
                        except Exception:
                            pass
                    elif device.type == 'mps':
                        logger.warning('AMP disabled on MPS; running in FP32.')
                        amp_enabled = False
                    else:
                        logger.warning('AMP requested but not supported on device %s; running in FP32.', device)
                        amp_enabled = False

                # Forward
                with timing('Forward Pass', logger=logger):
                    autocast_ctx = (
                        torch.autocast(device_type=amp_device_type, dtype=(torch.bfloat16 if amp_device_type == 'cuda' else torch.float16))
                        if amp_enabled and amp_device_type is not None else contextlib.nullcontext()
                    )
                    with torch.no_grad(), autocast_ctx:
                        outputs = self(tokens=tokens, rna_fm_tokens=rna_fm_tokens, seq=seq, profile=bool(self.hparams.profile), logger=logger)

                output = outputs[-1]

                with timing('Save Outputs (.ct + .npz + .pdb)', logger=logger):
                    # Secondary structure map and CT
                    ss_prob_map = torch.sigmoid(output['ss'][0, 0].float()).detach().cpu().numpy()
                    save_ss2ct(ss_prob_map, seq, os.path.join(out_dir, 'ss.ct'), threshold=0.5)

                    # NPZ package
                    np.savez_compressed(
                        os.path.join(out_dir, 'results.npz'),
                        dist_n=torch.softmax(output['n'].squeeze(0).float(), dim=0).detach().cpu().numpy(),
                        dist_p=torch.softmax(output['p'].squeeze(0).float(), dim=0).detach().cpu().numpy(),
                        dist_c=torch.softmax(output['c4_'].squeeze(0).float(), dim=0).detach().cpu().numpy(),
                        ss_prob_map=ss_prob_map,
                        plddt=output['plddt'][0].float().detach().cpu().numpy(),
                    )

                    # Unrelaxed PDB
                    unrelaxed_model = os.path.join(out_dir, 'unrelaxed_model.pdb')
                    node_cords_pred = output['cord_tns_pred'][-1].squeeze(0).float()
                    # Use the model's built-in converter to save PDB
                    self.model.structure_module.converter.export_pdb_file(
                        seq,
                        node_cords_pred.data.cpu().numpy(),
                        path=unrelaxed_model,
                        chain_id=None,
                        confidence=output['plddt'][0].float().detach().cpu().numpy(),
                        logger=logger,
                    )

                # Amber relaxation (optional)
                relax_steps = self.hparams.relax_steps
                if relax_steps is not None and int(relax_steps) > 0:
                    use_gpu = (device.type == 'cuda')
                    with timing(f'Amber Relaxation : {int(relax_steps)} iterations', logger=logger):
                        amber_relax = AmberRelaxation(max_iterations=int(relax_steps), logger=logger, use_gpu=use_gpu)
                        relaxed_model = os.path.join(out_dir, f'relaxed_{int(relax_steps)}_model.pdb')
                        amber_relax.process(unrelaxed_model, relaxed_model)

        return None


# ------------------------
# LightningDataModule
# ------------------------

class RhoFoldDataModule(pl.LightningDataModule):
    def __init__(self,
                 fasta: Optional[str] = None,
                 msa: Optional[str] = None,
                 fasta_dir: Optional[str] = None,
                 msa_dir: Optional[str] = None,
                 use_single_seq: bool = False,
                 num_workers: int = 2,
                 ):
        super().__init__()
        self.single_fasta = fasta
        self.single_msa = msa
        self.fasta_dir = fasta_dir
        self.msa_dir = msa_dir
        self.use_single_seq = bool(use_single_seq)
        self.num_workers = int(num_workers)
        self.samples: List[Sample] = []
        self.is_single_sample = False

    def _find_msa_for_stem(self, stem: str) -> Optional[str]:
        if not self.msa_dir:
            return None
        for ext in MSA_EXTS:
            cand = os.path.join(self.msa_dir, stem + ext)
            if os.path.isfile(cand):
                return cand
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        self.samples = []
        # Single-sample mode
        if self.single_fasta is not None:
            sample_id = os.path.splitext(os.path.basename(self.single_fasta))[0]
            msa_path = None
            if not self.use_single_seq and self.single_msa:
                msa_path = self.single_msa
            elif not self.use_single_seq and not self.single_msa:
                print(f"[datamodule] No --msa provided for {sample_id}; falling back to single-sequence.")
            self.samples.append(Sample(fasta=self.single_fasta, msa=msa_path, sample_id=sample_id))
            self.is_single_sample = True
            # When running in explicit single-sample mode, write directly to output_dir
            self.samples[0].sample_id = '.'
            return

        # Directory mode
        if not self.fasta_dir:
            raise ValueError("Provide either --fasta (single) or --fasta-dir (batch) input.")
        if not os.path.isdir(self.fasta_dir):
            raise ValueError(f"FASTA dir not found: {self.fasta_dir}")
        if self.msa_dir and not os.path.isdir(self.msa_dir):
            raise ValueError(f"MSA dir not found: {self.msa_dir}")

        fasta_names = sorted(n for n in os.listdir(self.fasta_dir)
                             if os.path.isfile(os.path.join(self.fasta_dir, n)) and _is_fasta(os.path.join(self.fasta_dir, n)))
        if not fasta_names:
            raise ValueError(f"No FASTA files discovered in {self.fasta_dir}")

        for name in fasta_names:
            fas = os.path.join(self.fasta_dir, name)
            stem, _ = os.path.splitext(name)
            msa_path = None
            if not self.use_single_seq:
                msa_path = self._find_msa_for_stem(stem)
                if msa_path is None:
                    print(f"[datamodule] MSA not found for {stem} in {self.msa_dir or '<none>'}; falling back to single-sequence.")
            self.samples.append(Sample(fasta=fas, msa=msa_path, sample_id=stem))

        self.is_single_sample = (len(self.samples) == 1)

    def predict_dataloader(self):
        dataset = RhoFoldInferenceDataset(self.samples)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=_list_collate)


# ------------------------
# CLI
# ------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RhoFold+ Lightning Inference")

    # Mirror original flags
    parser.add_argument("--ckpt", default='./checkpoints/rhofold_pretrained_params.pt', help="Path to pretrained model checkpoint.")
    parser.add_argument("--device", default='cpu', help="Device: cpu | cuda | mps.")
    parser.add_argument("--fasta", default=None, help="Path to input FASTA (single-sample mode).")
    parser.add_argument("--msa", default=None, help="Path to input MSA (single-sample mode).")
    parser.add_argument("--fasta-dir", default=None, help="Directory containing FASTA files (batch mode).")
    parser.add_argument("--msa-dir", default=None, help="Directory containing MSA files (batch mode, matched by stem).")
    parser.add_argument("--output-dir", required=True, help="Output root directory. In directory mode, per-sample subfolders are created.")
    parser.add_argument("--relax-steps", default=1000, help="Amber relaxation steps (int). Set 0 to disable.")
    parser.add_argument("--recycles", type=int, default=None, help="Override recycling iterations (default from config).")
    parser.add_argument("--use-single-seq", action='store_true', help="Use single sequence only (no MSA).")
    parser.add_argument("--profile", action='store_true', help="Enable detailed per-step timing logs.")
    parser.add_argument("--amp", "--mixed-precision", dest='amp', action='store_true', help="Enable automated mixed precision.")

    # Parallelism
    parser.add_argument("--num-devices", type=int, default=1, help="Number of CUDA devices to use when --device=cuda.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers for feature building.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Optional runtime overrides
    if getattr(args, 'recycles', None) is not None:
        rhofold_config.model.recycling_embedder.recycles = int(args.recycles)

    # Data module builds and manages per-sample MSA fallback
    dm = RhoFoldDataModule(
        fasta=args.fasta,
        msa=(None if args.use_single_seq else args.msa),
        fasta_dir=args.fasta_dir,
        msa_dir=args.msa_dir,
        use_single_seq=bool(args.use_single_seq),
        num_workers=int(args.num_workers),
    )
    # Explicitly set up to know single vs batch and possibly adjust output
    dm.setup(stage='predict')

    # Lightning model wrapper
    lit_model = LightningRhoFold(
        ckpt_path=args.ckpt,
        recycles=args.recycles,
        relax_steps=int(args.relax_steps) if args.relax_steps is not None else None,
        profile=bool(args.profile),
        amp=bool(args.amp),
        output_root=args.output_dir,
    )

    # Configure accelerator/devices
    dev = str(args.device).strip().lower()
    if dev == 'cuda':
        accelerator = 'gpu'
        devices = int(args.num_devices)
        precision = 'bf16-mixed' if args.amp else 32
    elif dev == 'mps':
        accelerator = 'mps'
        devices = 1
        precision = 32
        if args.amp:
            print("[warn] AMP is not supported on MPS; running in FP32.", file=sys.stderr)
    else:
        accelerator = 'cpu'
        devices = 1
        precision = 32

    os.makedirs(args.output_dir, exist_ok=True)

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy='auto',
        precision=precision,
        logger=False,
        enable_checkpointing=False,
    )

    # In single-sample mode, write directly to output-dir (avoid nested folder)
    if dm.is_single_sample and dm.samples:
        dm.samples[0].sample_id = '.'

    trainer.predict(lit_model, datamodule=dm)


if __name__ == "__main__":
    main()
