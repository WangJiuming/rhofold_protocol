import argparse
from pathlib import Path

import neffy
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_neff(msa_path):
    msa_length, msa_depth, neff = neffy.compute_neff(str(msa_path), alphabet=neffy.Alphabet.RNA)
    
    return neff, msa_depth

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Compute Neff for a given MSA file')
    parser.add_argument('--msa', type=str, help='Path to the MSA file.')
    
    args = parser.parse_args()

    msa_path = Path(args.msa)

    if not msa_path.exists():
        raise FileNotFoundError(f'MSA file not found: {msa_path}')
    
    neff, msa_depth = compute_neff(msa_path)
    
    print(f'Computed Neff for MSA file: {msa_path}')
    print(f'Neff: {neff:.2f}')
