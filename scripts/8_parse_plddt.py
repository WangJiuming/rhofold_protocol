from pathlib import Path

import numpy as np

import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Parse pLDDT scores from npz file')
    parser.add_argument('--npz', type=str, help='Path to npz file')
    parser.add_argument('--save-plddt', type=str, help='Path to save pLDDT scores, None by default for not saving', default=None)

    args = parser.parse_args()

    npz_path = args.npz  # './results/rhofold/3owz_A/results.npz'
    save_plddt_path = args.save_plddt  # './results/rhofold/3owz_A/plddt.npy'

    distogram_data = np.load(npz_path)
    plddt_scores = distogram_data['plddt']  # shape = (1, L)

    mean_plddt = np.mean(plddt_scores)
    print(f'mean pLDDT = {mean_plddt:.4f}')
    
    # Save the extracted pLDDT scores if the path is specified
    if save_plddt_path is not None:
        Path(save_plddt_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_plddt_path, plddt_scores)
        print(f'pLDDT scores saved to {save_plddt_path}')
