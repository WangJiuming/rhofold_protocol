import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize secondary structure from a CT file')
    parser.add_argument('--ct', type=str, help='Path to CT file')
    parser.add_argument('--save-db', type=str, help='Path to save the dot-bracket notation')
    parser.add_argument('--save-npy', type=str, help='Path to save the .npy contact map')
    parser.add_argument('--save-plot', type=str, help='Path to save the contact map plot')
    args = parser.parse_args()

    # input
    ct_path = args.ct
    
    print(f'Loading secondary structure in CT format from {ct_path}')
    
    # outputs
    db_path = args.save_db
    npy_path = args.save_npy
    save_plot_path = args.save_plot

    ct_df = pd.read_csv(ct_path, sep='\t', header=None, skiprows=[0],
                        usecols=[1, 4])  # skip the first row which is header line
    # column 2 is the nucleotide, and column 5 is the one-based index for the paired nucleotide

    seq = ''.join(ct_df[1].values)

    # convert to dot-bracket format
    db_list = ['.' for i in range(len(seq))]  # initialize all as unpaired
    for i in range(len(seq)):
        pair_idx = ct_df.iloc[i, 1] - 1

        if pair_idx != 0 and pair_idx > i:
            db_list[i] = '('
            db_list[pair_idx - 1] = ')'

    db = ''.join(db_list)

    # save the dot-bracket notation to a file
    with open(db_path, 'w') as f:
        f.write(f'{seq}\n{db}')

    # convert to contact map
    contact_map = np.zeros((len(seq), len(seq)))

    for i in range(len(seq)):
        pair_idx = ct_df.iloc[i, 1] - 1

        if pair_idx != 0 and pair_idx > i:
            contact_map[i, pair_idx] = 1
            contact_map[pair_idx, i] = 1

    np.save(npy_path, contact_map)

    plt.figure(figsize=(10, 8))

    # heat map
    sns.heatmap(contact_map, cmap='Purples')

    # boundary box
    plt.gca().add_patch(
        plt.Rectangle((0, 0), contact_map.shape[1], contact_map.shape[0], fill=False, edgecolor='black', lw=3))

    # move the x ticks to the top
    plt.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)

    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
    
    print(f'Dot-bracket notation saved to {db_path}')
    print(f'Contact map .npy file saved to {npy_path}')
    print(f'Secondary structure contact map plot saved to {save_plot_path}')
