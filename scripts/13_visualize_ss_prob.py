import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize secondary structure probabilities.')
    parser.add_argument('--npz', type=str, help='Path to NPZ file with secondary structure probabilities.')
    parser.add_argument('--save-plot', type=str, help='Path to save the secondary structure plot.')
    args = parser.parse_args()

    npz_path = args.npz
    save_ss_plot_path = args.save_plot

    print(f'Retrieving secondary structure probability map from {npz_path}')

    distogram_data = np.load(npz_path)

    ss_prob_map = distogram_data['ss_prob_map']

    # visualize the probability map
    plt.figure(figsize=(10, 8))

    # draw the heatmap
    sns.heatmap(ss_prob_map, cmap='Purples')

    # boundary box
    plt.gca().add_patch(plt.Rectangle((0, 0), ss_prob_map.shape[1], ss_prob_map.shape[0], fill=False, edgecolor='black', lw=3))

    # move the x ticks to the top
    plt.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)

    plt.savefig(save_ss_plot_path, dpi=300, bbox_inches='tight')
    
    print(f'Secondary structure probability map saved to {save_ss_plot_path}')
