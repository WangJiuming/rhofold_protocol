import argparse

from pymol import cmd

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Visualize pLDDT scores on a protein structure.')
    parser.add_argument('--pdb', type=str, help='Path to PDB file with pLDDT scores as B-factors.')
    parser.add_argument('--save-plot', type=str, help='Path to save the pLDDT visualization plot.')
    args = parser.parse_args()

    pdb_path = args.pdb

    # Load the PDB file.
    cmd.load(pdb_path, 'pdb_with_plddt')

    # Color the model based on the B-factor using the 'blue_white_red' spectrum,
    # with the color range specified from 0 to 1.
    # This can be adjusted based on the expected pLDDT score range.
    cmd.spectrum('b', 'blue_white_red', 'pdb_with_plddt', minimum=0, maximum=1)

    # Automatically orient the structure: centers the object and applies a rotation based on its mass distribution.
    cmd.orient('pdb_with_plddt')

    # Set the background color to white for better contrast.
    cmd.bg_color('white')

    # Render the scene using ray tracing for a high-quality image and save as PNG.
    # You can adjust the width, height, dpi, and ray tracing options as needed.
    cmd.png(args.save_plot, width=1200, height=800, dpi=300, ray=1)
    
    print(f'Visualized structure colored by pLDDT saved to {args.save_plot}')
