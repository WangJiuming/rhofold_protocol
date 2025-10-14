from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, PDBIO

import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Add pLDDT scores as B-factors to a PDB file.')
    parser.add_argument('--plddt', type=str, help='Path to pLDDT scores saved as an NPY file.')
    parser.add_argument('--pdb', type=str, help='Path to PDB file.')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save the updated PDB file. If not provided, the updated PDB will be saved in the same directory as the input PDB file.')
    args = parser.parse_args()

    plddt_path = Path(args.plddt)
    pdb_path = Path(args.pdb)

    # Load and flatten the pLDDT scores
    plddt_scores = np.load(plddt_path).flatten()

    # Parse the structure from the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('RNA', str(pdb_path))

    # Update B-factors for each residue using corresponding pLDDT scores
    for i, residue in enumerate(structure.get_residues()):
        if i < len(plddt_scores):
            score = plddt_scores[i]
        else:
            print(f'Warning: pLDDT score not available for residue {residue.get_id()}, setting B-factor to 0.0')
            score = 0.0
        for atom in residue:
            atom.set_bfactor(score)

    # Save the updated structure to a new PDB file
    if args.output_dir:
        output_path = Path(args.output_dir) / f'{pdb_path.stem}_plddt.pdb'
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = pdb_path.parent / f'{pdb_path.stem}_plddt.pdb'
    
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))

    print(f'Updated PDB file saved to {output_path}')
