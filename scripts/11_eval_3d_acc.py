import argparse
from pathlib import Path
import shutil
import subprocess


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluate 3D structure using TMscore.')
    parser.add_argument('--pred-pdb', type=str, help='Path to the predicted PDB file.')
    parser.add_argument('--gt-pdb', type=str, help='Path to native PDB file.')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save evaluation results.')
    args = parser.parse_args()

    pred_pdb_path = Path(args.pred_pdb)
    gt_pdb_path = Path(args.gt_pdb)

    cmd = f'TMscore {pred_pdb_path} {gt_pdb_path} -seq -mol RNA -atom " C4\'"'
    
    print(f'Running command: {cmd}')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print('Command output:')
    print(result.stdout)
    print('Command error (if any):')
    print(result.stderr)
    
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the command output to a file in the specified directory
        output_file = output_dir / 'log_3d_acc.txt'
        with open(output_file, 'w') as f:
            f.write('Command output:\n')
            f.write(result.stdout)
            f.write('\nCommand error (if any):\n')
            f.write(result.stderr)
        
        # Optionally, copy the PDB files to the output directory for reference
        shutil.copy(pred_pdb_path, output_dir / pred_pdb_path.name)
        shutil.copy(gt_pdb_path, output_dir / gt_pdb_path.name)
        
        print(f'Evaluation results and PDB files saved in {output_dir}')
    
    