import argparse

import torch

from lddt_lib.lddt import LddtAssessor
from lddt_lib.tools.pdb_parser import PdbParser


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='lDDT evaluation')
    parser.add_argument('--fasta', type=str, help='path to the fasta file')
    parser.add_argument('--pred-pdb', type=str, help='path to the predicted pdb file')
    parser.add_argument('--gt-pdb', type=str, help='path to the ground truth pdb file')
    args = parser.parse_args()

    fasta_path = args.fasta
    pred_pdb_path = args.pred_pdb
    gt_pdb_path = args.gt_pdb

    pdb_parser = PdbParser()

    aa_seq_ref, cord_tns_ref, cmsk_mat_ref, structure_ref, error_msg1 = pdb_parser.run(gt_pdb_path, fas_fpath=fasta_path)
    aa_seq_dcy, cord_tns_qry, cmsk_mat_qry, structure_dcy, error_msg2 = pdb_parser.run(pred_pdb_path, fas_fpath=fasta_path)

    cmsk_mat = cmsk_mat_ref * cmsk_mat_qry

    assessor = LddtAssessor()

    plddt_vec, plmsk_vec, clddt_val = assessor.run(torch.FloatTensor(cord_tns_ref),
                                                torch.FloatTensor(cord_tns_qry),
                                                torch.LongTensor(cmsk_mat))

    print(f'Predicted structure: {pred_pdb_path}')
    print(f'Native structure: {gt_pdb_path}')
    
    print(f'lDDT (C1\'-only): {clddt_val.item():.4f}')
