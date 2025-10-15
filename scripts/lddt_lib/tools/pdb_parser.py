"""Parse the FASTA sequence & atom coordinates from the PDB file."""

import os
import re
import logging
import warnings
from datetime import datetime
import random
import hashlib

import numpy as np
import torch
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
from .rna_utils import parse_fas_file
from .rna_constants import get_frame_definition


def get_rand_str():
    """Get a randomized string.

    Args: n/a

    Returns:
    * rand_str: randomized string
    """

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    rand_val = random.random()
    rand_str_raw = '%s_%f' % (timestamp, rand_val)
    rand_str = hashlib.md5(rand_str_raw.encode('utf-8')).hexdigest()

    return rand_str


class PdbParserError(Exception):
    """Exceptions raised by the <PdbParser> class."""


class PdbParser():
    """Parser for PDB files.

    Note: GZ-compressed PDB file can also be provided (*.gz).
    """

    def __init__(self, pdb_version='v0.0', frame_version = 'v1', check_mode='strict'):
        """Constructor function."""

        self.check_mode = check_mode  # choices: 'strict' OR 'lenient'

        self.frame_info = get_frame_definition(frame_version=frame_version)

        self.pdb_version = pdb_version
        self.frame_version = frame_version

        logging.info(f"pdb_version {pdb_version}, frame_version {frame_version}")

        if pdb_version == 'v0.0':
            self.atom_names = ['C4', "C1'", 'N9', 'N1', 'C2', "C4'", 'P',
                               'N2', 'O6', "C2'", 'N4', "C3'",  "O4'", 'N3', 'OP2',  'OP1',
                               "O2'", 'N6', "O5'", 'N7', "C5'", 'C8', 'O2', 'C5', 'C6', 'O4', "O3'"]
        else:
            raise NotImplementedError
        
        self.atom_index_dict = {self.atom_names[i]: i for i in range(len(self.atom_names))}
        self.n_atoms_per_resd = len(self.atom_names)
        self.atom_mask_thres = 0.5  # minimal ratio of atoms w/ coordinates
        self.parser = PDBParser()
        # suppress all the warnings raised by <BioPython>
        warnings.simplefilter('ignore', BiopythonWarning)


    def run(self, pdb_fpath, fas_fpath=None, structure=None, chain_id=None, aa_seq=None, sel=True):
        """Parse the PDB file.

        Args:
        * pdb_fpath: path to the PDB file
        * fas_fpath: (optional) path to the FASTA file
        * structure: (optional) BioPython's parsing results (to parse multiple chains from one PDB)
        * chain_id: (optional) chain ID

        Returns:
        * aa_seq: amino-acid sequence
        * atom_cords: atom coordinates of size L x N x 3
        * atom_masks: atom coordinates' validness masks of size L x 3
        * structure: BioPython's parsing results
        * error_msg: error message raised when parsing the PDB file

        Note:
        1. The amino-acid sequence is determined in the following order:
           a) parsed from the FASTA file, if exists
           b) parsed from SEQRES records in the PDB file
           c) parsed from ATOM records in the PDB file
        2. If <chain_id> is not provided, then the first chain will be returned. The specific order
           is defined by the <BioPython> package. If <chain_id> is provided, then the first model
           with the specified chain ID will be returned.
        """

        # show the greeting message
        logging.debug('parsing the PDB file: %s (chain ID: <%s>)', pdb_fpath, chain_id)
        # attempt to parse the PDB file
        try:
            # check inputs
            if not os.path.exists(pdb_fpath):
                raise PdbParserError('PDB_FILE_NOT_FOUND')
            if (aa_seq is None and not os.path.exists(fas_fpath)):
                raise PdbParserError('FASTA_FILE_NOT_FOUND')

            if aa_seq is None:
                _, aa_seq = parse_fas_file(fas_fpath)

            # parse the PDB file w/ biopython
            if structure is None:
                structure = self.__get_structure(pdb_fpath)

            # find the first chain matching the chain ID
            chain = self.__get_chain(structure, chain_id)

            # obtain atom coordinates & validness masks
            atom_cords, atom_masks = self.__get_atoms(chain, aa_seq)

            if sel:
                atom_cords, atom_masks = self.get_atoms_sel(aa_seq,
                                                            torch.FloatTensor(atom_cords),
                                                            torch.FloatTensor(atom_masks),
                                                            )
                atom_cords, atom_masks = atom_cords.data.cpu().numpy(), atom_masks.data.cpu().numpy()

            # set the error message to None
            error_msg = None

        except PdbParserError as error:
            logging.warning('PDB file path: %s / Error: %s', pdb_fpath, error)
            aa_seq, atom_cords, atom_masks, structure, error_msg = None, None, None, None, error

        return aa_seq, atom_cords, atom_masks, structure, error_msg


    def __get_structure(self, path):
        """Get the structure from the PDB file."""

        try:
            with open(path, 'r') as i_file:
                structure = self.parser.get_structure(get_rand_str(), i_file)
        except PDBConstructionException as error:
            raise PdbParserError('BIOPYTHON_FAILED_TO_PARSE') from error

        return structure


    @classmethod
    def __get_chain(cls, structure, chain_id):
        """Get the first chain matching the specified chain ID (could be None)."""

        chain = None
        for model in structure:
            for chain_curr in model:
                if (chain_id is None) or (chain_curr.get_id() == chain_id):
                    chain = chain_curr
                    break
            if chain is not None:
                break

        # check whether the specified chain has been found
        if chain is None:
            raise PdbParserError('CHAIN_NOT_FOUND')

        return chain


    def __get_atoms(self, chain, aa_seq):
        """Get atom coordinates & masks from the specified chain."""

        # obtain all the segments' information
        seg_infos = self.__get_seg_infos(chain)

        # find the valid offset for residue indices, if possible
        offset = self.__get_offset(seg_infos, aa_seq)

        # obtain atom coordinates & masks
        seq_len = len(aa_seq)
        atom_cords = np.zeros((seq_len, self.n_atoms_per_resd, 3), dtype=np.float32)
        atom_masks = np.zeros((seq_len, self.n_atoms_per_resd), dtype=np.int8)
        
        for residue in chain:
            # skip hetero-residues, and obtain the residue's index
            het_flag, idx_resd, _ = residue.get_id()
            if het_flag.strip() != '':
                continue  # skip hetero-residues

            # update atom coordinates & masks
            for idx_atom, atom_name in enumerate(self.atom_names):
                if residue.has_id(atom_name):
                    atom_cords[idx_resd + offset, idx_atom] = residue[atom_name].get_coord()
                    atom_masks[idx_resd + offset, idx_atom] = 1

        # check whether sufficient ratio of atoms have coordinates
        if (self.check_mode == 'strict') and (np.mean(atom_masks) < self.atom_mask_thres):
            raise PdbParserError('INSUFFICIENT_ATOMS_W_COORDINATES')

        return atom_cords, atom_masks

    def get_atoms_sel(self, aa_seq, atom_tns_all, mask_tns_all, topn_cords = -1):
        """Get per-atom 3D coordinates or validness masks for selected atom(s).

        Args:
        * aa_seq: amino-acid sequence
        * atom_tns_all: full per-atom 3D coordinates (L x M x 3) or validness masks (L x M)
        * atom_names_sel: list of selected atom names of length M'

        Returns:
        * atom_tns_sel: selected per-atom 3D coordinates (L x M' x 3) or validness masks (L x M')

        Note:
        * If only one atom name if provided, then its corresponding dimension is squeezed.
        """

        # initialization
        device = atom_tns_all.device
        n_atoms = self.frame_info.ATOM_NUM_MAX if topn_cords < 0 else topn_cords

        # build the indexing tensor for selected atom(s)
        idxs_vec_dict = {}  # atom indices
        msks_vec_dict = {}  # atom indices' validness masks

        for resd_name in ['A', 'U', 'G', 'C']:
            atom_names_all = self.frame_info.ATOM_NAMES_PER_RESD[resd_name]
            idxs_vec_np = np.zeros((n_atoms), dtype=np.int64)
            msks_vec_np = np.zeros((n_atoms), dtype=np.int8)
            for idx_atom_sel, atom_name_sel in enumerate(atom_names_all):

                if topn_cords > 0 and idx_atom_sel >= topn_cords:
                    continue

                if atom_name_sel in self.atom_index_dict.keys():
                    idxs_vec_np[idx_atom_sel] = self.atom_index_dict[atom_name_sel]
                    msks_vec_np[idx_atom_sel] = 1

            idxs_vec_dict[resd_name] = idxs_vec_np
            msks_vec_dict[resd_name] = msks_vec_np

        # # determine the overall indexing tensor based on the amino-acid sequence
        idxs_mat_full_np = np.stack([idxs_vec_dict[x] for x in aa_seq], axis=0)
        msks_mat_full_np = np.stack([msks_vec_dict[x] for x in aa_seq], axis=0)
        idxs_mat_full = torch.tensor(idxs_mat_full_np, dtype=torch.int64, device=device)  # L x M'
        msks_mat_full = torch.tensor(msks_mat_full_np, dtype=torch.int64, device=device)  # L x M'

        # get per-atom 3D coordinates or validness masks for specified residue(s) & atom(s)
        if atom_tns_all.ndim == 2:
            atom_tns_sel = msks_mat_full * torch.gather(atom_tns_all, 1, idxs_mat_full)
            mask_tns_sel = msks_mat_full * torch.gather(mask_tns_all, 1, idxs_mat_full)
        else:
            n_dims_addi = atom_tns_all.shape[-1]
            atom_tns_sel = msks_mat_full.unsqueeze(dim=2) * torch.gather(
                atom_tns_all, 1, idxs_mat_full.unsqueeze(dim=2).repeat(1, 1, n_dims_addi))

            mask_tns_sel = msks_mat_full * torch.gather(mask_tns_all, 1, idxs_mat_full)

        # squeeze the dimension if only one atom is selected
        if n_atoms == 1:
            atom_tns_sel.squeeze_(dim=1)

        return atom_tns_sel, mask_tns_sel


    @classmethod
    def __get_seg_infos(cls, chain):
        """Get discontinous segments' information for the specified chain."""

        seg_infos = []
        for residue in chain:
            # obtain the current residue's basic information
            resd_name = residue.get_resname()
            resd_name_1c = resd_name
            het_flag, idx_resd, ins_code = residue.get_id()
            if het_flag.strip() != '':
                continue  # skip hetero-residues
            if ins_code.strip() != '':
                raise PdbParserError('HAS_INSERTED_RESIDUES')

            # update the last segment, or add a new segment
            if len(seg_infos) >= 1 and seg_infos[-1]['ie'] == idx_resd:
                seg_infos[-1]['ie'] += 1
                seg_infos[-1]['seq'] += resd_name_1c
            else:
                seg_infos.append({
                    'ib': idx_resd,  # inclusive
                    'ie': idx_resd + 1,  # exclusive
                    'seq': resd_name_1c,  # 20 amino-acids + '.' for wild-card matches
                })

        # sort discontinous segments in the descending order of segment length
        seg_infos.sort(key=lambda x: x['ie'] - x['ib'], reverse=True)

        return seg_infos


    @classmethod
    def __get_offset(cls, seg_infos, aa_seq):
        """Get a valid offset of residue indices in ATOM records & amino-acid sequence."""

        offset_list = None
        for seg_info in seg_infos:
            regex = re.compile(seg_info['seq'])
            if offset_list is None:
                offset_list = [m.start() - seg_info['ib'] for m in re.finditer(regex, aa_seq)]
            else:
                offset_list_new = []
                for offset in offset_list:
                    if re.search(regex, aa_seq[seg_info['ib'] + offset:seg_info['ie'] + offset]):
                        offset_list_new.append(offset)
                offset_list = offset_list_new

            if len(offset_list) == 0:
                raise PdbParserError('NO_VALID_OFFSET')

        return offset_list[0]  # use the first valid offset

    def export_pdb_file_(self, seq, atom_cords, path, atom_masks=None, atom_names=None):
        """Export the 3D structure to a PDB file.

        Args:
        * seq: amino-acid sequence
        * atom_cords: atom coordinates (['']: L x 3 / ["C1'", "C4'", "O5'", "N91"]: L x 4 x 3 / 4L x 3)
        * path: path to the PDB file
        * (optional) atom_masks: atom masks (['']: L / ["C1'", "C4'", "O5'", "N91"]: L x 4 / 4L)

        Returns: n/a
        """

        # configurations
        i_code = ' '
        occupancy = 1.0
        temp_factor = 0.0
        charge = ' '
        cord_min = -999.0
        cord_max = 999.0
        seq_len = len(seq)

        atom_names = self.atom_names if atom_names is None else atom_names
        n_key_atoms = len(atom_names)

        # take all the atom coordinates as valid, if not specified
        if atom_masks is None:
            atom_masks = np.ones(atom_cords.shape[:-1], dtype=np.int8)

        # determine the set of atom names (per residue)
        if atom_cords.ndim == 2:
            if atom_cords.shape[0] == seq_len * n_key_atoms:
                atom_cords_ext = np.reshape(atom_cords, [seq_len, n_key_atoms, 3])
                atom_masks_ext = np.reshape(atom_masks, [seq_len, n_key_atoms])
            else:
                raise ValueError('atom coordinates\' shape does not match the sequence length')

        elif atom_cords.ndim == 3:
            assert atom_cords.shape[0] == seq_len
            atom_cords_ext = atom_cords
            atom_masks_ext = atom_masks
        else:
            raise ValueError('atom coordinates must be a 2D or 3D np.ndarray')

        # reset invalid values in atom coordinates
        atom_cords_ext = np.clip(atom_cords_ext, cord_min, cord_max)
        atom_cords_ext[np.isnan(atom_cords_ext)] = 0.0
        atom_cords_ext[np.isinf(atom_cords_ext)] = 0.0

        # export the 3D structure to a PDB file
        os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
        with open(path, 'w') as o_file:
            n_atoms = 0
            for idx_resd, resd_name in enumerate(seq):
                for idx_atom, atom_name in enumerate(atom_names):
                    if atom_masks_ext[idx_resd, idx_atom] == 0:
                        continue
                    n_atoms += 1
                    charge = atom_name[0]
                    line_str = ''.join([
                        'ATOM  ',
                        '%5d' % n_atoms,
                        '  ' + atom_name + ' ' * (3 - len(atom_name)),
                        '   %s' % resd_name,
                        ' ' * (2),
                        ' ' * (4 - len(str(idx_resd + 1))),
                        '%s' % str(idx_resd + 1),
                        '%s   ' % i_code,
                        '%8.3f' % atom_cords_ext[idx_resd, idx_atom, 0],
                        '%8.3f' % atom_cords_ext[idx_resd, idx_atom, 1],
                        '%8.3f' % atom_cords_ext[idx_resd, idx_atom, 2],
                        '%6.2f' % occupancy,
                        '%6.2f' % temp_factor,
                        ' ' * 10,
                        '%2s' % charge,
                        '%2s' % ' ',
                    ])
                    assert len(line_str) == 80, 'line length must be exactly 80 characters: ' + line_str
                    o_file.write(line_str + '\n')

    def export_pdb_file_list(self, seq, atom_cords, path, atom_masks=None, n_key_atoms=None, atom_names=None):
        """Export the 3D structure to a PDB file.

        Args:
        * seq: amino-acid sequence
        * atom_cords: atom coordinates (['']: L x 3 / ["C1'", "C4'", "O5'", "N91"]: L x 4 x 3 / 4L x 3)
        * path: path to the PDB file
        * (optional) atom_masks: atom masks (['']: L / ["C1'", "C4'", "O5'", "N91"]: L x 4 / 4L)

        Returns: n/a
        """

        # configurations
        i_code = ' '
        chain_id = 0
        occupancy = 1.0
        temp_factor = 0.0
        charge = ' '
        cord_min = -999.0
        cord_max = 999.0
        if isinstance(seq, list):
            seq_len = 0
            for s in seq:
                seq_len += len(s)
        else:
            seq_len = len(seq)

        seq_lens = [len(s) for s in seq]

        n_key_atoms = self.frame_info.ATOM_NUM_MAX if n_key_atoms is None else n_key_atoms

        logging.info(f'self.frame_info.ATOM_NUM_MAX {self.frame_info.ATOM_NUM_MAX}')
        # print(self.frame_info.ATOM_NAMES_PER_RESD)

        # take all the atom coordinates as valid, if not specified
        if atom_masks is None:
            atom_masks = np.ones(atom_cords.shape[:-1], dtype=np.int8)

        # determine the set of atom names (per residue)
        if atom_cords.ndim == 2:
            if atom_cords.shape[0] == seq_len * n_key_atoms:
                atom_cords_ext = np.reshape(atom_cords, [seq_len, n_key_atoms, 3])
                atom_masks_ext = np.reshape(atom_masks, [seq_len, n_key_atoms])
            else:
                raise ValueError('atom coordinates\' shape does not match the sequence length')

        elif atom_cords.ndim == 3:
            assert atom_cords.shape[0] == seq_len
            atom_cords_ext = atom_cords
            atom_masks_ext = atom_masks
        else:
            raise ValueError('atom coordinates must be a 2D or 3D np.ndarray')

        # reset invalid values in atom coordinates
        atom_cords_ext = np.clip(atom_cords_ext, cord_min, cord_max)
        atom_cords_ext[np.isnan(atom_cords_ext)] = 0.0
        atom_cords_ext[np.isinf(atom_cords_ext)] = 0.0

        # export the 3D structure to a PDB file
        os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
        with open(path, 'w') as o_file:
            n_atoms = 0
            for chain_id, s in enumerate(seq):
                for idx_resd, resd_name in enumerate(s):
                    for idx_atom, atom_name in enumerate(self.frame_info.ATOM_NAMES_PER_RESD[resd_name]):
                        if atom_masks_ext[idx_resd, idx_atom] == 0:
                            continue
                        n_atoms += 1
                        charge = atom_name[0]
                        line_str = ''.join([
                            'ATOM  ',
                            '%5d' % n_atoms,
                            '  ' + atom_name + ' ' * (3 - len(atom_name)),
                            '   %s' % resd_name,
                            ' %d' % chain_id,
                            ' ' * (4 - len(str(idx_resd + 1))),
                            '%s' % str(idx_resd + 1),
                            '%s   ' % i_code,
                            '%8.3f' % atom_cords_ext[idx_resd + sum(seq_lens[:chain_id]), idx_atom, 0],
                            '%8.3f' % atom_cords_ext[idx_resd + sum(seq_lens[:chain_id]), idx_atom, 1],
                            '%8.3f' % atom_cords_ext[idx_resd + sum(seq_lens[:chain_id]), idx_atom, 2],
                            '%6.2f' % occupancy,
                            '%6.2f' % temp_factor,
                            ' ' * 10,
                            '%2s' % charge,
                            '%2s' % ' ',
                        ])
                        assert len(line_str) == 80, 'line length must be exactly 80 characters: ' + line_str
                        o_file.write(line_str + '\n')

        logging.info(f'save pdb to {path}')

    def export_pdb_file(self, seq, atom_cords, path, atom_masks=None, n_key_atoms = None, atom_names=None,
                        chain_id = None):
        """Export the 3D structure to a PDB file.

        Args:
        * seq: amino-acid sequence
        * atom_cords: atom coordinates (['']: L x 3 / ["C1'", "C4'", "O5'", "N91"]: L x 4 x 3 / 4L x 3)
        * path: path to the PDB file
        * (optional) atom_masks: atom masks (['']: L / ["C1'", "C4'", "O5'", "N91"]: L x 4 / 4L)

        Returns: n/a
        """

        # configurations
        i_code = ' '
        chain_id = '0' if chain_id is None else chain_id
        occupancy = 1.0
        temp_factor = 0.0
        charge = ' '
        cord_min = -999.0
        cord_max = 999.0
        seq_len = len(seq)

        n_key_atoms = self.frame_info.ATOM_NUM_MAX if n_key_atoms is None else n_key_atoms

        logging.info(f'self.frame_info.ATOM_NUM_MAX {self.frame_info.ATOM_NUM_MAX}')
        # print(self.frame_info.ATOM_NAMES_PER_RESD)

        # take all the atom coordinates as valid, if not specified
        if atom_masks is None:
            atom_masks = np.ones(atom_cords.shape[:-1], dtype=np.int8)

        # determine the set of atom names (per residue)
        if atom_cords.ndim == 2:
            if atom_cords.shape[0] == seq_len * n_key_atoms:
                atom_cords_ext = np.reshape(atom_cords, [seq_len, n_key_atoms, 3])
                atom_masks_ext = np.reshape(atom_masks, [seq_len, n_key_atoms])
            else:
                raise ValueError('atom coordinates\' shape does not match the sequence length')
            
        elif atom_cords.ndim == 3:
            assert atom_cords.shape[0] == seq_len
            atom_cords_ext = atom_cords
            atom_masks_ext = atom_masks
        else:
            raise ValueError('atom coordinates must be a 2D or 3D np.ndarray')

        # reset invalid values in atom coordinates
        atom_cords_ext = np.clip(atom_cords_ext, cord_min, cord_max)
        atom_cords_ext[np.isnan(atom_cords_ext)] = 0.0
        atom_cords_ext[np.isinf(atom_cords_ext)] = 0.0
        
        # export the 3D structure to a PDB file
        os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
        with open(path, 'w') as o_file:
            n_atoms = 0
            for idx_resd, resd_name in enumerate(seq):
                for idx_atom, atom_name in enumerate(self.frame_info.ATOM_NAMES_PER_RESD[resd_name]):
                    if atom_masks_ext[idx_resd, idx_atom] == 0:
                        continue
                    n_atoms += 1
                    charge = atom_name[0]
                    line_str = ''.join([
                        'ATOM  ',
                        '%5d' % n_atoms,
                        '  ' + atom_name + ' ' * (3 - len(atom_name)),
                        '   %s' % resd_name,
                        ' %s' % chain_id,
                        ' ' * (4 - len(str(idx_resd + 1))),
                        '%s' % str(idx_resd + 1),
                        '%s   ' % i_code,
                        '%8.3f' % atom_cords_ext[idx_resd, idx_atom, 0],
                        '%8.3f' % atom_cords_ext[idx_resd, idx_atom, 1],
                        '%8.3f' % atom_cords_ext[idx_resd, idx_atom, 2],
                        '%6.2f' % occupancy,
                        '%6.2f' % temp_factor,
                        ' ' * 10,
                        '%2s' % charge,
                        '%2s' % ' ',
                    ])
                    assert len(line_str) == 80, 'line length must be exactly 80 characters: ' + line_str
                    o_file.write(line_str + '\n')

        logging.info(f'save pdb to {path}')

if __name__ == '__main__':
    from . import zfold_init

    version = 'v0.0'
    frame_version = 'v2'
    n_key_atoms = 5
    n_key_atoms_eval = 5

    # configurations
    eps = 1e-6
    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    root = '/public/home/taoshen/data/rna/RNA3D/data_px'

    pdb_root = f'{root}/full_pdb/'
    seq_root = f'{root}/key_seq_v2/'

    tgts = os.listdir(seq_root)
    tgts = [t.replace('.seq','') for t in tgts if t.endswith('.seq')]
    tgts = tgts[-100:]

    rmsds = []
    for tgt in tgts:
        pdb_fpath = f'{pdb_root}/{tgt}.full_pdb'
        fas_fpath = f'{seq_root}/{tgt}.seq'
        print(pdb_fpath)
        # initialization
        zfold_init()
        pdb_parser = PdbParser(pdb_version=version, frame_version=frame_version)
        aa_seq, cord_tns_base, cmsk_mat_base, _, _ = pdb_parser.run(pdb_fpath, fas_fpath=fas_fpath)
        cords, masks = pdb_parser.get_atoms_sel(aa_seq, torch.FloatTensor(cord_tns_base), torch.LongTensor(cmsk_mat_base))
        print(aa_seq)
        print(cord_tns_base.shape)
        print(cords.shape, masks.shape)
        exit()