/**
 * Generate PDB-format string from coordinates.
 * Ported from rhofold/utils/converter.py export_pdb_file
 */

import { ATOM_NUM_MAX, ATOM_NAMES, type ResdName } from './constants';

/**
 * Generate a PDB string from all-atom coordinates.
 *
 * @param seq - RNA sequence string
 * @param coords - Flat Float32Array [L*23*3]
 * @param mask - Uint8Array [L*23], 1 if atom present
 * @param plddt - Float32Array [L], per-residue confidence (0–1 scale)
 * @param chainId - Chain identifier (default 'A')
 */
export function writePDB(
  seq: string,
  coords: Float32Array,
  mask: Uint8Array,
  plddt?: Float32Array,
  chainId: string = 'A',
): string {
  const lines: string[] = [];
  let atomSerial = 0;

  for (let i = 0; i < seq.length; i++) {
    const resdName = seq[i] as ResdName;
    const atomNames = ATOM_NAMES[resdName];
    if (!atomNames) continue;

    const bfactor = plddt ? plddt[i] * 100 : 0;

    for (let a = 0; a < ATOM_NUM_MAX; a++) {
      if (!mask[i * ATOM_NUM_MAX + a]) continue;
      const name = atomNames[a];
      if (!name) continue;

      atomSerial++;
      const base = (i * ATOM_NUM_MAX + a) * 3;
      const x = coords[base];
      const y = coords[base + 1];
      const z = coords[base + 2];

      // Element symbol = first non-digit character of atom name
      const element = name.replace(/[0-9']/g, '')[0] || name[0];

      lines.push(formatAtomLine(
        atomSerial, name, resdName, chainId, i + 1,
        x, y, z, 1.0, bfactor, element,
      ));
    }
  }

  lines.push('END');
  return lines.join('\n') + '\n';
}

function formatAtomLine(
  serial: number, atomName: string, resName: string,
  chainId: string, resSeq: number,
  x: number, y: number, z: number,
  occupancy: number, bfactor: number, element: string,
): string {
  // PDB ATOM record format, 80 chars
  const record = 'ATOM  ';
  const serialStr = serial.toString().padStart(5);
  // Atom name: left-justified in 4-char field if name length < 4, else from col 13
  const nameStr = atomName.length < 4 ? (' ' + atomName).padEnd(4) : atomName.padEnd(4);
  const altLoc = ' ';
  const resNameStr = resName.padEnd(3);
  const chainStr = chainId;
  const resSeqStr = resSeq.toString().padStart(4);
  const iCode = ' ';

  return (
    record +
    serialStr +
    ' ' +
    nameStr +
    altLoc +
    resNameStr +
    ' ' +
    chainStr +
    resSeqStr +
    iCode +
    '   ' +
    x.toFixed(3).padStart(8) +
    y.toFixed(3).padStart(8) +
    z.toFixed(3).padStart(8) +
    occupancy.toFixed(2).padStart(6) +
    bfactor.toFixed(2).padStart(6) +
    '          ' +
    element.padStart(2) +
    '  '
  );
}
