/**
 * Convert frames + torsion angles to all-atom 3D coordinates.
 * Port of rhofold/utils/converter.py RNAConverter.build_cords
 */

import {
  ATOM_NUM_MAX, ATOM_INFOS, ATOM_NAMES, RESD_NAMES, RGRP_NAMES,
  RGRP_PARENT, TRANS_DICT_INIT, type ResdName,
} from './constants';
import { tensor7ToFrame, composeFrames, angleToRotMat, applyFrame } from './rigid';

export interface BuildCoordsResult {
  /** All-atom coordinates, shape [L*ATOM_NUM_MAX*3] flat */
  coords: Float32Array;
  /** Atom mask (1 = real atom, 0 = padding), shape [L*ATOM_NUM_MAX] flat */
  mask: Uint8Array;
  /** C1' coordinates for recycling, shape [L*3] flat */
  c1Coords: Float32Array;
}

/**
 * Build all-atom coordinates from frames and angles.
 *
 * @param seq - RNA sequence string (e.g. "GGCUCUGG...")
 * @param framesFlat - Flat Float32Array of shape [L, 7]: [qw,qx,qy,qz,tx,ty,tz] per residue
 * @param anglesFlat - Flat Float32Array of shape [L, 6, 2]: [cos,sin] per angle per residue
 * @returns coords [L*23*3], mask [L*23], c1Coords [L*3]
 */
export function buildCoords(
  seq: string,
  framesFlat: Float32Array,
  anglesFlat: Float32Array,
): BuildCoordsResult {
  const L = seq.length;
  const coords = new Float32Array(L * ATOM_NUM_MAX * 3);
  const mask = new Uint8Array(L * ATOM_NUM_MAX);
  const c1Coords = new Float32Array(L * 3);

  // Normalize angles: angl /= ||angl|| + eps
  const normAngles = new Float32Array(anglesFlat.length);
  for (let i = 0; i < L; i++) {
    for (let a = 0; a < 6; a++) {
      const base = (i * 6 + a) * 2;
      const cos = anglesFlat[base];
      const sin = anglesFlat[base + 1];
      const norm = Math.sqrt(cos * cos + sin * sin) + 1e-8;
      normAngles[base] = cos / norm;
      normAngles[base + 1] = sin / norm;
    }
  }

  // Parse backbone frames
  const backboneRot: Float32Array[] = new Array(L);
  const backboneTrans: Float32Array[] = new Array(L);
  for (let i = 0; i < L; i++) {
    const frame = tensor7ToFrame(framesFlat.subarray(i * 7, i * 7 + 7));
    backboneRot[i] = frame.rot;
    backboneTrans[i] = frame.trans;
  }

  // Process each residue type
  for (const resdName of RESD_NAMES) {
    // Find residue indices matching this type
    const indices: number[] = [];
    for (let i = 0; i < L; i++) {
      if (seq[i] === resdName) indices.push(i);
    }
    if (indices.length === 0) continue;

    const atomInfos = ATOM_INFOS[resdName];
    const transDict = TRANS_DICT_INIT[resdName];

    // Build rigid group frames for each residue
    for (const idx of indices) {
      const mainRot = backboneRot[idx];
      const mainTrans = backboneTrans[idx];

      // Compute frame for each rigid group
      const groupFrames: Map<string, { rot: Float32Array; trans: Float32Array }> = new Map();
      groupFrames.set('main', { rot: mainRot, trans: mainTrans });

      for (let rg = 0; rg < RGRP_NAMES.length; rg++) {
        const rgName = RGRP_NAMES[rg];
        const parentName = RGRP_PARENT[rgName];

        // Get parent frame
        let parentRot: ArrayLike<number>, parentTrans: ArrayLike<number>;
        const parentFrame = groupFrames.get(parentName);
        if (parentFrame) {
          parentRot = parentFrame.rot;
          parentTrans = parentFrame.trans;
        } else {
          parentRot = mainRot;
          parentTrans = mainTrans;
        }

        // Get reference transform from parent to this group
        const refKey = `${rgName}-${parentName}`;
        const ref = transDict[refKey];
        if (!ref) continue;

        const refRot = new Float32Array(ref.rot.flat());
        const refTsl = new Float32Array(ref.tsl);

        // Get angle rotation for this group
        const angleCos = normAngles[(idx * 6 + rg) * 2];
        const angleSin = normAngles[(idx * 6 + rg) * 2 + 1];
        const angleRot = angleToRotMat(angleCos, angleSin);
        const angleTsl = new Float32Array(3); // always zero

        // Compose: parent * ref * angle
        const intermediate = composeFrames(parentRot, parentTrans, refRot, refTsl);
        const final = composeFrames(intermediate.rot, intermediate.trans, angleRot, angleTsl);

        groupFrames.set(rgName, final);
      }

      // Place atoms
      for (let ai = 0; ai < atomInfos.length; ai++) {
        const info = atomInfos[ai];
        const rg = info.rigidGroup;

        // Determine which frame to use
        let frame: { rot: ArrayLike<number>; trans: ArrayLike<number> };
        if (rg === 0) {
          frame = { rot: mainRot, trans: mainTrans };
        } else {
          // Map rigidGroup index to RGRP_NAMES index:
          // 3→angl_0, 4→angl_1, 5→angl_2, 6→angl_3
          const rgName = RGRP_NAMES[rg - 1]; // rg 3 → index 2 → 'angl_0', etc.
          const groupFrame = groupFrames.get(rgName);
          if (!groupFrame) continue;
          frame = groupFrame;
        }

        const [wx, wy, wz] = applyFrame(frame.rot, frame.trans, info.localCoords);

        // Find the atom's position in the output tensor
        // atomInfos are NOT in the same order as ATOM_NAMES.
        // We need to find which slot this atom occupies in the padded output.
        const atomSlot = findAtomSlot(resdName, info.name);
        if (atomSlot < 0) continue;

        const coordBase = (idx * ATOM_NUM_MAX + atomSlot) * 3;
        coords[coordBase] = wx;
        coords[coordBase + 1] = wy;
        coords[coordBase + 2] = wz;
        mask[idx * ATOM_NUM_MAX + atomSlot] = 1;
      }

      // Extract C1' (always atom slot 1)
      c1Coords[idx * 3] = coords[(idx * ATOM_NUM_MAX + 1) * 3];
      c1Coords[idx * 3 + 1] = coords[(idx * ATOM_NUM_MAX + 1) * 3 + 1];
      c1Coords[idx * 3 + 2] = coords[(idx * ATOM_NUM_MAX + 1) * 3 + 2];
    }
  }

  return { coords, mask, c1Coords };
}

/** Atom slot lookup cache */
const atomSlotCache = new Map<string, number>();

function findAtomSlot(resdName: ResdName, atomName: string): number {
  const key = `${resdName}:${atomName}`;
  let slot = atomSlotCache.get(key);
  if (slot !== undefined) return slot;

  const nameList = ATOM_NAMES[resdName];
  slot = nameList.indexOf(atomName);
  atomSlotCache.set(key, slot);
  return slot;
}
