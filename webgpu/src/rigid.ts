/**
 * Rigid body math: quaternion, rotation matrix, frame composition.
 * Ported from rhofold/utils/rigid_utils.py
 */

/**
 * Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix (row-major, flat).
 * Uses the same formula as rigid_utils.py quat_to_rot()
 */
export function quatToRotMat(q: ArrayLike<number>): Float32Array {
  const [w, x, y, z] = [q[0], q[1], q[2], q[3]];
  const rot = new Float32Array(9);
  // Row 0
  rot[0] = 1 - 2*(y*y + z*z);
  rot[1] = 2*(x*y - w*z);
  rot[2] = 2*(x*z + w*y);
  // Row 1
  rot[3] = 2*(x*y + w*z);
  rot[4] = 1 - 2*(x*x + z*z);
  rot[5] = 2*(y*z - w*x);
  // Row 2
  rot[6] = 2*(x*z - w*y);
  rot[7] = 2*(y*z + w*x);
  rot[8] = 1 - 2*(x*x + y*y);
  return rot;
}

/**
 * Parse Rigid.to_tensor_7() format: [qw, qx, qy, qz, tx, ty, tz]
 * Returns rotation matrix (3x3 flat) and translation (3).
 */
export function tensor7ToFrame(t7: ArrayLike<number>): { rot: Float32Array; trans: Float32Array } {
  // Normalize quaternion
  const qw = t7[0], qx = t7[1], qy = t7[2], qz = t7[3];
  const norm = Math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) || 1e-8;
  const nq = [qw/norm, qx/norm, qy/norm, qz/norm];
  return {
    rot: quatToRotMat(nq),
    trans: new Float32Array([t7[4], t7[5], t7[6]]),
  };
}

/**
 * Compose two rigid transforms: (R1,t1) * (R2,t2) = (R1@R2, R1@t2 + t1)
 * Rotation matrices are 3x3 flat row-major, translations are 3-vectors.
 */
export function composeFrames(
  r1: ArrayLike<number>, t1: ArrayLike<number>,
  r2: ArrayLike<number>, t2: ArrayLike<number>,
): { rot: Float32Array; trans: Float32Array } {
  const rot = new Float32Array(9);
  const trans = new Float32Array(3);

  // R_out = R1 @ R2
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        sum += r1[i*3+k] * r2[k*3+j];
      }
      rot[i*3+j] = sum;
    }
  }

  // t_out = R1 @ t2 + t1
  for (let i = 0; i < 3; i++) {
    trans[i] = t1[i];
    for (let k = 0; k < 3; k++) {
      trans[i] += r1[i*3+k] * t2[k];
    }
  }

  return { rot, trans };
}

/**
 * Build rotation matrix from torsion angle (cos, sin) — rotation around X-axis.
 * Matches calc_angl_rot_tsl in converter.py
 */
export function angleToRotMat(cos: number, sin: number): Float32Array {
  return new Float32Array([
    1,   0,    0,
    0,   cos, -sin,
    0,   sin,  cos,
  ]);
}

/**
 * Apply a rotation matrix + translation to a 3D point.
 * world = R @ local + t
 */
export function applyFrame(
  rot: ArrayLike<number>, trans: ArrayLike<number>,
  local: ArrayLike<number>,
): [number, number, number] {
  return [
    rot[0]*local[0] + rot[1]*local[1] + rot[2]*local[2] + trans[0],
    rot[3]*local[0] + rot[4]*local[1] + rot[5]*local[2] + trans[1],
    rot[6]*local[0] + rot[7]*local[1] + rot[8]*local[2] + trans[2],
  ];
}
