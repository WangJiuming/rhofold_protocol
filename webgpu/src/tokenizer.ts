/**
 * MSA + RNA-FM tokenization.
 * Ported from rhofold/utils/alphabet.py and rhofold/model/rna_fm/data.py
 */

/** MSA Alphabet: used by E2Eformer (MSA row/col attention) */
const MSA_VOCAB: Record<string, number> = {
  '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3,
  'A': 4, 'U': 5, 'G': 6, 'C': 7, '-': 8,
};
const MSA_PAD_IDX = 1;

/** RNA-FM Alphabet: ESM-1b RNA variant. Note: C=5, U=7 (different from MSA) */
const FM_VOCAB: Record<string, number> = {
  '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3,
  'A': 4, 'C': 5, 'G': 6, 'U': 7,
  'R': 8, 'Y': 9, 'K': 10, 'M': 11, 'S': 12, 'W': 13,
  'B': 14, 'D': 15, 'H': 16, 'V': 17, 'N': 18, '-': 19,
};

export const MSA_PADDING_IDX = MSA_PAD_IDX;

/**
 * Tokenize aligned MSA sequences for the Embedder ONNX model.
 * - Prepends <cls>, then strips it (matching Python behavior)
 * - Pads shorter sequences with <pad>
 * - T→U replacement applied
 *
 * @returns BigInt64Array of shape [K, L] flattened row-major
 */
export function tokenizeMSA(sequences: string[], maxDepth: number = 128): { tokens: BigInt64Array; K: number; L: number } {
  const seqs = sequences.slice(0, maxDepth);
  const K = seqs.length;
  // All sequences should be aligned (same length after insertion removal)
  const L = seqs[0].length;

  const tokens = new BigInt64Array(K * L);
  for (let k = 0; k < K; k++) {
    const seq = seqs[k];
    for (let i = 0; i < L; i++) {
      const ch = i < seq.length ? seq[i].toUpperCase() : '<pad>';
      // The Python code: prepend CLS then strip it. Net effect: just tokenize the chars.
      const mapped = ch === 'T' ? 'U' : ch;
      tokens[k * L + i] = BigInt(MSA_VOCAB[mapped] ?? MSA_VOCAB['<unk>']);
    }
  }
  return { tokens, K, L };
}

/**
 * Tokenize a single RNA sequence for the RNA-FM ONNX model.
 * - Uses the ESM-1b RNA alphabet (A=4, C=5, G=6, U=7)
 * - Python prepends <cls> and appends <eos>, then strips both
 * - Net effect: just the nucleotide token IDs
 *
 * @returns BigInt64Array of shape [L]
 */
export function tokenizeRNAFM(sequence: string): BigInt64Array {
  const L = sequence.length;
  const tokens = new BigInt64Array(L);
  for (let i = 0; i < L; i++) {
    let ch = sequence[i].toUpperCase();
    if (ch === 'T') ch = 'U';
    tokens[i] = BigInt(FM_VOCAB[ch] ?? FM_VOCAB['<unk>']);
  }
  return tokens;
}
