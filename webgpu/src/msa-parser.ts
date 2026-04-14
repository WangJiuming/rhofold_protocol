/**
 * Parse aligned FASTA (.afa) MSA files.
 * Ported from rhofold/utils/alphabet.py read_msa() + remove_insertions()
 */

/**
 * Characters to strip from MSA sequences (insertions, dots, stars, spaces).
 * Lowercase letters represent insertions in aligned FASTA format.
 */
function removeInsertions(seq: string): string {
  let result = '';
  for (const ch of seq) {
    // Skip lowercase (insertions), '.', '*', spaces
    if (ch >= 'a' && ch <= 'z') continue;
    if (ch === '.' || ch === '*' || ch === ' ') continue;
    result += ch;
  }
  return result;
}

/**
 * Parse an aligned FASTA / MSA file content.
 * @param content - Raw text content of the .afa file
 * @param maxDepth - Maximum number of sequences to keep
 * @returns Array of aligned sequences (equal length, uppercase, T→U, insertions removed)
 */
export function parseMSA(content: string, maxDepth: number = 128): string[] {
  const sequences: string[] = [];
  let currentSeq = '';

  for (const line of content.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith('>')) {
      if (currentSeq) {
        sequences.push(removeInsertions(currentSeq).replace(/T/g, 'U'));
      }
      currentSeq = '';
      if (sequences.length >= maxDepth) break;
    } else {
      currentSeq += trimmed;
    }
  }
  // Don't forget the last sequence
  if (currentSeq && sequences.length < maxDepth) {
    sequences.push(removeInsertions(currentSeq).replace(/T/g, 'U'));
  }

  return sequences;
}

/**
 * Parse a FASTA file and return the first sequence.
 */
export function parseFASTA(content: string): { name: string; sequence: string } {
  const lines = content.trim().split('\n');
  let name = '';
  let seq = '';
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('>')) {
      if (seq) break; // Only take first sequence
      name = trimmed.slice(1).trim();
    } else {
      seq += trimmed;
    }
  }
  // Replace T→U, strip whitespace
  const sequence = seq.replace(/T/gi, 'U').replace(/\s/g, '').toUpperCase();
  return { name, sequence };
}
