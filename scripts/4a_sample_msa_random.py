import argparse
from pathlib import Path
import shutil

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def read_fasta(filename):
    """
    Use BioPython's sequence parsing module to convert any file format to a Pandas DataFrame.

    The resulting DataFrame has the following columns:
        - name
        - id
        - description
        - sequence
    """

    # Prepare DataFrame fields
    data = {
        'id': [],
        'sequence': [],
        'description': [],
        'label': []
    }

    # Parse the fasta file
    for i, record in enumerate(SeqIO.parse(filename, format='fasta')):
        data['id'].append(record.id)
        data['sequence'].append(str(record.seq))
        data['description'].append(record.description)
        data['label'].append(record.name)

    # Port to DataFrame
    return pd.DataFrame(data)


def write_fasta(sampled_df, output_path):
    # Convert the sampled dataframe to biopython seqrecords
    seq_records = [SeqRecord(Seq(row['sequence']), id=row['id'], description=row['description']) for _, row in sampled_df.iterrows()]
    
    SeqIO.write(seq_records, output_path, format='fasta-2line')


def sample_sequences(input_path, output_dir, sample_rounds):
    
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_size = 128
    min_seq_num = 256

    # Load the MSA file
    alignment_df = read_fasta(input_path) 

    # If the alignment has fewer than the minimum number of sequences,
    # just copy the input file to the output directory for consistency
    if len(alignment_df) < min_seq_num:
        # copy the input file to the output directory with the same extension
        # this is because the input file could be in .a3m, .afa, .fa, etc.
        print(f'Input MSA has only {len(alignment_df)} sequences, which is fewer than the minimum required {min_seq_num}.')

        output_path = output_dir / f'original_msa{input_path.suffix}'
        
        print(f'Copying the original MSA file to {output_path} without sampling.')
        
        shutil.copy(input_path, output_path)
        
        return

    # Randomly sample sequences and save them
    for i in range(sample_rounds):
        sampled_df = alignment_df.sample(n=sample_size, random_state=42)

        # Save the sampled alignment
        output_path = output_dir / f"sample_{i+1:03d}{input_path.suffix}"
        write_fasta(sampled_df, output_path)
        
    print(f'Sampled {sample_rounds} sets of MSA of size {sample_size} from {input_path} and saved to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Sample sequences from an MSA file.')
    parser.add_argument('--msa', required=True, help='Path to the input MSA file.')
    parser.add_argument('--output-dir', required=True, help='Path to the output directory.')
    parser.add_argument('--sample-rounds', type=int, default=200, help='Number of times to sample.')

    args = parser.parse_args()

    sample_sequences(Path(args.msa), Path(args.output_dir), args.sample_rounds)


if __name__ == "__main__":
    main()

