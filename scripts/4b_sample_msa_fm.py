import argparse
from pathlib import Path

import torch
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import math
import fm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm


# Load the RNA-FM model globally
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

fm_model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()

fm_model.eval().to(device)


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


def compute_embeddings(seqs):
    # Initialize list to store embeddings
    emb_list = []
    
    # Compute embeddings for each sequence in the alignment
    for seq in tqdm(seqs):
        batch_labels, batch_strs, data = batch_converter([('', seq)])
        with torch.no_grad():
            results = fm_model(data.to(device), repr_layers=[12])
            emb = results['representations'][12]
            
            # keep the BOS token
            emb_list.append(emb[0, 0, :].cpu().numpy())
    
    return np.array(emb_list)


def cluster_sequences(embeddings, n_clusters):
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(embeddings)
    
    return labels


def generate_clusters(input_path, output_dir, n_clusters=None):
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the input file into a DataFrame
    df = read_fasta(input_path)

    # Extract sequences from the DataFrame
    sequences = df['sequence'].tolist()

    # If the number of sequences is less than 256, write the file directly to the output directory
    if len(sequences) < 256:
        print(f'Input MSA has only {len(sequences)} sequences, which is fewer than the minimum required 256.')
        print(f'Copying the original MSA file to {output_path} without sampling.')
        
        output_path = output_dir / f'original_msa{input_path.suffix}'
        write_fasta(df, output_path)
        return

    # Compute embeddings for each sequence
    print(f'Computing embeddings for {len(sequences):,} sequences')
    
    embeddings = compute_embeddings(sequences)

    # If n_clusters is not provided, set it to num_seq / 64
    if n_clusters is None:
        n_clusters = max(1, math.ceil(len(sequences) / 64.0))
    
    print(f'Clustering into {n_clusters} clusters')

    # Perform clustering on the embeddings
    labels = cluster_sequences(embeddings, n_clusters)

    # Generate new MSAs for each cluster
    for i in range(n_clusters):
        cluster_df = df[labels == i]
        output_path = output_dir / f'cluster_{i+1:03d}{input_path.suffix}'
        write_fasta(cluster_df, output_path)
        
    print(f'Generated {n_clusters} clusters and saved to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Generate clustered MSAs from a single MSA file.')
    parser.add_argument('--msa', required=True, help='Path to the input MSA file.')
    parser.add_argument('--output-dir', required=True, help='Path to the output directory.')
    parser.add_argument('--cluster-num', type=int, default=None, help='Number of clusters to generate. If not provided, it will be set to num_seq / 64.')
    
    args = parser.parse_args()

    generate_clusters(Path(args.msa), Path(args.output_dir), args.cluster_num)

if __name__ == '__main__':
    main()

