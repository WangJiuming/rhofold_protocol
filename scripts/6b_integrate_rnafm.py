import argparse

import fm
import torch
from Bio import SeqIO


parser = argparse.ArgumentParser(description='Integrate RNA-FM for embedding generation')
parser.add_argument('--fasta', type=str, help='Path to fasta file')

args = parser.parse_args()

# Load data from a FASTA file
fasta_path = args.fasta  #'./data/rnafm/rf02684/seqs.fasta'
records = list(SeqIO.parse(fasta_path, 'fasta'))

# Prepare the input data format
data = [(record.id, str(record.seq)) for record in records]

# Get the device, CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load RNA-FM model
fm_ckpt_path = ''  # specify the path here or omit
fm_model, alphabet = fm.pretrained.rna_fm_t12(fm_ckpt_path)
batch_converter = alphabet.get_batch_converter()

fm_model.to(device)  # use GPU if available

# Begin generating the embeddings
fm_model.eval()
batch_labels, batch_strs, batch_tokens = batch_converter(data)

print(f'Generating embeddings for {len(data)} sequences on device {device}')

with torch.no_grad():
    results = fm_model(batch_tokens.to(device), repr_layers=[12])
# results is a dictionary with keys: logits, representations
emb = results['representations'][12].cpu().numpy()  # emb should have shape (B, L+2, 640)

# Truncate the embeddings to remove the <SOS> and <EOS> tokens
emb = emb[:, 1:-1, :]  # after truncating, the shape becomes (B, L, 640)

print('Embedding has been generated (but not saved)')
print(f'Embedding has shape: {emb.shape}')  # (B, L, 640)
