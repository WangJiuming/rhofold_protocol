import argparse
from pathlib import Path

import fm
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Integrate RNA-FM for embedding generation.')
    parser.add_argument('--fasta', type=str, help='Path to the input fasta file containing one sequence.')
    parser.add_argument('--output-dir', type=str, help='Path to the output directory for saving the npy embedding file.')
    parser.add_argument('--use-batch', action='store_true', help='Flag of whether to generate the embeddings in batches, which will result in padded sequence embeddings if the sequences are not of the same length.')
    parser.add_argument('--batch-size',type=int, default=None, help='The batch size when generating embeddings in batches. Default to processing all sequences in one batch if not provided.')

    args = parser.parse_args()

    # Load data from a FASTA file
    fasta_path = Path(args.fasta)  #'./data/rnafm/rf02684/seqs.fasta'
    output_dir = Path(args.output_dir)

    name = fasta_path.stem

    records = list(SeqIO.parse(fasta_path, 'fasta'))

    # Prepare the input data format
    data_list = [(record.id, str(record.seq)) for record in records]

    # Get the device, CPU or GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load RNA-FM model
    fm_ckpt_path = ''  # specify the path here or omit
    fm_model, alphabet = fm.pretrained.rna_fm_t12(fm_ckpt_path)
    batch_converter = alphabet.get_batch_converter()

    fm_model.to(device)  # use GPU if available

    # Begin generating the embeddings
    fm_model.eval()

    if args.use_batch:
        batch_size = args.batch_size if args.batch_size else len(data_list)
        
        emb_list = []
        
        print(f'Generating embeddings in batches of size {batch_size}')
        
        for i in tqdm(range(0, len(data_list), batch_size)):
            data_batch = data_list[i:i+batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(data_batch)

            with torch.no_grad():
                results = fm_model(batch_tokens.to(device), repr_layers=[12])

            # Results is a dictionary with keys: logits, representations
            emb = results['representations'][12].cpu().numpy()  # emb should have shape (B, L+2, 640)

            # print(f'Embedding generated with shape: {emb.shape}')  # (B, L, 640)

            emb_list.append(emb)

        # get the pad index
        pad_idx = alphabet.padding_idx
        
        # pad the embeddings to the max length in the batch
        max_len = max(emb.shape[1] for emb in emb_list)
        padded_emb_list = []
        for emb in emb_list:
            if emb.shape[1] < max_len:
                pad_width = ((0, 0), (0, max_len - emb.shape[1]), (0, 0))  # pad only the sequence length dimension
                emb = np.pad(emb, pad_width, mode='constant', constant_values=pad_idx)
            padded_emb_list.append(emb)
            
        # concatenate all batches
        combined_emb = np.concatenate(padded_emb_list, axis=0)  # (N, L_max, 640)
        print(f'All embeddings concatenated with shape: {combined_emb.shape}, padding token index: {pad_idx}')

        # Save the embedding
        output_npy_path = output_dir / f'{name}_rnafm_embedding.npy'
        np.save(output_npy_path, combined_emb)
    
    else:
        # generate the embeddings for each single sequence
        print(f'Generating embeddings for {len(data_list)} sequences individually')
        
        for data in tqdm(data_list):
            batch_labels, batch_strs, batch_tokens = batch_converter([data])
            
            with torch.no_grad():
                results = fm_model(batch_tokens.to(device), repr_layers=[12])

            emb = results['representations'][12].cpu().numpy()[0]  # emb should have shape (L+2, 640)
            
            output_npy_path = output_dir / f'{name}_rnafm_embedding' / f'{data[0]}.npy'
            output_npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_npy_path, emb)
            
            # tqdm.write(f'Embedding generated with shape: {emb.shape}')
            # tqdm.write(f'Embedding saved at {output_npy_path}')

    print(f'All embeddings generated and saved to {output_dir}')

