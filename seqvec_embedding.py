import torch
import numpy as np
from transformers import BertModel, BertTokenizer
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
import os
import requests
import time

class SeqVecEmbedder:
    def __init__(self, model_name: str = "Rostlab/prot_bert"):
        """
        Initialize the SeqVec embedder using ProtBERT.
        
        Args:
            model_name: Name of the pretrained model to use
        """
        print(f"Loading model: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Using GPU for embeddings")
        else:
            print("Using CPU for embeddings")
            
        print("Model loaded successfully")
        
    def embed(self, sequence: str) -> np.ndarray:
        """
        Generate embeddings for a single protein sequence.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            numpy array of embeddings with shape [L x 1024] where L is sequence length
        """
        if not sequence or len(sequence) < 5:
            raise ValueError(f"Sequence too short: '{sequence}'")
            
        # Add spaces between amino acids as required by ProtBert
        sequence = " ".join(sequence)
        
        # Tokenize
        encoded = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            padding=True,
            return_tensors='pt'
        )
        
        # Move to same device as model
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state
            
        # Remove special tokens and convert to numpy
        embeddings = embeddings[:, 1:-1].cpu().numpy()
        return embeddings[0]  # Remove batch dimension

def fetch_sequence_from_uniprot(protein_id_or_seq):
    """
    Fetch protein sequence from UniProt or handle raw sequence input
    
    Args:
        protein_id_or_seq: Protein ID or raw sequence
        
    Returns:
        Tuple of (protein_id, sequence) or (None, None) if not found
    """
    # If input looks like a sequence (>40 chars, mostly amino acids), return it directly
    if len(protein_id_or_seq) > 40 and not protein_id_or_seq.startswith('>sp|'):
        return f"SEQ_{hash(protein_id_or_seq)}", protein_id_or_seq
        
    # If it's a FASTA header, extract the ID
    if protein_id_or_seq.startswith('>'):
        if '|' in protein_id_or_seq:
            protein_id = protein_id_or_seq.split('|')[1]
        else:
            protein_id = protein_id_or_seq[1:].split()[0]
    else:
        protein_id = protein_id_or_seq
    
    # Try UniProt API
    url = f"https://rest.uniprot.org/uniprotkb/search?query={protein_id}&format=fasta"
    
    try:
        response = requests.get(url)
        if response.status_code == 200 and response.text.strip():
            lines = response.text.strip().split('\n')
            if len(lines) < 2:
                print(f"No sequence found for {protein_id}")
                return None, None
                
            header = lines[0]
            if header.startswith('>'):
                # Extract UniProt ID from header
                uniprot_id = header.split('|')[1] if '|' in header else protein_id
                # Join remaining lines to get the sequence
                sequence = ''.join(lines[1:]).replace(' ', '')
                return uniprot_id, sequence
            
        print(f"Failed to retrieve sequence for {protein_id}")
        return None, None
    except Exception as e:
        print(f"Error fetching {protein_id}: {e}")
        return None, None

def read_protein_list(file_path):
    """
    Read protein IDs from a file, one per line
    
    Args:
        file_path: Path to the file with protein IDs
        
    Returns:
        List of protein IDs
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Protein list file not found: {file_path}")
        
    with open(file_path, 'r') as f:
        proteins = [line.strip() for line in f if line.strip()]
        
    print(f"Read {len(proteins)} protein IDs from {file_path}")
    return proteins

def read_fasta(fasta_file):
    """
    Read a FASTA file and return a dictionary of sequences.
    
    Args:
        fasta_file: Path to the FASTA file
        
    Returns:
        Dictionary with protein IDs as keys and sequences as values
    """
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
        
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]  # Extract ID part after '>'
                current_seq = []
            else:
                current_seq.append(line)
    
    # Add the last sequence if there is one
    if current_id and current_seq:
        sequences[current_id] = ''.join(current_seq)
        
    print(f"Read {len(sequences)} sequences from {fasta_file}")
    return sequences

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate protein embeddings from sequences')
    parser.add_argument('--input', type=str, default='/home/olarinoyem/Project/PPI_GNN1/my_new_dataset/processed/covid_protein_list.txt',
                     help='Path to input file (FASTA or protein list)')
    parser.add_argument('--output', type=str, default='/home/olarinoyem/Project/PPI_GNN1/my_new_dataset/processed/covid_embeddings.pkl',
                     help='Path to output pickle file')
    parser.add_argument('--mode', type=str, default='list', choices=['list', 'fasta'],
                     help='Input file mode: list of proteins or FASTA file')
    parser.add_argument('--limit', type=int, default=100,  # Changed default from 0 to 100
                     help='Limit processing to this many proteins (0 for all)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize the embedder
    embedder = SeqVecEmbedder()
    
    # Get sequences based on input mode
    sequences = {}
    
    if args.mode == 'fasta':
        # Read sequences from FASTA file
        sequences = read_fasta(args.input)
    else:
        # Read protein IDs from list file and fetch sequences
        protein_ids = read_protein_list(args.input)
        
        # Apply limit if specified
        if args.limit > 0:
            protein_ids = protein_ids[:args.limit]
            print(f"Processing first {args.limit} proteins")
            
        # Shuffle protein IDs for a random sample
        if args.limit > 0:
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(protein_ids)
            protein_ids = protein_ids[:args.limit]
            print(f"Processing random sample of {args.limit} proteins")
        
        # Fetch sequences for each protein
        print(f"Fetching sequences for {len(protein_ids)} proteins...")
        for protein_id in tqdm(protein_ids, desc="Fetching sequences"):
            uniprot_id, sequence = fetch_sequence_from_uniprot(protein_id)
            if sequence:
                sequences[uniprot_id or protein_id] = sequence
            time.sleep(1)  # Be nice to the UniProt server
    
    print(f"Retrieved {len(sequences)} sequences successfully")
    
    # Generate embeddings for each sequence
    embeddings = {}
    for protein_id, sequence in tqdm(sequences.items(), desc="Generating embeddings"):
        try:
            # Get sequence-level embeddings
            embedding = embedder.embed(sequence)
            
            # Store both the full embedding and a pooled version
            embeddings[protein_id] = {
                'sequence': sequence,
                'embedding': embedding,
                'pooled_embedding': np.mean(embedding, axis=0)  # Average pooling
            }
        except Exception as e:
            print(f"Error processing sequence {protein_id}: {e}")
    
    # Save the embeddings to a pickle file
    with open(args.output, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"Embeddings saved to {args.output}")
    print(f"Processed {len(embeddings)} out of {len(sequences)} sequences successfully")

if __name__ == "__main__":
    main()