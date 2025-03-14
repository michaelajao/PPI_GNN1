import os
import time
import requests
import random
import argparse
from Bio import SeqIO
from io import StringIO

def fetch_uniprot_sequence(protein_id):
    """Fetch protein sequence from UniProt by gene name/symbol"""
    # Updated URL for the new UniProt API
    url = f"https://rest.uniprot.org/uniprotkb/search?query={protein_id}&format=fasta"
    try:
        response = requests.get(url)
        if response.status_code == 200 and response.text.strip():
            # Parse the FASTA response
            fasta_io = StringIO(response.text)
            records = list(SeqIO.parse(fasta_io, "fasta"))
            if records:
                return records[0]
        return None
    except Exception as e:
        print(f"Error fetching {protein_id}: {e}")
        return None

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Fetch protein sequences from UniProt.')
    parser.add_argument('--limit', type=int, default=0, 
                        help='Number of proteins to process (0 = process all)')
    parser.add_argument('--random', action='store_true',
                        help='Select proteins randomly when using a limit')
    parser.add_argument('--input', type=str, 
                        default='/home/olarinoyem/Project/PPI_GNN1/my_new_dataset/processed/covid_protein_list.txt',
                        help='Path to the protein list file')
    parser.add_argument('--output', type=str,
                        default='/home/olarinoyem/Project/PPI_GNN1/my_new_dataset/processed/covid_proteins.fasta',
                        help='Path to output FASTA file')
    args = parser.parse_args()

    # Path to the protein list file
    protein_list_file = args.input
    output_fasta = args.output

    # Read the protein list
    with open(protein_list_file, 'r') as f:
        proteins = [line.strip() for line in f if line.strip()]
    
    total_proteins = len(proteins)
    print(f"Total proteins available: {total_proteins}")
    
    # Apply limit if specified
    if args.limit > 0:
        if args.limit > total_proteins:
            print(f"Warning: Requested limit ({args.limit}) exceeds available proteins ({total_proteins})")
            args.limit = total_proteins
            
        if args.random:
            print(f"Selecting {args.limit} proteins randomly...")
            proteins = random.sample(proteins, args.limit)
        else:
            print(f"Taking the first {args.limit} proteins...")
            proteins = proteins[:args.limit]

    print(f"Fetching sequences for {len(proteins)} proteins...")

    sequences = []
    for i, protein_id in enumerate(proteins):
        print(f"Processing {i+1}/{len(proteins)}: {protein_id}")
        record = fetch_uniprot_sequence(protein_id)
        if record:
            sequences.append(record)
        # Be nice to the UniProt server
        time.sleep(1)

    # Write sequences to FASTA file
    if sequences:
        with open(output_fasta, 'w') as f:
            SeqIO.write(sequences, f, "fasta")
        print(f"Successfully retrieved {len(sequences)}/{len(proteins)} sequences")
    else:
        print("No sequences were retrieved")

if __name__ == "__main__":
    main()