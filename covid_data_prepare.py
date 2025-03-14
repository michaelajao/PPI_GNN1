import os
import torch
import numpy as np
import argparse
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader  # Current version
from torch_geometric.data import Data
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='my_new_dataset/processed/covid_protein_graphs.pkl')
    parser.add_argument('--interactions', type=str, default='my_new_dataset/processed/covid_ppi_interactions.npy')
    parser.add_argument('--output', type=str, default='my_new_dataset/processed/covid_model_data.pt')
    return parser.parse_args()

def bump(g):
    return Data.from_dict(g.__dict__)

class CovidPPIDataset(Dataset):
    def __init__(self, graphs_file, interactions_file):
        # Load protein graphs
        with open(graphs_file, 'rb') as f:
            self.protein_graphs = pickle.load(f)
        
        # Load interactions with allow_pickle=True
        self.interactions = np.load(interactions_file, allow_pickle=True)
        self.protein_1 = self.interactions[:,0]
        self.protein_2 = self.interactions[:,1]
        self.labels = self.interactions[:,2].astype(float)
        self.n_samples = len(self.interactions)
        
        print(f"Loaded {self.n_samples} interactions with {len(self.protein_graphs)} proteins")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        prot_1_id = self.protein_1[index]
        prot_2_id = self.protein_2[index]
        
        # Get graph representations
        try:
            prot_1_data = self.protein_graphs.get(prot_1_id)
            prot_2_data = self.protein_graphs.get(prot_2_id)
            
            if prot_1_data is None or prot_2_data is None:
                raise ValueError(f"Missing protein data for {prot_1_id} or {prot_2_id}")
                
            return prot_1_data, prot_2_data, torch.tensor(self.labels[index])
            
        except Exception as e:
            print(f"Error processing interaction {index}: {e}")
            # Return a default item
            return self.protein_graphs[0], self.protein_graphs[0], torch.tensor(0.0)

def main():
    args = parse_args()
    
    # Create dataset
    dataset = CovidPPIDataset(args.input, args.interactions)
    
    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    # Generate indices for train/test split
    indices = list(range(len(dataset)))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Save the processed data - store indices instead of DataLoader objects
    torch.save({
        'train_indices': train_indices,
        'test_indices': test_indices,
        'dataset': dataset,
        'train_size': train_size,
        'test_size': test_size
    }, args.output)
    
    print(f"Data prepared and saved to {args.output}")
    print(f"Train samples: {train_size}")
    print(f"Test samples: {test_size}")

if __name__ == "__main__":
    main()