# note that this custom dataset is not prepared on the top of geometric Dataset(pytorch's inbuilt)
import os
import torch
import glob
import numpy as np 
import random
import math
from os import listdir
from os.path import isfile, join



processed_dir="human_features/processed/"
npy_file = "human_features/npy_file_new(human_dataset).npy"
npy_ar = np.load(npy_file)
print(npy_ar.shape)
print(f"Processed directory exists: {os.path.exists(processed_dir)}")
print(f"Number of .pt files: {len(glob.glob(os.path.join(processed_dir, '*.pt')))}")
print(f"First few protein IDs: {list(npy_ar[:,2][:5])}")

from torch.utils.data import Dataset as Dataset_n
from torch_geometric.data import DataLoader as DataLoader_n
from torch_geometric.data import Data

def bump(g):
    return Data.from_dict(g.__dict__)

class LabelledDataset(Dataset_n):
    def __init__(self, npy_file, processed_dir):
      self.npy_ar = np.load(npy_file)
      self.processed_dir = processed_dir
      self.protein_1 = self.npy_ar[:,2]
      self.protein_2 = self.npy_ar[:,5]
      self.label = self.npy_ar[:,6].astype(float)
      self.n_samples = self.npy_ar.shape[0]

    def __len__(self):
      return(self.n_samples)

    def __getitem__(self, index):
        # Get file paths
        prot_1_path = os.path.join(self.processed_dir, self.protein_1[index]+".pt")
        prot_2_path = os.path.join(self.processed_dir, self.protein_2[index]+".pt")
        
        # Check if files exist directly instead of using glob
        if not os.path.exists(prot_1_path):
            raise FileNotFoundError(f"Protein 1 file not found: {prot_1_path}")
        if not os.path.exists(prot_2_path):
            raise FileNotFoundError(f"Protein 2 file not found: {prot_2_path}")
            
        # Load the protein data and convert to new format
        try:
            prot_1 = bump(torch.load(prot_1_path))
            prot_2 = bump(torch.load(prot_2_path))
        except Exception as e:
            raise RuntimeError(f"Error loading protein files: {str(e)}")
            
        return prot_1, prot_2, torch.tensor(self.label[index])



dataset = LabelledDataset(npy_file = npy_file ,processed_dir= processed_dir)

final_pairs =  np.load(npy_file)
size = final_pairs.shape[0]
print("Size is : ")
print(size)
seed = 42
torch.manual_seed(seed)
#print(math.floor(0.8 * size))
#Make iterables using dataloader class  
trainset, testset = torch.utils.data.random_split(dataset, [math.floor(0.8 * size), size - math.floor(0.8 * size) ])
#print(trainset[0])
trainloader = DataLoader_n(dataset= trainset, batch_size= 4, num_workers = 0)
testloader = DataLoader_n(dataset= testset, batch_size= 4, num_workers = 0)
print("Length")
print(len(trainloader))
print(len(testloader))
