# # Install bio_embeddings using the command: pip install bio-embeddings[all]

# from bio_embeddings.embed import ProtTransBertBFDEmbedder,SeqVecEmbedder
# import numpy as np
# import torch 

# seq = 'MVTYDFGSDEMHD' # A protein sequence of length L

# embedder = SeqVecEmbedder()
# embedding = embedder.embed(seq)
# protein_embd = torch.tensor(embedding).sum(dim=0) # Vector with shape [L x 1024]
# np_arr = protein_embd.cpu().detach().numpy()

import torch
import numpy as np
from transformers import BertModel, BertTokenizer

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
            
        print("Model loaded successfully")
        
    def embed(self, sequence: str) -> np.ndarray:
        """
        Generate embeddings for a single protein sequence.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            numpy array of embeddings with shape [L x 1024] where L is sequence length
        """
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

# Usage example
def main():
    seq = 'MVTYDFGSDEMHD'  # A protein sequence of length L
    
    # Initialize the embedder
    embedder = SeqVecEmbedder()
    
    # Get embeddings
    embedding = embedder.embed(seq)
    print(f"Embedding shape: {embedding.shape}")  # Should be [L x 1024]
    
    # Sum across sequence length (alternative to your pooling)
    protein_embd = torch.tensor(embedding).sum(dim=0)
    print(f"Summed embedding shape: {protein_embd.shape}")  # Should be [1024]
    
    # Convert to numpy array
    np_arr = protein_embd.cpu().detach().numpy()
    print(f"Numpy array shape: {np_arr.shape}")
    
    return np_arr

if __name__ == "__main__":
    main()