# PPI_GNN: Protein-Protein Interaction Prediction with Graph Neural Networks

This repository contains the implementation of a Graph Neural Network (GNN) approach for predicting protein-protein interactions (PPIs) from protein sequence and structure data.

## Overview

Protein-Protein Interactions (PPIs) are fundamental to biological processes and understanding these interactions is crucial for therapeutic development and systems biology. This project uses graph-based deep learning to model proteins as graphs and predict their interactions. The approach leverages:

- **Graph Convolutional Networks (GCNs)** and **Graph Attention Networks (GATs)**
- **Protein sequence embeddings** from SeqVec/ProtBERT
- **Structural information** encoded as node features and graph topology

## Datasets

The project uses multiple protein-protein interaction datasets:

1. **Human PPI Dataset**: A comprehensive set of validated human protein-protein interactions
2. **S. cerevisiae (Yeast) Dataset**: Well-studied PPI dataset from yeast
3. **COVID-19 BioGRID Dataset**: Protein interactions between SARS-CoV-2 and human host proteins

## Installation

### Environment Setup

Create the conda environment using the provided configuration:

```bash
conda env create -f ppi_env.yml
conda activate ppi_env
```

Install PyTorch with CUDA support:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install additional packages:

```bash
pip install gdown matplotlib seaborn scikit-learn transformers requests
```

## Data Preparation

### Using Pre-processed Data

1. **Human PPI Dataset**:
   - Download the human features file from the link in `human_features/README.md`
   - Place the files in the `human_features/processed/` directory
   
   To download the pre-processed data, run the following command:
   ```python
   python download_data.py
   ```   

2. **S. cerevisiae PPI Dataset**:
   - Download the input feature file from the link in `S. cerevisiae/README.md`
   - Place it in the `S. cerevisiae/` directory

### Using the COVID-19 BioGRID Dataset

We've added support for the COVID-19 Coronavirus Project dataset from BioGRID. This dataset contains protein-protein interactions related to SARS-CoV-2 and human host proteins.

1. **Process the BioGRID interactions**:
   ```bash
   python process_covid_ppi.py
   ```
   This script extracts protein interaction pairs from the BioGRID dataset and saves them in the format required by the model.

2. **Fetch protein sequences** from UniProt:
   ```bash
   # Fetch sequences for all proteins
   python fetch_protein_sequences.py
   
   # Fetch sequences for a limited number of proteins
   python fetch_protein_sequences.py --limit 100
   
   # Fetch sequences for randomly selected proteins
   python fetch_protein_sequences.py --limit 100 --random
   ```
   
   Available options:
   - `--limit`: Number of proteins to process (0 = process all)
   - `--random`: Select proteins randomly when using a limit
   - `--input`: Path to the protein list file
   - `--output`: Path to output FASTA file

3. **Generate embeddings for protein sequences**:
   ```bash
   python seqvec_embedding.py --input my_new_dataset/processed/covid_proteins.fasta --output my_new_dataset/processed/covid_embeddings.pkl
   ```
   
   Available options:
   - `--input`: Path to input file (FASTA or protein list)
   - `--output`: Path to output pickle file with embeddings
   - `--mode`: Input file mode ('list' or 'fasta')
   - `--limit`: Limit processing to this many proteins (0 for all)

4. **Convert proteins to graphs**:
   ```bash
   python proteins_to_graphs.py --input my_new_dataset/processed/covid_embeddings.pkl --output my_new_dataset/processed/covid_protein_graphs.pkl
   ```

5. **Prepare data for model training**:
   ```bash
   python data_prepare.py --input my_new_dataset/processed/covid_protein_graphs.pkl --output my_new_dataset/processed/covid_model_data.pt
   ```

### Processing New Protein Data

To prepare your own protein dataset:

1. **Generate Embeddings**:
   ```bash
   python seqvec_embedding.py --input your_sequences.fasta --output your_embeddings.pkl
   ```

2. **Convert Proteins to Graphs**:
   ```bash
   python proteins_to_graphs.py --input your_embeddings.pkl --output your_protein_graphs.pkl
   ```

3. **Prepare Data for Model**:
   ```bash
   python data_prepare.py --input your_protein_graphs.pkl --output your_model_data.pt
   ```

## Model Architecture

This project implements two graph-based deep learning architectures:

1. **GCNN (Graph Convolutional Neural Network)**:
   - Uses GCNConv layers to learn node representations
   - Employs global mean pooling for graph-level representations
   - Predicts interaction probability with an MLP classification head

2. **AttGNN (Graph Attention Network)**:
   - Implements GAT layers with multi-head attention
   - Captures complex relationships between protein residues
   - Uses attention mechanisms to focus on important structural elements
   - Provides better interpretability through attention weights

## Usage

### Check GPU Availability

To check if your system can use GPU acceleration:

```bash
python gpu_check.py
```

### Training the Model

Train the model with default parameters:

```bash
python train.py
```

With custom parameters:

```bash
python train.py --model AttGNN --epochs 100 --lr 0.0005 --optimizer radam --scheduler plateau --early_stop 10 --save_dir visualizations
```

Available arguments:
- `--model`: Model architecture (`GCNN` or `AttGNN`)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--batch_size`: Batch size
- `--optimizer`: Optimizer type (`adam`, `sgd`, `adagrad`, `radam`)
- `--scheduler`: Learning rate scheduler (`multistep`, `plateau`, `stepLR`, `none`)
- `--early_stop`: Number of epochs for early stopping
- `--save_dir`: Directory to save the model
- `--seed`: Random seed for reproducibility

### Evaluating the Model

Evaluate the trained model on test data:

```bash
python test.py --model GCNN --model_path human_features/GCNN.pth
```

## Results and Visualization

The training process automatically generates visualizations in the `visualizations` directory:

- **Learning curves**: Training and validation loss/accuracy over epochs
- **Confusion matrix**: Displays true positives, false positives, true negatives, and false negatives
- **ROC curve**: Shows the trade-off between sensitivity and specificity with AUROC score
- **Precision-Recall curve**: Displays the precision-recall trade-off with AUPRC score
- **Performance metrics at different thresholds**: Helps in selecting optimal classification threshold
- **Results summary**: Text file with comprehensive evaluation metrics

## Project Structure

```
PPI_GNN/
├── data_prepare.py           # Data preprocessing script
├── download_data.py          # Downloads pre-processed datasets
├── fetch_protein_sequences.py # Fetches sequences from UniProt
├── gpu_check.py              # Checks GPU availability
├── metrics.py                # Evaluation metrics implementation
├── models.py                 # Neural network architecture definitions
├── ppi_env.yml               # Conda environment configuration
├── process_covid_ppi.py      # COVID-19 data processing script
├── proteins_to_graphs.py     # Converts proteins to graph representations
├── README.md                 # Project documentation
├── seqvec_embedding.py       # Generates protein sequence embeddings
├── test.py                   # Model evaluation script
├── train.py                  # Model training script
├── human_features/           # Human PPI dataset
├── my_new_dataset/           # COVID-19 dataset
├── S. cerevisiae/            # Yeast PPI dataset
└── visualizations/           # Generated plots and evaluation results
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

<!-- ## Citation

If you use this code in your research, please cite:
```
@article{ppi_gnn,
  title={PPI_GNN: Predicting Protein-Protein Interactions using Graph Neural Networks},
  author={Author1 and Author2},
  journal={Journal Name},
  year={20XX}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. -->