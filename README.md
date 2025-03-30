# PPI_GNN: Protein-Protein Interaction Prediction with Graph Neural Networks

This repository contains the implementation of a Graph Neural Network (GNN) approach for predicting protein-protein interactions (PPIs) from protein sequence and structure data.

## Overview

PPI_GNN uses graph-based deep learning to model proteins as graphs and predict their interactions. The model leverages:

- Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs), and Residual Graph Isomorphism Networks (ResGIN)
- Protein sequence embeddings from SeqVec
- Structural information encoded as node features and graph topology

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
pip install gdown matplotlib seaborn scikit-learn 
```


## Data Preparation

### Using Pre-processed Data

1. **Human PPI Dataset**:
   - Download the human features file from the link in `Human_features/README.md`
   - Place the files in `../human_features/processed/`

To download the pre-processed data, run the following command:

```python
python download_data.py
```   


2. **S. cerevisiae PPI Dataset**:
   - Download the input feature file from the link in `S. cerevisiae/README.md`
   - Place it in `../S. cerevisiae/processed/`

### Processing New Protein Data

To prepare your own protein dataset:

1. **Generate Embeddings**:
   ```bash
   python seqvec_embedding.py --input your_sequences.fasta
   ```

2. **Convert Proteins to Graphs**:
   ```bash
   python proteins_to_graphs.py --input embeddings.pkl --output protein_graphs.pkl
   ```

3. **Prepare Data for Model**:
   ```bash
   python data_prepare.py --input protein_graphs.pkl --output model_data.pt
   ```

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

```bash
python train.py --model GCNN --epochs 10 --lr 0.0005 --optimizer radam --scheduler plateau --early_stop 2 --save_dir report
```

### Running Model Experiments

To systematically evaluate different model architectures, optimizers, and schedulers:

```bash
python run_experiments.py --save_dir experiment_results --epochs 50 --early_stop 5
```

This will:
- Test both GCNN and AttGNN models
- Try all optimizers (adam, sgd, adagrad, radam)
- Try all schedulers (multistep, plateau, none)
- Test different learning rates (0.001, 0.0005, 0.0001)
- Run for up to 50 epochs with early stopping after 5 epochs without improvement
- Save all results in the experiment_results directory

For faster execution on systems with more memory, you can use parallel processing:

```bash
python run_experiments.py --save_dir experiment_results --epochs 50 --early_stop 5 --parallel
```

### Visualizing Experiment Results

After running the experiments, generate comparative visualizations:

```bash
python visualize_results.py --results_csv experiment_results/experiment_results.csv --output_dir experiment_results/visualizations
```

This will create comprehensive visualizations including:
- Optimizer comparisons across models
- Scheduler performance analysis
- Learning rate impact analysis
- Model performance comparisons
- Heatmaps showing best parameter combinations
- Top performing configurations

### Command Line Arguments

Available arguments for training:
- `--model`: Model architecture (`GCNN`, `AttGNN`, or `ResGIN`)
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

## Model Architectures

This repository provides three model architectures:

1. **GCNN**: Graph Convolutional Neural Network
   - Uses GCNConv layers for graph representation
   - Global mean pooling for graph-level representations

2. **AttGNN**: Graph Attention Network
   - Uses GAT layers with multi-head attention
   - Captures complex relationships between protein residues

3. **ResGIN**: Residual Graph Isomorphism Network
   - Combines Graph Isomorphism Networks (GIN) with residual connections
   - Features a bottleneck MLP design in each GIN layer (128→256→128)
   - Uses batch normalization and dropout for regularization
   - Mathematical formulation:
     ```
     h_i^(l+1) = h_i^(l) + GINConv(h_i^(l), N(i))
     ```
   - Provides stronger gradient flow and deeper feature extraction
   - Better captures structural patterns in protein graphs
   - 4 GIN layers with residual connections for hierarchical feature extraction
   - Parameter-efficient with approximately 280,000 trainable parameters
   - Theoretical advantages in graph structure distinction over traditional GCN

### Training the ResGIN Model

To train the ResGIN model:

```bash
python train.py --model ResGIN --epochs 50 --lr 0.0005 --optimizer radam --scheduler plateau --early_stop 5 --save_dir resgin_results
```

For a quick test run of the ResGIN model:

```bash
python train.py --model ResGIN --epochs 5 --lr 0.0005 --optimizer radam --scheduler plateau --early_stop 3 --save_dir resgin_test
```

## Visualization

The `train.py` script automatically generates visualizations in the `visualizations` directory:
- Learning curves (loss and accuracy)
- Confusion matrix
- ROC curve with AUROC
- Precision-Recall curve with AUPRC
- Performance metrics at different thresholds
- Results summary

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