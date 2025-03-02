# Environment Setup Instructions

This document provides instructions for setting up the environment required to run the PPI_GNN project.

## Requirements

- CUDA-compatible GPU (for faster training)
- Anaconda or Miniconda

## Basic Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PPI_GNN.git
   cd PPI_GNN
   ```

2. Create the conda environment using the provided configuration:
   ```bash
   conda env create -f environment.yml
   conda activate ppi_gnn
   ```

## Troubleshooting

### PyTorch Geometric Installation Issues

If you encounter issues with the PyTorch Geometric installation, you can install it manually:

```bash
# Find the correct versions for your CUDA and PyTorch setup
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
```

### CUDA Compatibility

If you're using a different CUDA version, you may need to modify the environment file to match your CUDA version for PyTorch.

## Additional Dependencies (Optional)

For advanced features or development, you might need additional packages:

```bash
# Install development tools
pip install black isort flake8 pytest

# Install visualization tools
pip install plotly

# Install additional analysis tools
pip install biopython
```

## Verifying Installation

Verify the setup by running:

```bash
python gpu_check.py
```

This will check if PyTorch can access your GPU and verify that the core dependencies are correctly installed.
