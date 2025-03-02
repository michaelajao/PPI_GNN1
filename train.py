import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse  # Add this import

from tqdm import tqdm
import math

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns
import torch_optimizer as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR
from metrics import *



import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from data_prepare import dataset, trainloader, testloader
from models import GCNN, AttGNN
from torch_geometric.data import DataLoader as DataLoader_n


# Set device with more explicit GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Set GPU to use CuDNN for better performance if available
    torch.backends.cudnn.benchmark = True
    # For reproducibility with deterministic algorithms
    # torch.backends.cudnn.deterministic = True
else:
    print("CUDA is not available. Using CPU instead.")
    

# GPU memory tracking function
def print_gpu_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"  Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Create directory for visualizations
os.makedirs("visualizations", exist_ok=True)

# Print dataset information
print("Dataset Information:")
print(f"Total samples: {len(dataset)}")
print(f"Training batches: {len(trainloader)}")
print(f"Testing batches: {len(testloader)}")

total_samples = len(dataset)
n_iterations = math.ceil(total_samples/5)

# Add argument parser function
def parse_arguments():
    parser = argparse.ArgumentParser(description='PPI-GNN: Training configuration')
    
    # Model selection
    parser.add_argument('--model', type=str, default='GCNN', choices=['GCNN', 'AttGNN'],
                        help='Model architecture to train (GCNN, AttGNN)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--early_stop', type=int, default=6,
                        help='Number of epochs with no improvement after which training will stop')
    
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adagrad', 'radam'],
                        help='Optimizer to use (adam, sgd, adagrad, radam)')
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'plateau', 'none'],
                        help='Learning rate scheduler (multistep, plateau, none)')
    
    # Other settings
    parser.add_argument('--save_dir', type=str, default='human_features',
                        help='Directory to save model and results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

# Training function with GPU optimizations
def train(model, device, trainloader, optimizer, epoch, scheduler=None):
    print(f'Training on {len(trainloader)} batches...')
    model.train()
    loss_func = nn.MSELoss()
    predictions_tr = torch.Tensor().to(device)
    labels_tr = torch.Tensor().to(device)
    
    total_loss = 0.0
    batch_count = 0
    
    # Use tqdm for progress bar
    with tqdm(trainloader, unit="batch") as tepoch:
        for count, (prot_1, prot_2, label) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            
            # Ensure data is on GPU
            prot_1 = prot_1.to(device)
            prot_2 = prot_2.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            output = model(prot_1, prot_2)
            
            # Keep predictions on GPU to avoid unnecessary transfers
            predictions_tr = torch.cat((predictions_tr, output), 0)
            labels_tr = torch.cat((labels_tr, label.view(-1, 1)), 0)
            
            loss = loss_func(output, label.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            batch_count += 1
            tepoch.set_postfix(loss=loss.item(), 
                              gpu_mem=f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" 
                              if torch.cuda.is_available() else "N/A")
            
            # Optional: Clear GPU cache periodically for large datasets
            if count % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Step scheduler if provided
    if scheduler is not None:
        scheduler.step()
    
    # Move to CPU for numpy operations
    labels_tr_np = labels_tr.cpu().detach().numpy()
    predictions_tr_np = predictions_tr.cpu().detach().numpy()
    
    # Calculate training metrics
    acc_tr = get_accuracy(labels_tr_np, predictions_tr_np, 0.5)
    loss_tr = total_loss / batch_count
    
    print(f'Epoch {epoch} - train_loss: {loss_tr:.4f} - train_accuracy: {acc_tr:.2f}%')
    
    # Optional: Print GPU memory usage
    if torch.cuda.is_available():
        print_gpu_memory_usage()
    
    return loss_tr, acc_tr, labels_tr_np, predictions_tr_np

# Prediction function with GPU optimizations
def predict(model, device, loader):
    model.eval()
    predictions = torch.Tensor().to(device)
    labels = torch.Tensor().to(device)
    
    with torch.no_grad():
        for prot_1, prot_2, label in loader:
            # Ensure data is on GPU
            prot_1 = prot_1.to(device)
            prot_2 = prot_2.to(device)
            label = label.to(device)
            
            output = model(prot_1, prot_2)
            predictions = torch.cat((predictions, output), 0)
            labels = torch.cat((labels, label.view(-1, 1)), 0)
    
    # Move to CPU for numpy operations
    return labels.cpu().numpy().flatten(), predictions.cpu().numpy().flatten()

# Visualization functions
def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot learning curves for loss and accuracy"""
    plt.figure(figsize=(12, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('visualizations/learning_curves.png')
    plt.close()
    print("Learning curves saved to visualizations/learning_curves.png")

def plot_confusion_matrix(labels, predictions, threshold=0.5):
    """Plot confusion matrix"""
    pred_classes = pred_to_classes(labels, predictions, threshold)
    cm = confusion_matrix(labels, pred_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to visualizations/confusion_matrix.png")

def plot_roc_curve(labels, predictions):
    """Plot ROC curve with AUC score"""
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    auc_score = auroc(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('visualizations/roc_curve.png')
    plt.close()
    print("ROC curve saved to visualizations/roc_curve.png")

def plot_pr_curve(labels, predictions):
    """Plot Precision-Recall curve with AUPRC score"""
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    auprc_score = auprc(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'PR curve (AUPRC = {auprc_score:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('visualizations/pr_curve.png')
    plt.close()
    print("PR curve saved to visualizations/pr_curve.png")

def plot_metrics_at_thresholds(labels, predictions):
    """Plot various metrics across different thresholds"""
    thresholds = np.arange(0.1, 1.0, 0.1)
    accuracies = []
    precisions = []
    recalls = []
    specificities = []
    f1_scores = []
    mccs = []
    
    for threshold in thresholds:
        accuracies.append(get_accuracy(labels, predictions, threshold))
        precisions.append(precision(labels, predictions, threshold))
        recalls.append(sensitivity(labels, predictions, threshold))
        specificities.append(specificity(labels, predictions, threshold))
        f1_scores.append(f_score(labels, predictions, threshold))
        mccs.append(mcc(labels, predictions, threshold))
    
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, accuracies, '-o', label='Accuracy')
    plt.plot(thresholds, precisions, '-o', label='Precision')
    plt.plot(thresholds, recalls, '-o', label='Recall/Sensitivity')
    plt.plot(thresholds, specificities, '-o', label='Specificity')
    plt.plot(thresholds, f1_scores, '-o', label='F1 Score')
    plt.plot(thresholds, mccs, '-o', label='MCC')
    plt.xlabel('Threshold')
    plt.ylabel('Value')
    plt.title('Metrics at Different Thresholds')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('visualizations/metrics_vs_thresholds.png')
    plt.close()
    print("Threshold metrics plot saved to visualizations/metrics_vs_thresholds.png")

def generate_results_summary(best_epoch_metrics):
    """Generate textual summary of best model results"""
    with open('visualizations/results_summary.txt', 'w') as f:
        f.write("MODEL EVALUATION RESULTS SUMMARY\n")
        f.write("===============================\n\n")
        
        f.write(f"Best model saved at epoch: {best_epoch_metrics['epoch']}\n")
        f.write(f"Validation Loss (MSE): {best_epoch_metrics['val_loss']:.4f}\n")
        f.write(f"Validation Accuracy: {best_epoch_metrics['val_acc']:.2f}%\n")
        f.write(f"AUROC: {best_epoch_metrics['auroc']:.4f}\n")
        f.write(f"AUPRC: {best_epoch_metrics['auprc']:.4f}\n\n")
        
        f.write("Metrics at threshold 0.5:\n")
        f.write(f"  Precision: {best_epoch_metrics['precision']:.4f}\n")
        f.write(f"  Recall/Sensitivity: {best_epoch_metrics['recall']:.4f}\n")
        f.write(f"  Specificity: {best_epoch_metrics['specificity']:.4f}\n")
        f.write(f"  F1 Score: {best_epoch_metrics['f1_score']:.4f}\n")
        f.write(f"  MCC: {best_epoch_metrics['mcc']:.4f}\n\n")
        
        f.write("Training completed with early stopping: ")
        f.write("Yes\n" if best_epoch_metrics['early_stopped'] else "No\n")
    
    print("Results summary saved to visualizations/results_summary.txt")

# Main training loop with visualizations
def train_and_visualize(args):
    # Set random seed for reproducibility
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Model setup based
    if args.model == 'GCNN':
        model = GCNN()
        model_name = 'GCN'
    elif args.model == 'AttGNN':
        model = AttGNN()
        model_name = 'AttGNN'
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    model.to(device)
    
    # Verify model is on correct device
    print(f"Model type: {args.model}")
    print(f"Model is on device: {next(model.parameters()).device}")
    
    # Hyperparameters
    num_epochs = args.epochs
    learning_rate = args.lr
    
    # Early stopping parameters
    n_epochs_stop = args.early_stop
    epochs_no_improve = 0
    early_stop = False
    
    # Loss function and optimizer
    loss_func = nn.MSELoss()
    
    # Setup optimizer based on command line argument
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif args.optimizer == 'radam':
        optimizer = optim.RAdam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Setup scheduler based on command line argument
    if args.scheduler == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=[int(num_epochs*0.3), int(num_epochs*0.6), int(num_epochs*0.8)], gamma=0.5)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    elif args.scheduler == 'stepLR':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5, verbose=True)
    else:
        scheduler = None
    
    # Tracking variables
    min_val_loss = float('inf')
    best_val_accuracy = 0
    best_epoch = 0
    best_epoch_metrics = {}
    
    # History tracking
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Print initial GPU status
    if torch.cuda.is_available():
        print_gpu_memory_usage()
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*20} Epoch {epoch}/{num_epochs} {'='*20}")
        
        # Train
        train_loss, train_acc, train_labels, train_preds = train(model, device, trainloader, optimizer, epoch, 
                                                               scheduler if args.scheduler != 'plateau' else None)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Sync CUDA for accurate timing if using GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Evaluate
        val_labels, val_preds = predict(model, device, testloader)
        val_loss = get_mse(val_labels, val_preds)
        val_acc = get_accuracy(val_labels, val_preds, 0.5)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Step plateau scheduler here since it depends on validation metrics
        if args.scheduler == 'plateau' and scheduler is not None:
            scheduler.step(val_loss)
        
        # Calculate additional metrics
        auc_score = auroc(val_labels, val_preds)
        auprc_score = auprc(val_labels, val_preds)
        prec = precision(val_labels, val_preds, 0.5)
        rec = sensitivity(val_labels, val_preds, 0.5)
        spec = specificity(val_labels, val_preds, 0.5)
        f1 = f_score(val_labels, val_preds, 0.5)
        mcc_score = mcc(val_labels, val_preds, 0.5)
        
        print(f"Validation - loss: {val_loss:.4f}, accuracy: {val_acc:.2f}%, AUROC: {auc_score:.4f}, AUPRC: {auprc_score:.4f}")
        print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}, MCC: {mcc_score:.4f}")
        
        # Check for improvement in validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            # Save model
            model_path = os.path.join(args.save_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path} (best accuracy: {best_val_accuracy:.2f}%)")
            
            # Save best epoch metrics
            best_epoch_metrics = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'auroc': auc_score,
                'auprc': auprc_score,
                'precision': prec,
                'recall': rec,
                'specificity': spec,
                'f1_score': f1,
                'mcc': mcc_score,
                'val_labels': val_labels,
                'val_preds': val_preds,
                'early_stopped': False
            }
        
        # Check for improvement in validation loss (for early stopping)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stopping check
        if epoch > 5 and epochs_no_improve == n_epochs_stop:
            print(f'Early stopping triggered at epoch {epoch}')
            best_epoch_metrics['early_stopped'] = True
            early_stop = True
            break
        
        # Plot intermediate results every 5 epochs
        if epoch % 5 == 0:
            plot_learning_curves(train_losses, val_losses, train_accs, val_accs)
            
        # Clear GPU cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Plot final learning curves
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs)
    
    # Generate final visualizations using best model metrics
    val_labels = best_epoch_metrics['val_labels']
    val_preds = best_epoch_metrics['val_preds']
    
    plot_confusion_matrix(val_labels, val_preds)
    plot_roc_curve(val_labels, val_preds)
    plot_pr_curve(val_labels, val_preds)
    plot_metrics_at_thresholds(val_labels, val_preds)
    
    # Generate results summary
    generate_results_summary(best_epoch_metrics)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best model at epoch {best_epoch}/{num_epochs}")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Minimum validation loss: {min_val_loss:.4f}")
    print(f"All visualizations saved in the 'visualizations' directory")
    
    # Final GPU status
    if torch.cuda.is_available():
        print_gpu_memory_usage()
        
    print(f"{'='*60}")
    
    return model, best_epoch_metrics

if __name__ == "__main__":
    args = parse_arguments()
    print("Training with the following configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    model, best_metrics = train_and_visualize(args)