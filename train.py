import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse  # Add this import
import json
import datetime

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
    # print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Set GPU to use CuDNN for better performance if available
    torch.backends.cudnn.benchmark = True
    # For reproducibility with deterministic algorithms
    # torch.backends.cudnn.deterministic = True
else:
    print("CUDA is not available. Using CPU instead.")
    

# GPU memory tracking function
def get_gpu_info():
    """Get detailed GPU information"""
    if not torch.cuda.is_available():
        return "No GPU available"
    
    info = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info[f'gpu_{i}'] = {
            'name': torch.cuda.get_device_name(i),
            'compute_capability': f"{props.major}.{props.minor}",
            'total_memory': f"{props.total_memory / (1024**3):.2f} GB",
            'multi_processor_count': props.multi_processor_count
        }
    return info

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

# Enhanced visualization functions
def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_dir, model_name):
    """Plot learning curves for loss and accuracy"""
    plt.figure(figsize=(12, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'{model_name} - Loss Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} - Accuracy Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_learning_curves.png'), dpi=300)
    plt.close()
    print(f"Learning curves saved to {save_dir}/{model_name}_learning_curves.png")

def plot_confusion_matrix(labels, predictions, threshold, save_dir, model_name):
    """Plot confusion matrix"""
    pred_classes = pred_to_classes(labels, predictions, threshold)
    cm = confusion_matrix(labels, pred_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_dir}/{model_name}_confusion_matrix.png")

def plot_roc_curve(labels, predictions, save_dir, model_name):
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
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_roc_curve.png'), dpi=300)
    plt.close()
    print(f"ROC curve saved to {save_dir}/{model_name}_roc_curve.png")

def plot_performance_summary(metrics, save_dir, model_name):
    """Plot performance metrics in a bar chart"""
    plt.figure(figsize=(10, 6))
    
    metric_names = ['AUC', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'Specificity']
    metric_values = [
        metrics.get('auroc', 0),
        metrics.get('val_acc', 0) / 100 if metrics.get('val_acc', 0) > 1 else metrics.get('val_acc', 0),
        metrics.get('f1_score', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('specificity', 0)
    ]
    
    bars = plt.bar(metric_names, metric_values, color='steelblue', width=0.6)
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=12)
    
    plt.ylim(0, 1.1)  # Metrics are between 0 and 1
    plt.ylabel('Score')
    plt.title(f'{model_name} - Performance Summary')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_performance_summary.png'), dpi=300)
    plt.close()
    print(f"Performance summary saved to {save_dir}/{model_name}_performance_summary.png")

def generate_results_summary(metrics, experiment_settings, save_dir, model_name):
    """Generate textual summary of model results with experiment settings"""
    with open(os.path.join(save_dir, f'{model_name}_results_summary.txt'), 'w') as f:
        f.write(f"MODEL EVALUATION RESULTS: {model_name.upper()}\n")
        f.write("===============================\n\n")
        
        # Write experiment settings
        f.write("EXPERIMENT SETTINGS:\n")
        f.write("-------------------\n")
        for key, value in experiment_settings.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Write model metrics
        f.write("MODEL PERFORMANCE:\n")
        f.write("----------------\n")
        f.write(f"Best model saved at epoch: {metrics['epoch']}\n")
        f.write(f"Validation Loss (MSE): {metrics['val_loss']:.4f}\n")
        f.write(f"Validation Accuracy: {metrics['val_acc']:.2f}%\n")
        f.write(f"AUROC: {metrics['auroc']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall/Sensitivity: {metrics['recall']:.4f}\n")
        f.write(f"Specificity: {metrics['specificity']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"MCC: {metrics['mcc']:.4f}\n\n")
        
        f.write("Training completed with early stopping: ")
        f.write("Yes\n" if metrics['early_stopped'] else "No\n")
    
    print(f"Results summary saved to {save_dir}/{model_name}_results_summary.txt")

# Main training function with improved model-specific naming and visualizations
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
    
    # Model setup based on argument
    if args.model == 'GCNN':
        model = GCNN()
        model_name = 'GCNN'
    elif args.model == 'AttGNN':
        model = AttGNN()
        model_name = 'AttGNN'
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    model.to(device)
    
    # Verify model is on correct device
    print(f"Model type: {model_name}")
    print(f"Model is on device: {next(model.parameters()).device}")
    
    # Check if epochs is 0 or negative, handle gracefully
    if args.epochs <= 0:
        print(f"Warning: Number of epochs set to {args.epochs}, which is invalid.")
        print("Running initial evaluation but skipping training.")
        # Evaluate model without training
        val_labels, val_preds = predict(model, device, testloader)
        val_loss = get_mse(val_labels, val_preds)
        val_acc = get_accuracy(val_labels, val_preds, 0.5)
        auc_score = auroc(val_labels, val_preds)
        prec = precision(val_labels, val_preds, 0.5)
        rec = sensitivity(val_labels, val_preds, 0.5)
        spec = specificity(val_labels, val_preds, 0.5)
        f1 = f_score(val_labels, val_preds, 0.5)
        mcc_score = mcc(val_labels, val_preds, 0.5)
        
        # Initialize best_epoch_metrics with current evaluation
        best_epoch_metrics = {
            'epoch': 0,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'auroc': auc_score,
            'precision': prec,
            'recall': rec,
            'specificity': spec,
            'f1_score': f1,
            'mcc': mcc_score,
            'val_labels': val_labels,
            'val_preds': val_preds,
            'early_stopped': False
        }
        
        # Save untrained model
        model_path = os.path.join(args.save_dir, f"{model_name}_initial_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Untrained model saved to {model_path}")
        
        # Create empty lists for plotting
        train_losses = []
        val_losses = [val_loss]
        train_accs = []
        val_accs = [val_acc]
        
        experiment_settings = {
            'model': model_name,
            'epochs': args.epochs,
            'actual_epochs': 0,
            'learning_rate': args.lr,
            'optimizer': args.optimizer,
            'scheduler': args.scheduler,
            'early_stop': args.early_stop,
            'batch_size': args.batch_size,
            'device': device.type,
            'gpu_info': get_gpu_info() if torch.cuda.is_available() else "CPU only",
            'date_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Generate results summary
        generate_results_summary(best_epoch_metrics, experiment_settings, args.save_dir, model_name)
        
        return model, best_epoch_metrics, experiment_settings
    
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
        prec = precision(val_labels, val_preds, 0.5)
        rec = sensitivity(val_labels, val_preds, 0.5)
        spec = specificity(val_labels, val_preds, 0.5)
        f1 = f_score(val_labels, val_preds, 0.5)
        mcc_score = mcc(val_labels, val_preds, 0.5)
        
        print(f"Validation - loss: {val_loss:.4f}, accuracy: {val_acc:.2f}%, AUROC: {auc_score:.4f}")
        print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}, MCC: {mcc_score:.4f}")
        
        # Check for improvement in validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            # Save model with specific name
            model_path = os.path.join(args.save_dir, f"{model_name}_best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path} (best accuracy: {best_val_accuracy:.2f}%)")
            
            # Save best epoch metrics
            best_epoch_metrics = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'auroc': auc_score,
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
        
        # Plot intermediate learning curves every 10 epochs with model name
        if epoch % 10 == 0 or epoch == num_epochs:
            plot_learning_curves(train_losses, val_losses, train_accs, val_accs, 
                               args.save_dir, model_name)
            
        # Clear GPU cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Make sure best_epoch_metrics is initialized even if no training happened
    if not best_epoch_metrics:
        print("Warning: No training occurred or no improvement was found.")
        # Initialize with the last evaluation metrics
        val_labels, val_preds = predict(model, device, testloader)
        val_loss = get_mse(val_labels, val_preds)
        val_acc = get_accuracy(val_labels, val_preds, 0.5)
        auc_score = auroc(val_labels, val_preds)
        prec = precision(val_labels, val_preds, 0.5)
        rec = sensitivity(val_labels, val_preds, 0.5)
        spec = specificity(val_labels, val_preds, 0.5)
        f1 = f_score(val_labels, val_preds, 0.5)
        mcc_score = mcc(val_labels, val_preds, 0.5)
        
        best_epoch_metrics = {
            'epoch': 0,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'auroc': auc_score,
            'precision': prec,
            'recall': rec,
            'specificity': spec,
            'f1_score': f1,
            'mcc': mcc_score,
            'val_labels': val_labels,
            'val_preds': val_preds,
            'early_stopped': False
        }
    
    # Generate final visualizations using best model metrics
    val_labels = best_epoch_metrics['val_labels']
    val_preds = best_epoch_metrics['val_preds']
    
    # Plot final learning curves with model name
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs, 
                       args.save_dir, model_name)
    
    # Plot confusion matrix with model name
    plot_confusion_matrix(val_labels, val_preds, 0.5, args.save_dir, model_name)
    
    # Plot ROC curve with model name
    plot_roc_curve(val_labels, val_preds, args.save_dir, model_name)
    
    # Plot performance summary with model name
    plot_performance_summary(best_epoch_metrics, args.save_dir, model_name)
    
    # Collect experiment information - without environment details
    experiment_settings = {
        'model': model_name,
        'epochs': args.epochs,
        'actual_epochs': epoch if 'epoch' in locals() else 0,  # How many epochs were actually run
        'learning_rate': args.lr,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'early_stop': args.early_stop,
        'batch_size': args.batch_size,
        'device': device.type,
        'gpu_info': get_gpu_info() if torch.cuda.is_available() else "CPU only",
        'date_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Generate results summary with experiment settings
    generate_results_summary(best_epoch_metrics, experiment_settings, args.save_dir, model_name)
    
    print(f"\n{'='*60}")
    print(f"Training completed for {model_name}!")
    print(f"Best model at epoch {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Minimum validation loss: {min_val_loss:.4f}")
    print(f"All visualizations saved in {args.save_dir} directory")
    
    return model, best_epoch_metrics, experiment_settings

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    log_dir = os.path.join(args.save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Train the model and get metrics
    model, best_metrics, experiment_settings = train_and_visualize(args)
    
    # Save the model with its specific name and full configuration
    model_save_path = os.path.join(args.save_dir, f'{args.model}_best_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_metrics': best_metrics,
        'experiment_settings': experiment_settings
    }, model_save_path)
    
    # Save experiment settings separately as JSON for easier access
    with open(os.path.join(args.save_dir, f'{args.model}_experiment_settings.json'), 'w') as f:
        json.dump(experiment_settings, f, indent=4)

if __name__ == "__main__":
    main()