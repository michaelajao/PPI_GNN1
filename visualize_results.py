import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from matplotlib.colors import LinearSegmentedColormap

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize PPI-GNN experiment results')
    
    parser.add_argument('--results_csv', type=str, default='experiment_results/experiment_results.csv',
                        help='Path to the experiment results CSV file')
    parser.add_argument('--output_dir', type=str, default='experiment_results/visualizations',
                        help='Directory to save visualization outputs')
    
    return parser.parse_args()

def prepare_data(df):
    """Prepare the dataframe for visualization"""
    # Handle non-numeric columns if needed
    numeric_cols = ['val_loss', 'val_acc', 'auroc', 'precision', 'recall', 
                    'specificity', 'f1_score', 'mcc', 'actual_epochs', 'duration_minutes']
    
    # Make sure numeric columns are properly typed
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def plot_optimizer_comparison(df, output_dir):
    """Plot performance by optimizer type"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance by Optimizer', fontsize=16)
    
    metrics = ['f1_score', 'auroc', 'precision', 'recall']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        sns.boxplot(x='optimizer', y=metric, hue='model', data=df, ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Optimizer')
        ax.set_ylabel(metric.replace('_', ' ').title())
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimizer_comparison.png'), dpi=300)
    plt.close()

def plot_scheduler_comparison(df, output_dir):
    """Plot performance by scheduler type"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance by Scheduler', fontsize=16)
    
    metrics = ['f1_score', 'auroc', 'precision', 'recall']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        sns.boxplot(x='scheduler', y=metric, hue='model', data=df, ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Scheduler')
        ax.set_ylabel(metric.replace('_', ' ').title())
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scheduler_comparison.png'), dpi=300)
    plt.close()

def plot_lr_comparison(df, output_dir):
    """Plot performance by learning rate"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance by Learning Rate', fontsize=16)
    
    metrics = ['f1_score', 'auroc', 'precision', 'recall']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        sns.boxplot(x='learning_rate', y=metric, hue='model', data=df, ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel(metric.replace('_', ' ').title())
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_comparison.png'), dpi=300)
    plt.close()

def plot_model_comparison(df, output_dir):
    """Plot overall model performance comparison"""
    metrics = ['f1_score', 'auroc', 'precision', 'recall', 'specificity', 'val_acc']
    
    # Calculate mean of each metric grouped by model
    model_means = df.groupby('model')[metrics].mean().reset_index()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bar_width = 0.35
    index = np.arange(len(metrics))
    
    models = model_means['model'].unique()
    
    for i, model in enumerate(models):
        model_data = model_means[model_means['model'] == model]
        values = [model_data[metric].values[0] for metric in metrics]
        ax.bar(index + i*bar_width, values, bar_width, label=model)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Average Performance by Model')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()

def plot_heatmap(df, output_dir):
    """Create heatmaps for each model showing optimizer/scheduler combinations"""
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        
        # Create pivot tables for different metrics
        for metric in ['f1_score', 'auroc']:
            pivot = model_df.pivot_table(
                values=metric, 
                index='optimizer', 
                columns='scheduler',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
            plt.title(f'{model_name}: {metric.replace("_", " ").title()} by Optimizer and Scheduler')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_{metric}_heatmap.png'), dpi=300)
            plt.close()
    
    # Create heatmaps for learning rates as well
    for model_name in df['model'].unique():
        for metric in ['f1_score', 'auroc']:
            pivot = df[df['model'] == model_name].pivot_table(
                values=metric,
                index='optimizer',
                columns='learning_rate',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
            plt.title(f'{model_name}: {metric.replace("_", " ").title()} by Optimizer and Learning Rate')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_{metric}_lr_heatmap.png'), dpi=300)
            plt.close()

def plot_top_configurations(df, output_dir):
    """Plot the top 5 configurations by F1 score for each model"""
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name].sort_values('f1_score', ascending=False).head(5)
        
        # Create a descriptive label for each configuration
        model_df['config'] = model_df.apply(
            lambda x: f"{x['optimizer']}\n{x['scheduler']}\nLR={x['learning_rate']}", 
            axis=1
        )
        
        plt.figure(figsize=(12, 8))
        
        metrics = ['f1_score', 'auroc', 'precision', 'recall', 'specificity']
        x = np.arange(len(model_df))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, model_df[metric], width, label=metric.replace('_', ' ').title())
        
        plt.xlabel('Configuration')
        plt.ylabel('Score')
        plt.title(f'Top 5 Configurations for {model_name}')
        plt.xticks(x + width*2, model_df['config'])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_top_configs.png'), dpi=300)
        plt.close()

def main():
    args = parse_arguments()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the results CSV
    df = pd.read_csv(args.results_csv)
    
    # Prepare dataframe
    df = prepare_data(df)
    
    # Filter successful experiments only
    df = df[df['success'] == True].copy()
    
    if len(df) == 0:
        print("No successful experiments found in the results file.")
        return
    
    # Create the visualizations
    plot_optimizer_comparison(df, args.output_dir)
    plot_scheduler_comparison(df, args.output_dir)
    plot_lr_comparison(df, args.output_dir)
    plot_model_comparison(df, args.output_dir)
    plot_heatmap(df, args.output_dir)
    plot_top_configurations(df, args.output_dir)
    
    print(f"Visualizations have been saved to {args.output_dir}")

if __name__ == "__main__":
    main()