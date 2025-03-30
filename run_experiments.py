import subprocess
import os
import pandas as pd
import json
import time
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run multiple PPI-GNN experiments')
    
    parser.add_argument('--save_dir', type=str, default='experiment_results',
                        help='Directory to save experiment results')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--early_stop', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--parallel', action='store_true',
                        help='Run experiments in parallel (requires more memory)')
    
    return parser.parse_args()

def run_experiment(model, optimizer, scheduler, lr, save_dir, epochs, early_stop):
    """Run a single experiment with the given parameters"""
    experiment_name = f"{model}_{optimizer}_{scheduler}_lr{lr}"
    experiment_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    command = [
        "python", "train.py",
        "--model", model,
        "--optimizer", optimizer,
        "--scheduler", scheduler,
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--early_stop", str(early_stop),
        "--save_dir", experiment_dir
    ]
    
    print(f"Running experiment: {experiment_name}")
    print(f"Command: {' '.join(command)}")
    
    start_time = time.time()
    process = subprocess.run(command, capture_output=True, text=True)
    end_time = time.time()
    
    duration = end_time - start_time
    
    # Check if the experiment was successful
    success = process.returncode == 0
    
    if not success:
        print(f"Experiment {experiment_name} failed!")
        print(f"Error: {process.stderr}")
        return {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "learning_rate": lr,
            "success": False,
            "duration_minutes": duration / 60
        }
    
    # Try to load experiment results
    results_file = os.path.join(experiment_dir, f"{model}_experiment_settings.json")
    performance_file = os.path.join(experiment_dir, f"{model}_results_summary.txt")
    
    result_data = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "learning_rate": lr,
        "success": True,
        "duration_minutes": duration / 60
    }
    
    # Load settings if available
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                settings = json.load(f)
                result_data["actual_epochs"] = settings.get("actual_epochs", 0)
                result_data["early_stopped"] = settings.get("early_stopped", False)
        except Exception as e:
            print(f"Error loading settings for {experiment_name}: {e}")
    
    # Extract metrics from the summary file
    if os.path.exists(performance_file):
        try:
            metrics = parse_results_summary(performance_file)
            result_data.update(metrics)
        except Exception as e:
            print(f"Error parsing results for {experiment_name}: {e}")
    
    print(f"Experiment {experiment_name} completed in {duration/60:.2f} minutes")
    return result_data

def parse_results_summary(file_path):
    """Extract key metrics from the results summary text file"""
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Extract metrics using simple parsing
            def extract_value(pattern, content):
                import re
                match = re.search(pattern, content)
                if match:
                    try:
                        return float(match.group(1))
                    except:
                        return match.group(1)
                return None
            
            metrics["val_loss"] = extract_value(r"Validation Loss \(MSE\): ([0-9.]+)", content)
            metrics["val_acc"] = extract_value(r"Validation Accuracy: ([0-9.]+)", content)
            metrics["auroc"] = extract_value(r"AUROC: ([0-9.]+)", content)
            metrics["precision"] = extract_value(r"Precision: ([0-9.]+)", content)
            metrics["recall"] = extract_value(r"Recall/Sensitivity: ([0-9.]+)", content)
            metrics["specificity"] = extract_value(r"Specificity: ([0-9.]+)", content)
            metrics["f1_score"] = extract_value(r"F1 Score: ([0-9.]+)", content)
            metrics["mcc"] = extract_value(r"MCC: ([0-9.]+)", content)
            
            # Early stopping status
            early_stopped = "Training completed with early stopping: Yes" in content
            metrics["early_stopped"] = early_stopped
    except Exception as e:
        print(f"Error parsing results summary: {e}")
    
    return metrics

def run_all_experiments(save_dir, epochs, early_stop, parallel=False):
    """Run all combinations of experiments"""
    # Experiment configurations
    models = ["GCNN", "AttGNN", "ResGIN"]
    optimizers = ["adam", "sgd", "adagrad", "radam"]
    schedulers = ["multistep", "plateau", "none"]
    learning_rates = [0.001, 0.0005, 0.0001]
    
    # Create directory for results
    os.makedirs(save_dir, exist_ok=True)
    
    results = []
    
    # Generate all experiment combinations
    experiments = []
    for model in models:
        for optimizer in optimizers:
            for scheduler in schedulers:
                for lr in learning_rates:
                    experiments.append((model, optimizer, scheduler, lr))
    
    print(f"Total experiments to run: {len(experiments)}")
    
    if parallel:
        # Use concurrent.futures for parallel execution
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:  # Adjust max_workers based on your system
            futures = []
            for model, optimizer, scheduler, lr in experiments:
                future = executor.submit(
                    run_experiment, model, optimizer, scheduler, lr, save_dir, epochs, early_stop
                )
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Experiment failed: {e}")
    else:
        # Run sequentially
        for model, optimizer, scheduler, lr in experiments:
            try:
                result = run_experiment(model, optimizer, scheduler, lr, save_dir, epochs, early_stop)
                results.append(result)
            except Exception as e:
                print(f"Experiment failed: {e}")
    
    # Create a DataFrame from results and save to CSV
    results_df = pd.DataFrame(results)
    
    # Sort by model type and then by performance metric (f1_score)
    results_df = results_df.sort_values(by=['model', 'f1_score'], ascending=[True, False])
    
    # Save to CSV
    csv_path = os.path.join(save_dir, 'experiment_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    # Also save a version with better formatting for important metrics
    formatted_df = results_df.copy()
    for col in ['val_acc', 'auroc', 'precision', 'recall', 'specificity', 'f1_score', 'mcc']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    
    formatted_csv_path = os.path.join(save_dir, 'experiment_results_formatted.csv')
    formatted_df.to_csv(formatted_csv_path, index=False)
    
    print(f"All experiments completed! Results saved to {csv_path}")
    
    # Create summary of best configurations per model
    best_per_model = results_df.loc[results_df.groupby('model')['f1_score'].idxmax()]
    best_csv_path = os.path.join(save_dir, 'best_configurations.csv')
    best_per_model.to_csv(best_csv_path, index=False)
    
    print(f"Best configuration for each model saved to {best_csv_path}")
    return results_df

if __name__ == "__main__":
    args = parse_arguments()
    run_all_experiments(args.save_dir, args.epochs, args.early_stop, args.parallel)