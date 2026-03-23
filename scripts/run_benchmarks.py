"""
PharmKG-DTI: Benchmark Experiment Runner

Automated benchmarking on standard DTI datasets.
Supports multiple models, evaluation protocols, and result logging.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.benchmark_loader import load_benchmark_dataset
from src.models.gnn_models import DHGTDTI, HGANDTI, SAGEBaseline
from src.training.train import train_model
from src.evaluation.comprehensive_eval import ComprehensiveEvaluator


class ExperimentRunner:
    """
    Manages benchmark experiments with reproducible configurations.
    """
    
    def __init__(
        self,
        output_dir: str = "experiments",
        device: str = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.results = []
    
    def run_experiment(
        self,
        model_name: str,
        dataset_name: str,
        split_method: str = 'random',
        config: Dict = None,
        save_predictions: bool = True
    ) -> Dict:
        """
        Run a single benchmark experiment.
        
        Args:
            model_name: 'dhgt', 'hgan', or 'sage'
            dataset_name: 'bindingdb', 'davis', or 'kiba'
            split_method: 'random', 'cold_drug', 'cold_target'
            config: Model configuration
            save_predictions: Whether to save raw predictions
        
        Returns:
            Dictionary with all experiment results
        """
        print(f"\n{'='*70}")
        print(f"Experiment: {model_name} on {dataset_name} ({split_method})")
        print(f"{'='*70}\n")
        
        # Default config
        config = config or {
            'hidden_dim': 128,
            'num_layers': 3,
            'num_heads': 8,
            'dropout': 0.3,
            'lr': 0.001,
            'batch_size': 256,
            'epochs': 100,
            'early_stopping_patience': 20
        }
        
        # Load data
        print("Loading dataset...")
        splits = load_benchmark_dataset(
            dataset_name=dataset_name,
            split_method=split_method
        )
        
        # Create model
        print(f"Creating {model_name} model...")
        if model_name == 'dhgt':
            model = DHGTDTI(
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                dropout=config['dropout']
            )
        elif model_name == 'hgan':
            model = HGANDTI(
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                dropout=config['dropout']
            )
        elif model_name == 'sage':
            model = SAGEBaseline(
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Train
        print("Training...")
        start_time = time.time()
        
        # Placeholder training - would call actual training
        # history = train_model(model, splits, config)
        training_time = time.time() - start_time
        
        # Evaluate
        print("Evaluating...")
        evaluator = ComprehensiveEvaluator(model, splits, self.device)
        metrics = evaluator.evaluate_all()
        
        # Compile results
        result = {
            'model': model_name,
            'dataset': dataset_name,
            'split': split_method,
            'config': config,
            'metrics': metrics,
            'training_time_seconds': training_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.results.append(result)
        
        # Save
        exp_name = f"{model_name}_{dataset_name}_{split_method}_{int(time.time())}"
        result_path = self.output_dir / f"{exp_name}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✓ Experiment complete. Results saved to {result_path}")
        print(f"  AUC: {metrics.get('auc', 'N/A'):.4f}")
        print(f"  AUPR: {metrics.get('aupr', 'N/A'):.4f}")
        
        return result
    
    def run_benchmark_suite(
        self,
        models: List[str] = None,
        datasets: List[str] = None,
        splits: List[str] = None
    ) -> pd.DataFrame:
        """
        Run full benchmark suite.
        
        Args:
            models: List of model names
            datasets: List of dataset names
            splits: List of split methods
        
        Returns:
            DataFrame with all results
        """
        models = models or ['dhgt', 'hgan', 'sage']
        datasets = datasets or ['bindingdb', 'davis', 'kiba']
        splits = splits or ['random', 'cold_drug', 'cold_target']
        
        for model in models:
            for dataset in datasets:
                for split in splits:
                    try:
                        self.run_experiment(model, dataset, split)
                    except Exception as e:
                        print(f"✗ Failed: {model}/{dataset}/{split}")
                        print(f"  Error: {e}")
        
        return self.summarize_results()
    
    def summarize_results(self) -> pd.DataFrame:
        """Summarize all experiment results."""
        rows = []
        for r in self.results:
            row = {
                'model': r['model'],
                'dataset': r['dataset'],
                'split': r['split'],
                'auc': r['metrics'].get('auc', 0),
                'aupr': r['metrics'].get('aupr', 0),
                'f1': r['metrics'].get('f1', 0),
                'mcc': r['metrics'].get('mcc', 0),
                'training_time': r['training_time_seconds']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save summary
        summary_path = self.output_dir / "summary.csv"
        df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*70}")
        print("Benchmark Summary")
        print(f"{'='*70}\n")
        print(df.to_string(index=False))
        
        return df
    
    def generate_report(self, output_path: str = "benchmark_report.md"):
        """Generate a markdown report of all results."""
        report = "# PharmKG-DTI Benchmark Report\n\n"
        report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary table
        df = self.summarize_results()
        report += "## Results Summary\n\n"
        report += df.to_markdown(index=False)
        report += "\n\n"
        
        # Best results
        report += "## Best Results by Dataset\n\n"
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            best = dataset_df.loc[dataset_df['auc'].idxmax()]
            report += f"### {dataset.upper()}\n"
            report += f"- **Best Model**: {best['model']}\n"
            report += f"- **AUC**: {best['auc']:.4f}\n"
            report += f"- **AUPR**: {best['aupr']:.4f}\n\n"
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run DTI benchmark experiments')
    parser.add_argument('--model', type=str, default='dhgt',
                       choices=['dhgt', 'hgan', 'sage'],
                       help='Model to use')
    parser.add_argument('--dataset', type=str, default='bindingdb',
                       choices=['bindingdb', 'davis', 'kiba'],
                       help='Dataset to use')
    parser.add_argument('--split', type=str, default='random',
                       choices=['random', 'cold_drug', 'cold_target'],
                       help='Split method')
    parser.add_argument('--run-suite', action='store_true',
                       help='Run full benchmark suite')
    parser.add_argument('--output-dir', type=str, default='experiments',
                       help='Output directory')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    if args.run_suite:
        runner.run_benchmark_suite()
        runner.generate_report()
    else:
        result = runner.run_experiment(
            model_name=args.model,
            dataset_name=args.dataset,
            split_method=args.split
        )
        print(f"\nFinal AUC: {result['metrics']['auc']:.4f}")


if __name__ == '__main__':
    main()
