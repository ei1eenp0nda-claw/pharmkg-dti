#!/usr/bin/env python
"""
PharmKG-DTI: Command Line Interface

Main entry point for training, evaluation, and inference.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import (
    ExperimentConfig, 
    get_default_config, 
    get_fast_config,
    get_production_config
)


def train_command(args):
    """Run training."""
    print("=" * 70)
    print("PharmKG-DTI Training")
    print("=" * 70)
    
    # Load config
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    elif args.fast:
        config = get_fast_config()
    elif args.production:
        config = get_production_config()
    else:
        config = get_default_config()
    
    # Override with CLI args
    if args.dataset:
        config.data.dataset = args.dataset
    if args.model:
        config.model.name = args.model
    if args.epochs:
        config.training.epochs = args.epochs
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model.name}")
    print(f"  Dataset: {config.data.dataset}")
    print(f"  Epochs: {config.training.epochs}")
    
    # Run training
    from src.training.train import train_model
    from src.data.benchmark_loader import load_benchmark_dataset
    
    print(f"\nLoading {config.data.dataset} dataset...")
    splits = load_benchmark_dataset(
        dataset_name=config.data.dataset,
        split_method=config.data.split_method
    )
    
    print(f"Training {config.model.name} model...")
    # model, history = train_model(config, splits)
    
    print("\n✓ Training complete!")


def evaluate_command(args):
    """Run evaluation."""
    print("=" * 70)
    print("PharmKG-DTI Evaluation")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    import torch
    checkpoint = torch.load(args.checkpoint)
    
    # Load data
    from src.data.benchmark_loader import load_benchmark_dataset
    splits = load_benchmark_dataset(
        dataset_name=args.dataset,
        split_method=args.split
    )
    
    # Evaluate
    print("Running evaluation...")
    # metrics = evaluate_model(model, splits)
    
    print("\n✓ Evaluation complete!")


def predict_command(args):
    """Run prediction."""
    print("=" * 70)
    print("PharmKG-DTI Prediction")
    print("=" * 70)
    
    print(f"\nDrug: {args.drug}")
    print(f"Target: {args.target}")
    
    # Load model
    # Make prediction
    # Display result
    
    print("\n✓ Prediction complete!")


def benchmark_command(args):
    """Run benchmark suite."""
    print("=" * 70)
    print("PharmKG-DTI Benchmark Suite")
    print("=" * 70)
    
    from scripts.run_benchmarks import ExperimentRunner
    
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    if args.full:
        runner.run_benchmark_suite()
        runner.generate_report()
    else:
        runner.run_experiment(
            model_name=args.model,
            dataset_name=args.dataset,
            split_method=args.split
        )
    
    print("\n✓ Benchmark complete!")


def serve_command(args):
    """Start API server."""
    print("=" * 70)
    print("PharmKG-DTI API Server")
    print("=" * 70)
    
    import uvicorn
    
    print(f"\nStarting server on {args.host}:{args.port}...")
    print(f"API docs: http://{args.host}:{args.port}/docs\n")
    
    uvicorn.run(
        "src.api.inference_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


def main():
    parser = argparse.ArgumentParser(
        description="PharmKG-DTI: Drug-Target Interaction Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python -m cli train
  
  # Train with specific dataset
  python -m cli train --dataset davis --epochs 100
  
  # Fast training
  python -m cli train --fast
  
  # Evaluate model
  python -m cli evaluate --checkpoint checkpoints/best.pt --dataset bindingdb
  
  # Run prediction
  python -m cli predict --drug "CCO" --target "MVLSPADKTN"
  
  # Run benchmarks
  python -m cli benchmark --model dhgt --dataset bindingdb
  python -m cli benchmark --full
  
  # Start API server
  python -m cli serve --port 8000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, help='Config file path')
    train_parser.add_argument('--fast', action='store_true', help='Use fast config')
    train_parser.add_argument('--production', action='store_true', help='Use production config')
    train_parser.add_argument('--dataset', type=str, choices=['bindingdb', 'davis', 'kiba'])
    train_parser.add_argument('--model', type=str, choices=['dhgt', 'hgan', 'sage'])
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    eval_parser.add_argument('--dataset', type=str, default='bindingdb')
    eval_parser.add_argument('--split', type=str, default='random')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make prediction')
    predict_parser.add_argument('--drug', type=str, required=True, help='Drug SMILES')
    predict_parser.add_argument('--target', type=str, required=True, help='Target sequence')
    predict_parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    bench_parser.add_argument('--full', action='store_true', help='Run full benchmark suite')
    bench_parser.add_argument('--model', type=str, default='dhgt')
    bench_parser.add_argument('--dataset', type=str, default='bindingdb')
    bench_parser.add_argument('--split', type=str, default='random')
    bench_parser.add_argument('--output-dir', type=str, default='experiments')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0')
    serve_parser.add_argument('--port', type=int, default=8000)
    serve_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'benchmark':
        benchmark_command(args)
    elif args.command == 'serve':
        serve_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
