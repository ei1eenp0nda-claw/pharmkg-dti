"""
PharmKG-DTI: Training Script

Implements training loop with:
- Early stopping
- Learning rate scheduling
- WandB/TensorBoard logging
- Checkpoint saving
"""

import os
import random
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import yaml

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Try importing tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

from src.data.dataset import PharmKGDataset
from src.models.gnn_models import create_model
from src.evaluation.metrics import compute_metrics


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def negative_sampling(
    num_nodes_src: int,
    num_nodes_dst: int,
    num_samples: int,
    existing_edges: torch.Tensor,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample negative edges that don't exist in the graph.
    
    Args:
        num_nodes_src: Number of source nodes
        num_nodes_dst: Number of destination nodes
        num_samples: Number of negative samples to generate
        existing_edges: Existing positive edges [2, N]
        device: Device to create tensor on
    
    Returns:
        Negative edge indices [2, num_samples]
    """
    existing_set = set()
    if existing_edges.numel() > 0:
        existing_set = set(
            (int(existing_edges[0, i]), int(existing_edges[1, i]))
            for i in range(existing_edges.size(1))
        )
    
    negative_edges = []
    max_attempts = num_samples * 10
    attempts = 0
    
    while len(negative_edges) < num_samples and attempts < max_attempts:
        src = random.randint(0, num_nodes_src - 1)
        dst = random.randint(0, num_nodes_dst - 1)
        
        if (src, dst) not in existing_set:
            negative_edges.append([src, dst])
        
        attempts += 1
    
    # If we couldn't sample enough, fill with random
    while len(negative_edges) < num_samples:
        negative_edges.append([
            random.randint(0, num_nodes_src - 1),
            random.randint(0, num_nodes_dst - 1)
        ])
    
    return torch.tensor(negative_edges, dtype=torch.long, device=device).t()


class Trainer:
    """Trainer class for DTI prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: torch.Tensor,
        val_labels: torch.Tensor,
        graph_data: 'HeteroData',
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_edges = train_data.to(device)
        self.train_labels = train_labels.to(device)
        self.val_edges = val_data.to(device)
        self.val_labels = val_labels.to(device)
        self.graph_data = graph_data.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['optimizer']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config['training']['scheduler']['factor'],
            patience=config['training']['scheduler']['patience'],
            verbose=True
        )
        
        # Loss function
        pos_weight = torch.tensor([config['training']['loss'].get('pos_weight', 1.0)])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        
        # Logging
        self.use_wandb = config['logging'].get('use_wandb', False) and WANDB_AVAILABLE
        self.use_tensorboard = config['logging'].get('use_tensorboard', False) and TB_AVAILABLE
        
        if self.use_wandb:
            wandb.init(
                project=config['logging']['wandb'].get('project', 'pharmkg-dti'),
                entity=config['logging']['wandb'].get('entity'),
                config=config
            )
        
        if self.use_tensorboard:
            log_dir = Path('runs') / config.get('experiment_name', 'default')
            self.tb_writer = SummaryWriter(log_dir=log_dir)
        
        # Checkpointing
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.early_stop_patience = config['training']['early_stopping_patience']
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Create positive and negative training samples
        pos_mask = self.train_labels == 1
        neg_mask = self.train_labels == 0
        
        pos_edges = self.train_edges[:, pos_mask]
        neg_edges = self.train_edges[:, neg_mask]
        
        # If no pre-sampled negatives, generate them
        if neg_edges.size(1) == 0:
            neg_edges = negative_sampling(
                num_nodes_src=self.graph_data['drug'].x.size(0),
                num_nodes_dst=self.graph_data['protein'].x.size(0),
                num_samples=pos_edges.size(1),
                existing_edges=pos_edges,
                device=self.device
            )
        
        # Combine positive and negative edges
        all_edges = torch.cat([pos_edges, neg_edges], dim=1)
        all_labels = torch.cat([
            torch.ones(pos_edges.size(1)),
            torch.zeros(neg_edges.size(1))
        ]).to(self.device)
        
        # Shuffle
        perm = torch.randperm(all_edges.size(1))
        all_edges = all_edges[:, perm]
        all_labels = all_labels[perm]
        
        # Mini-batch training
        batch_size = self.config['training']['batch_size']
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, all_edges.size(1), batch_size):
            batch_edges = all_edges[:, i:i+batch_size]
            batch_labels = all_labels[i:i+batch_size]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(self.graph_data, batch_edges)
            loss = self.criterion(logits, batch_labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {'train_loss': avg_loss}
    
    @torch.no_grad()
    def evaluate(self, edges: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate model on given edges."""
        self.model.eval()
        
        batch_size = self.config['evaluation']['link_prediction']['eval_batch_size']
        all_scores = []
        
        for i in range(0, edges.size(1), batch_size):
            batch_edges = edges[:, i:i+batch_size]
            logits = self.model(self.graph_data, batch_edges)
            scores = torch.sigmoid(logits)
            all_scores.append(scores.cpu())
        
        all_scores = torch.cat(all_scores)
        labels_cpu = labels.cpu()
        
        # Compute metrics
        metrics = compute_metrics(labels_cpu.numpy(), all_scores.numpy())
        
        return metrics
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training edges: {self.train_edges.size(1)}")
        print(f"Validation edges: {self.val_edges.size(1)}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.evaluate(self.val_edges, self.val_labels)
            
            # Combine metrics
            metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
            
            # Log
            if self.use_wandb:
                wandb.log(metrics, step=epoch)
            
            if self.use_tensorboard:
                for k, v in metrics.items():
                    self.tb_writer.add_scalar(k, v, epoch)
            
            # Print progress
            if epoch % self.config['logging'].get('log_interval', 10) == 0:
                print(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Loss: {metrics['train_loss']:.4f} | "
                    f"Val AUC: {metrics['val_auc']:.4f} | "
                    f"Val AUPR: {metrics['val_aupr']:.4f}"
                )
            
            # Learning rate scheduling
            self.scheduler.step(metrics['val_auc'])
            
            # Checkpointing
            if metrics['val_auc'] > self.best_val_auc:
                self.best_val_auc = metrics['val_auc']
                self.patience_counter = 0
                
                # Save best model
                checkpoint_path = self.checkpoint_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_auc': self.best_val_auc,
                    'config': self.config
                }, checkpoint_path)
                print(f"Saved best model with Val AUC: {self.best_val_auc:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Close loggers
        if self.use_wandb:
            wandb.finish()
        if self.use_tensorboard:
            self.tb_writer.close()
        
        print(f"Training complete. Best Val AUC: {self.best_val_auc:.4f}")
        return self.best_val_auc


def main():
    parser = argparse.ArgumentParser(description='Train PharmKG-DTI model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, default='drugbank',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='dhgt',
                        choices=['dhgt', 'hgan', 'sage'],
                        help='Model architecture')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = PharmKGDataset(
        root=config['data']['processed_dir'],
        dataset_name=args.dataset,
        use_structure=config['model']['feature_fusion']['use_drug_structure'],
        use_sequence=config['model']['feature_fusion']['use_protein_sequence']
    )
    
    print(dataset)
    
    # Get train/val/test split
    train_edges, train_labels, val_edges, val_labels, test_edges, test_labels = \
        dataset.get_train_val_test_split()
    
    # Get heterogeneous graph data
    graph_data = dataset.to_pyg_hetero_data()
    
    # Get feature dimensions
    num_drug_features = graph_data['drug'].x.size(1)
    num_protein_features = graph_data['protein'].x.size(1)
    
    # Create model
    print(f"Creating model: {args.model}")
    model = create_model(
        model_name=args.model,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        num_drug_features=num_drug_features,
        num_protein_features=num_protein_features,
        dropout=config['model']['dropout'],
        use_residual=config['model']['graph_transformer']['use_residual']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_data=train_edges,
        train_labels=train_labels,
        val_data=val_edges,
        val_labels=val_labels,
        graph_data=graph_data,
        config=config,
        device=device
    )
    
    # Train
    best_auc = trainer.train(config['training']['num_epochs'])
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_edges, test_labels)
    print("Test Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
