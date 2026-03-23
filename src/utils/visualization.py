"""
PharmKG-DTI: Visualization Module

Training curves, attention visualization, and result plotting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class TrainingVisualizer:
    """Visualize training progress and metrics."""
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_training_curves(
        self,
        history: Dict[str, List],
        save_name: str = "training_curves.png"
    ):
        """
        Plot training and validation loss/accuracy curves.
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', 'train_auc', 'val_auc'
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metric curves
        if 'train_auc' in history:
            axes[1].plot(history['train_auc'], label='Train AUC', linewidth=2)
        if 'val_auc' in history:
            axes[1].plot(history['val_auc'], label='Val AUC', linewidth=2)
        if 'val_aupr' in history:
            axes[1].plot(history['val_aupr'], label='Val AUPR', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric')
        axes[1].set_title('Training Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved to {save_path}")
    
    def plot_roc_curves(
        self,
        roc_data: Dict[str, Tuple],
        save_name: str = "roc_curves.png"
    ):
        """
        Plot ROC curves for multiple models.
        
        Args:
            roc_data: Dict of {model_name: (fpr, tpr, auc)}
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, (fpr, tpr, auc) in roc_data.items():
            ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curves saved to {save_path}")
    
    def plot_pr_curves(
        self,
        pr_data: Dict[str, Tuple],
        save_name: str = "pr_curves.png"
    ):
        """
        Plot Precision-Recall curves.
        
        Args:
            pr_data: Dict of {model_name: (precision, recall, aupr)}
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, (precision, recall, aupr) in pr_data.items():
            ax.plot(recall, precision, linewidth=2, label=f'{model_name} (AUPR = {aupr:.3f})')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str] = None,
        save_name: str = "confusion_matrix.png"
    ):
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        class_names = class_names or ['Non-interaction', 'Interaction']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14)
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        save_name: str = "model_comparison.png"
    ):
        """
        Bar plot comparing multiple models across metrics.
        
        Args:
            results: Dict of {model_name: {metric: value}}
            metrics: List of metrics to plot
            save_name: Filename to save plot
        """
        metrics = metrics or ['auc', 'aupr', 'f1']
        
        models = list(results.keys())
        n_models = len(models)
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in models]
            
            bars = axes[idx].bar(models, values, color=sns.color_palette("husl", n_models))
            axes[idx].set_ylabel(metric.upper(), fontsize=12)
            axes[idx].set_title(f'{metric.upper()} Comparison', fontsize=14)
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Model comparison saved to {save_path}")


class AttentionVisualizer:
    """Visualize attention weights from GNN models."""
    
    def __init__(self, save_dir: str = "visualizations/attention"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        x_labels: List[str] = None,
        y_labels: List[str] = None,
        title: str = "Attention Weights",
        save_name: str = "attention_heatmap.png"
    ):
        """
        Plot attention weights as heatmap.
        
        Args:
            attention_weights: Attention weight matrix
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis
            title: Plot title
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(attention_weights, cmap='viridis', ax=ax,
                   xticklabels=x_labels, yticklabels=y_labels,
                   cbar_kws={'label': 'Attention Weight'})
        
        ax.set_title(title, fontsize=14)
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Attention heatmap saved to {save_path}")
    
    def plot_binding_site_highlight(
        self,
        protein_sequence: str,
        binding_scores: np.ndarray,
        save_name: str = "binding_sites.png"
    ):
        """
        Visualize predicted binding sites on protein sequence.
        
        Args:
            protein_sequence: Amino acid sequence
            binding_scores: Binding probability for each residue
            save_name: Filename to save plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), 
                                        gridspec_kw={'height_ratios': [1, 3]})
        
        # Sequence view (top)
        residues = list(protein_sequence)
        positions = range(len(residues))
        
        # Color map based on binding scores
        colors = plt.cm.Reds(binding_scores[:len(residues)])
        
        ax1.bar(positions, [1]*len(residues), color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_xlim([-0.5, len(residues)-0.5])
        ax1.set_yticks([])
        ax1.set_xlabel('Residue Position')
        ax1.set_title('Predicted Binding Sites (Red = High Probability)')
        
        # Add residue letters
        for i, res in enumerate(residues):
            ax1.text(i, 0.5, res, ha='center', va='center', fontsize=8)
        
        # Score plot (bottom)
        ax2.plot(positions, binding_scores[:len(residues)], linewidth=2, color='red')
        ax2.fill_between(positions, binding_scores[:len(residues)], alpha=0.3, color='red')
        ax2.set_xlabel('Residue Position', fontsize=12)
        ax2.set_ylabel('Binding Probability', fontsize=12)
        ax2.set_title('Binding Site Confidence Scores', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Binding site visualization saved to {save_path}")


class ResultsDashboard:
    """Generate comprehensive results dashboard."""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.visualizer = TrainingVisualizer(self.experiment_dir / "visualizations")
    
    def generate_from_experiment(self, result_json: str):
        """Generate all visualizations from experiment results."""
        with open(result_json) as f:
            results = json.load(f)
        
        # Training curves
        if 'history' in results:
            self.visualizer.plot_training_curves(results['history'])
        
        # Model comparison
        # ROC/PR curves
        # Confusion matrix
        
        print(f"✓ Dashboard generated in {self.visualizer.save_dir}")


if __name__ == '__main__':
    # Test visualization
    print("Testing visualization module...")
    
    viz = TrainingVisualizer()
    
    # Test training curves
    history = {
        'train_loss': [0.5, 0.4, 0.35, 0.3, 0.25],
        'val_loss': [0.55, 0.45, 0.4, 0.38, 0.35],
        'train_auc': [0.7, 0.8, 0.85, 0.88, 0.9],
        'val_auc': [0.68, 0.78, 0.82, 0.85, 0.87]
    }
    viz.plot_training_curves(history)
    
    # Test model comparison
    results = {
        'DHGT-DTI': {'auc': 0.973, 'aupr': 0.662, 'f1': 0.89},
        'HGAN': {'auc': 0.960, 'aupr': 0.710, 'f1': 0.87},
        'SAGE': {'auc': 0.920, 'aupr': 0.650, 'f1': 0.82}
    }
    viz.plot_model_comparison(results)
    
    print("\n✓ Visualization tests passed!")
