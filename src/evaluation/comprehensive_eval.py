"""
PharmKG-DTI: Enhanced Evaluation with Additional Metrics and Visualization

Adds:
- F1-Score computation
- ROC and PR curve plotting
- Confusion matrix visualization
- Comprehensive reporting
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    matthews_corrcoef
)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    k_list: List[int] = [1, 10, 50, 100]
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for DTI prediction.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_score: Predicted scores/probabilities
        threshold: Classification threshold
        k_list: List of k values for Hits@K metric
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Binary predictions
    y_pred = (y_score >= threshold).astype(int)
    
    # Basic classification metrics
    metrics['threshold'] = threshold
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Ranking metrics (AUC, AUPR)
    try:
        metrics['auc'] = roc_auc_score(y_true, y_score)
    except:
        metrics['auc'] = 0.0
    
    try:
        metrics['aupr'] = average_precision_score(y_true, y_score)
    except:
        metrics['aupr'] = 0.0
    
    # Hits@K metrics
    ranked_indices = np.argsort(-y_score)
    y_true_sorted = y_true[ranked_indices]
    
    for k in k_list:
        if k <= len(y_true):
            hits_k = np.sum(y_true_sorted[:k]) / k
            metrics[f'hits@{k}'] = hits_k
        else:
            metrics[f'hits@{k}'] = 0.0
    
    # MRR (Mean Reciprocal Rank)
    positive_ranks = np.where(y_true_sorted == 1)[0]
    if len(positive_ranks) > 0:
        metrics['mrr'] = np.mean(1.0 / (positive_ranks + 1))
        metrics['median_rank'] = np.median(positive_ranks + 1)
        metrics['mean_rank'] = np.mean(positive_ranks + 1)
    else:
        metrics['mrr'] = 0.0
        metrics['median_rank'] = float('inf')
        metrics['mean_rank'] = float('inf')
    
    return metrics


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ROC Curve"
) -> plt.Figure:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curve"
) -> plt.Figure:
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AUPR = {aupr:.4f})')
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Baseline ({baseline:.4f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """Plot confusion matrix."""
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Predicted 0', 'Predicted 1'],
        yticklabels=['Actual 0', 'Actual 1'],
        ax=ax
    )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metrics_comparison(
    results_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of metrics across different models/datasets.
    
    Args:
        results_dict: Dict of {name: metrics_dict}
        metrics_to_plot: List of metric names to plot
        save_path: Path to save figure
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['auc', 'aupr', 'f1', 'precision', 'recall']
    
    # Prepare data
    models = list(results_dict.keys())
    data = {metric: [results_dict[m].get(metric, 0) for m in models] for metric in metrics_to_plot}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.15
    
    for i, metric in enumerate(metrics_to_plot):
        offset = width * (i - len(metrics_to_plot) / 2)
        ax.bar(x + offset, data[metric], width, label=metric.upper())
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_dir: Optional[str] = None,
    experiment_name: str = "experiment"
) -> str:
    """
    Generate comprehensive evaluation report with plots.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted scores
        save_dir: Directory to save report and plots
        experiment_name: Name of the experiment
    
    Returns:
        Path to report directory
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics
    metrics = compute_comprehensive_metrics(y_true, y_score)
    
    # Generate text report
    report_lines = [
        f"{'='*60}",
        f"PharmKG-DTI Evaluation Report",
        f"Experiment: {experiment_name}",
        f"{'='*60}",
        "",
        "Classification Metrics:",
        f"  Accuracy:    {metrics['accuracy']:.4f}",
        f"  Precision:   {metrics['precision']:.4f}",
        f"  Recall:      {metrics['recall']:.4f}",
        f"  F1-Score:    {metrics['f1']:.4f}",
        f"  MCC:         {metrics['mcc']:.4f}",
        "",
        "Ranking Metrics:",
        f"  AUC:         {metrics['auc']:.4f}",
        f"  AUPR:        {metrics['aupr']:.4f}",
        f"  MRR:         {metrics['mrr']:.4f}",
        "",
        "Hits@K Metrics:",
    ]
    
    for k in [1, 10, 50, 100]:
        if f'hits@{k}' in metrics:
            report_lines.append(f"  Hits@{k:3d}:     {metrics[f'hits@{k}']:.4f}")
    
    report_lines.extend([
        "",
        f"{'='*60}",
    ])
    
    report_text = '\n'.join(report_lines)
    print(report_text)
    
    if save_dir:
        # Save report
        with open(save_dir / f'{experiment_name}_report.txt', 'w') as f:
            f.write(report_text)
        
        # Save metrics as JSON
        import json
        with open(save_dir / f'{experiment_name}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate plots
        plot_roc_curve(y_true, y_score, save_path=save_dir / f'{experiment_name}_roc.png')
        plot_pr_curve(y_true, y_score, save_path=save_dir / f'{experiment_name}_pr.png')
        plot_confusion_matrix(y_true, y_score, save_path=save_dir / f'{experiment_name}_cm.png')
        
        print(f"\nReport saved to: {save_dir}")
    
    return str(save_dir) if save_dir else ""


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted scores
        metric: Metric to optimize ('f1', 'precision', 'recall', 'mcc')
    
    Returns:
        (optimal_threshold, best_score)
    """
    thresholds = np.linspace(0, 1, 101)
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    scores = np.array(scores)
    best_idx = np.argmax(scores)
    
    return thresholds[best_idx], scores[best_idx]
