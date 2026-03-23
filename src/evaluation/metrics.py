"""
PharmKG-DTI: Evaluation Metrics

Implements standard metrics for DTI prediction:
- AUC, AUPR
- Hits@K
- MRR (Mean Reciprocal Rank)
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report
)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_list: List[int] = [1, 10, 50]
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for DTI prediction.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted scores/probabilities
        k_list: List of k values for Hits@K metric
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
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
        hits_k = np.mean(y_true_sorted[:k]) if k <= len(y_true) else 0.0
        metrics[f'hits@{k}'] = hits_k
    
    # MRR (Mean Reciprocal Rank)
    positive_ranks = np.where(y_true_sorted == 1)[0]
    if len(positive_ranks) > 0:
        metrics['mrr'] = np.mean(1.0 / (positive_ranks + 1))
    else:
        metrics['mrr'] = 0.0
    
    # Binary predictions for accuracy-based metrics
    y_pred = (y_score >= 0.5).astype(int)
    metrics['accuracy'] = np.mean(y_pred == y_true)
    
    # F1 score
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    return metrics


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Dict[str, float]:
    """
    Compute ranking-specific metrics.
    
    Args:
        y_true: Ground truth labels (1 for positive, 0 for negative)
        y_score: Predicted scores
    
    Returns:
        Dictionary of ranking metrics
    """
    # Sort by score (descending)
    ranked_indices = np.argsort(-y_score)
    y_true_sorted = y_true[ranked_indices]
    
    # Find ranks of all positives
    positive_ranks = np.where(y_true_sorted == 1)[0] + 1  # 1-indexed
    
    metrics = {}
    
    # Mean Rank
    metrics['mean_rank'] = np.mean(positive_ranks)
    
    # Median Rank
    metrics['median_rank'] = np.median(positive_ranks)
    
    # Mean Reciprocal Rank
    metrics['mrr'] = np.mean(1.0 / positive_ranks)
    
    # Precision at different K
    for k in [1, 5, 10, 20, 50, 100]:
        if k <= len(y_true):
            metrics[f'p@{k}'] = np.sum(y_true_sorted[:k]) / k
    
    # nDCG
    dcg = np.sum(y_true_sorted / np.log2(np.arange(2, len(y_true_sorted) + 2)))
    ideal_dcg = np.sum(
        np.ones(np.sum(y_true)) / 
        np.log2(np.arange(2, np.sum(y_true) + 2))
    )
    metrics['ndcg'] = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    return metrics


def evaluate_link_prediction(
    model,
    graph_data,
    test_edges: 'torch.Tensor',
    test_labels: 'torch.Tensor',
    num_negatives: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate link prediction with negative sampling.
    
    For each positive test edge, generate k negative edges and compute ranking metrics.
    
    Args:
        model: Trained model
        graph_data: Graph data
        test_edges: Test edges [2, N]
        test_labels: Test labels [N]
        num_negatives: Number of negative samples per positive
        device: Device to run on
    
    Returns:
        Dictionary of metrics
    """
    import torch
    
    model.eval()
    
    with torch.no_grad():
        pos_mask = test_labels == 1
        pos_edges = test_edges[:, pos_mask]
        
        all_ranks = []
        
        for i in range(pos_edges.size(1)):
            pos_edge = pos_edges[:, i:i+1]
            
            # Generate negative samples
            neg_edges = []
            for _ in range(num_negatives):
                neg_drug = torch.randint(0, graph_data['drug'].x.size(0), (1,))
                neg_prot = torch.randint(0, graph_data['protein'].x.size(0), (1,))
                neg_edges.append([neg_drug.item(), neg_prot.item()])
            
            neg_edges = torch.tensor(neg_edges, dtype=torch.long, device=device).t()
            
            # Combine
            batch_edges = torch.cat([pos_edge, neg_edges], dim=1)
            
            # Score
            scores = model(graph_data, batch_edges)
            scores = torch.sigmoid(scores)
            
            # Rank
            rank = (scores[1:] > scores[0]).sum().item() + 1
            all_ranks.append(rank)
        
        all_ranks = np.array(all_ranks)
        
        metrics = {
            'mrr': np.mean(1.0 / all_ranks),
            'hits@1': np.mean(all_ranks <= 1),
            'hits@10': np.mean(all_ranks <= 10),
            'hits@50': np.mean(all_ranks <= 50),
            'mean_rank': np.mean(all_ranks),
            'median_rank': np.median(all_ranks)
        }
        
        return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    # Group metrics
    basic_metrics = ['auc', 'aupr', 'accuracy', 'precision', 'recall', 'f1']
    ranking_metrics = ['mrr', 'mean_rank', 'median_rank']
    hits_metrics = [k for k in metrics.keys() if 'hits@' in k]
    
    print("\nBasic Metrics:")
    for k in basic_metrics:
        if k in metrics:
            print(f"  {k.upper():12s}: {metrics[k]:.4f}")
    
    print("\nRanking Metrics:")
    for k in ranking_metrics:
        if k in metrics:
            print(f"  {k.upper():12s}: {metrics[k]:.4f}")
    
    print("\nHits@K Metrics:")
    for k in sorted(hits_metrics, key=lambda x: int(x.split('@')[1])):
        print(f"  {k.upper():12s}: {metrics[k]:.4f}")
    
    print(f"{'='*50}\n")
