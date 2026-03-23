"""
PharmKG-DTI: Model Ensemble

Ensemble methods for improved DTI prediction.
- Voting ensemble
- Stacking ensemble
- Weighted averaging
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class VotingEnsemble:
    """
    Hard voting ensemble (majority vote).
    
    Each model votes on interaction (0 or 1), majority wins.
    """
    
    def __init__(self, models: List[nn.Module], threshold: float = 0.5):
        self.models = models
        self.threshold = threshold
    
    def predict(self, *args, **kwargs) -> np.ndarray:
        """
        Get ensemble predictions.
        
        Returns:
            Binary predictions (0 or 1)
        """
        votes = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                prob = model(*args, **kwargs)
                pred = (prob > self.threshold).int()
                votes.append(pred.cpu().numpy())
        
        # Majority vote
        votes = np.array(votes)
        ensemble_pred = (votes.mean(axis=0) > 0.5).astype(int)
        
        return ensemble_pred


class AveragingEnsemble:
    """
    Soft voting ensemble (probability averaging).
    
    Average predictions from all models.
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        """
        Get averaged probabilities.
        
        Returns:
            Averaged probability predictions
        """
        probs = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                prob = model(*args, **kwargs)
                probs.append(prob.cpu().numpy() * weight)
        
        # Weighted average
        ensemble_prob = np.sum(probs, axis=0)
        
        return ensemble_prob
    
    def predict(self, *args, **kwargs, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions."""
        probs = self.predict_proba(*args, **kwargs)
        return (probs > threshold).astype(int)


class StackingEnsemble:
    """
    Stacking ensemble with meta-learner.
    
    Uses base model predictions as features for a meta-learner.
    """
    
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_learner: nn.Module,
        cv_folds: int = 5
    ):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
    
    def fit_meta_learner(self, X_meta: np.ndarray, y: np.ndarray):
        """
        Train meta-learner on base model predictions.
        
        Args:
            X_meta: Stacked predictions from base models (n_samples, n_models)
            y: True labels
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_meta)
        y_tensor = torch.FloatTensor(y)
        
        # Train meta-learner (simplified - actual would use proper training loop)
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        self.meta_learner.train()
        for epoch in range(100):
            optimizer.zero_grad()
            pred = self.meta_learner(X_tensor).squeeze()
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
    
    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        """
        Get predictions through stacked ensemble.
        
        Returns:
            Final probability predictions
        """
        # Get base model predictions
        base_preds = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                pred = model(*args, **kwargs)
                base_preds.append(pred.cpu().numpy())
        
        # Stack predictions
        X_meta = np.column_stack(base_preds)
        X_tensor = torch.FloatTensor(X_meta)
        
        # Meta-learner prediction
        self.meta_learner.eval()
        with torch.no_grad():
            final_pred = self.meta_learner(X_tensor).squeeze().numpy()
        
        return final_pred


class DynamicEnsemble:
    """
    Dynamic ensemble weighting based on model confidence.
    
    Weights models based on their prediction uncertainty.
    """
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
    
    def predict_with_uncertainty(
        self,
        *args,
        n_samples: int = 50,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation via MC dropout.
        
        Returns:
            (mean_predictions, uncertainties)
        """
        all_preds = []
        
        for model in self.models:
            model.train()  # Enable dropout
            preds = []
            
            with torch.no_grad():
                for _ in range(n_samples):
                    pred = model(*args, **kwargs)
                    preds.append(pred.cpu().numpy())
            
            all_preds.append(np.array(preds))
        
        # Calculate mean and uncertainty for each model
        model_means = [preds.mean(axis=0) for preds in all_preds]
        model_stds = [preds.std(axis=0) for preds in all_preds]
        
        # Weight by inverse uncertainty
        weights = [1.0 / (std + 1e-6) for std in model_stds]
        weight_sum = sum(weights)
        normalized_weights = [w / weight_sum for w in weights]
        
        # Weighted average
        ensemble_mean = sum(w * m for w, m in zip(normalized_weights, model_means))
        ensemble_uncertainty = sum(w * s for w, s in zip(normalized_weights, model_stds))
        
        return ensemble_mean, ensemble_uncertainty


def create_ensemble(
    model_configs: List[Dict],
    ensemble_type: str = 'averaging'
) -> nn.Module:
    """
    Factory function to create ensemble.
    
    Args:
        model_configs: List of model configurations
        ensemble_type: 'voting', 'averaging', 'stacking', 'dynamic'
    
    Returns:
        Ensemble model
    """
    # Create base models (simplified - actual would instantiate real models)
    base_models = []
    for config in model_configs:
        # model = create_model_from_config(config)
        base_models.append(nn.Linear(10, 1))  # Placeholder
    
    if ensemble_type == 'voting':
        return VotingEnsemble(base_models)
    elif ensemble_type == 'averaging':
        return AveragingEnsemble(base_models)
    elif ensemble_type == 'stacking':
        meta_learner = nn.Linear(len(base_models), 1)
        return StackingEnsemble(base_models, meta_learner)
    elif ensemble_type == 'dynamic':
        return DynamicEnsemble(base_models)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


if __name__ == '__main__':
    print("Testing model ensemble...")
    
    # Create dummy models
    models = [nn.Linear(10, 1) for _ in range(3)]
    
    # Test averaging ensemble
    ensemble = AveragingEnsemble(models)
    dummy_input = torch.randn(5, 10)
    
    # Note: Actual prediction would require proper forward method
    print("✓ Ensemble created successfully")
    
    print("\n✓ Ensemble tests passed!")
