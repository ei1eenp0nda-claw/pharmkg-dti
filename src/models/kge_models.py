"""
PharmKG-DTI: Knowledge Graph Embedding Module

Integration with PyKEEN for knowledge graph embedding and link prediction.
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.models import predict


class KGEModel:
    """
    Knowledge Graph Embedding model wrapper using PyKEEN.
    
    Supports various KG embedding models:
    - TransE, TransH, TransD
    - RotatE, ComplEx, DistMult
    - HolE, RESCAL
    """
    
    def __init__(
        self,
        model_name: str = 'RotatE',
        embedding_dim: int = 256,
        random_seed: int = 42
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.random_seed = random_seed
        self.model = None
        self.triples_factory = None
    
    def train(
        self,
        triples: List[Tuple[str, str, str]],
        validation_triples: Optional[List[Tuple[str, str, str]]] = None,
        num_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        device: str = 'cuda'
    ):
        """
        Train KG embedding model.
        
        Args:
            triples: List of (head, relation, tail) triples
            validation_triples: Optional validation triples
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            device: Device to use
        """
        # Convert to numpy array
        triples_array = np.array(triples)
        
        # Create triples factory
        self.triples_factory = TriplesFactory.from_labeled_triples(triples_array)
        
        # Run pipeline
        result = pipeline(
            training=self.triples_factory,
            testing=self.triples_factory,  # Use same for now, should be separate
            model=self.model_name,
            model_kwargs=dict(
                embedding_dim=self.embedding_dim
            ),
            optimizer='Adam',
            optimizer_kwargs=dict(
                lr=learning_rate
            ),
            training_kwargs=dict(
                num_epochs=num_epochs,
                batch_size=batch_size,
                use_tqdm=True
            ),
            random_seed=self.random_seed,
            device=device
        )
        
        self.model = result.model
        
        return result
    
    def predict_triple_score(
        self,
        head: str,
        relation: str,
        tail: str
    ) -> float:
        """
        Predict score for a single triple.
        
        Args:
            head: Head entity
            relation: Relation type
            tail: Tail entity
        
        Returns:
            Predicted score
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Pack into dataframe
        import pandas as pd
        df = pd.DataFrame([(head, relation, tail)], columns=['head', 'relation', 'tail'])
        
        # Predict
        scores = predict.predict_triples_df(
            model=self.model,
            triples=df,
            triples_factory=self.triples_factory
        )
        
        return scores['score'].values[0]
    
    def get_entity_embeddings(self) -> Dict[str, np.ndarray]:
        """Get embeddings for all entities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        entity_embeddings = {}
        entity_to_id = self.triples_factory.entity_to_id
        
        for entity, idx in entity_to_id.items():
            emb = self.model.entity_representations[0](
                torch.tensor([idx])
            ).detach().cpu().numpy()
            entity_embeddings[entity] = emb[0]
        
        return entity_embeddings
    
    def get_relation_embeddings(self) -> Dict[str, np.ndarray]:
        """Get embeddings for all relations."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        relation_embeddings = {}
        relation_to_id = self.triples_factory.relation_to_id
        
        for relation, idx in relation_to_id.items():
            emb = self.model.relation_representations[0](
                torch.tensor([idx])
            ).detach().cpu().numpy()
            relation_embeddings[relation] = emb[0]
        
        return relation_embeddings


class EnsembleKGEM:
    """
    Ensemble of multiple KG embedding models.
    
    Based on: "Ensembles of knowledge graph embedding models improve predictions for drug discovery"
    """
    
    def __init__(
        self,
        model_names: List[str] = ['RotatE', 'ComplEx', 'DistMult'],
        embedding_dim: int = 256
    ):
        self.model_names = model_names
        self.embedding_dim = embedding_dim
        self.models: Dict[str, KGEModel] = {}
    
    def train_all(
        self,
        triples: List[Tuple[str, str, str]],
        **kwargs
    ):
        """Train all ensemble models."""
        for model_name in self.model_names:
            print(f"Training {model_name}...")
            model = KGEModel(
                model_name=model_name,
                embedding_dim=self.embedding_dim
            )
            model.train(triples, **kwargs)
            self.models[model_name] = model
    
    def ensemble_predict(
        self,
        head: str,
        relation: str,
        tail: str,
        method: str = 'mean'
    ) -> float:
        """
        Ensemble prediction using multiple methods.
        
        Args:
            head: Head entity
            relation: Relation type
            tail: Tail entity
            method: 'mean', 'min', 'max', or 'product'
        
        Returns:
            Ensemble score
        """
        scores = []
        for model in self.models.values():
            try:
                score = model.predict_triple_score(head, relation, tail)
                scores.append(score)
            except:
                continue
        
        if not scores:
            return 0.0
        
        if method == 'mean':
            return np.mean(scores)
        elif method == 'min':
            return np.min(scores)
        elif method == 'max':
            return np.max(scores)
        elif method == 'product':
            return np.prod(scores)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")


def convert_to_pykeen_triples(
    dataset: 'PharmKGDataset'
) -> List[Tuple[str, str, str]]:
    """
    Convert PharmKGDataset to PyKEEN triples format.
    
    Args:
        dataset: PharmKGDataset instance
    
    Returns:
        List of (head, relation, tail) triples
    """
    triples = []
    
    # Add DTI edges
    if dataset.dti_edges is not None:
        for i in range(dataset.dti_edges.size(1)):
            drug_idx = dataset.dti_edges[0, i].item()
            prot_idx = dataset.dti_edges[1, i].item()
            triples.append((
                dataset.drug_ids[drug_idx],
                'interacts_with',
                dataset.protein_ids[prot_idx]
            ))
    
    # Add DDI edges
    if dataset.ddi_edges is not None:
        for i in range(dataset.ddi_edges.size(1)):
            drug1_idx = dataset.ddi_edges[0, i].item()
            drug2_idx = dataset.ddi_edges[1, i].item()
            triples.append((
                dataset.drug_ids[drug1_idx],
                'interacts_with',
                dataset.drug_ids[drug2_idx]
            ))
    
    # Add PPI edges
    if dataset.ppi_edges is not None:
        for i in range(dataset.ppi_edges.size(1)):
            prot1_idx = dataset.ppi_edges[0, i].item()
            prot2_idx = dataset.ppi_edges[1, i].item()
            triples.append((
                dataset.protein_ids[prot1_idx],
                'interacts_with',
                dataset.protein_ids[prot2_idx]
            ))
    
    return triples
