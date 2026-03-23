"""
PharmKG-DTI: Heterogeneous Graph Neural Network Models

Implements state-of-the-art architectures for DTI prediction:
1. DHGT-DTI: Dual-View Heterogeneous Graph Transformer
2. HGAN: Heterogeneous Graph Attention Network
3. GraphSAGE-based link prediction baseline
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    HeteroConv,
    GATConv,
    SAGEConv,
    GCNConv,
    TransformerConv,
    Linear,
    MessagePassing
)
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType


class HeterogeneousGNNLayer(nn.Module):
    """
    A generic heterogeneous GNN layer that handles multiple edge types.
    
    Uses different message passing mechanisms for different relation types.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.3,
        aggr: str = 'mean',
        use_attention: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_attention = use_attention
        
        # Will be populated based on edge types
        self.convs = None
        self.dropout = nn.Dropout(dropout)
        
    def build_convs(self, edge_types: List[EdgeType]):
        """Build convolutions for given edge types."""
        convs = {}
        for edge_type in edge_types:
            src, rel, dst = edge_type
            if self.use_attention:
                convs[edge_type] = GATConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim // self.num_heads,
                    heads=self.num_heads,
                    dropout=self.dropout.p,
                    add_self_loops=False,
                    concat=True
                )
            else:
                convs[edge_type] = SAGEConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    aggr='mean'
                )
        
        self.convs = HeteroConv(convs, aggr='mean')
    
    def forward(
        self,
        x_dict: Dict[NodeType, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor]
    ) -> Dict[NodeType, torch.Tensor]:
        """Forward pass through heterogeneous layer."""
        if self.convs is None:
            self.build_convs(list(edge_index_dict.keys()))
        
        out = self.convs(x_dict, edge_index_dict)
        out = {key: self.dropout(F.relu(x)) for key, x in out.items()}
        return out


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer with residual connections.
    
    Captures global structure through attention mechanism.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Edge connectivity (optional)
            mask: Attention mask (optional)
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = residual + attn_out
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


class DHGTDTI(nn.Module):
    """
    DHGT-DTI: Dual-View Heterogeneous Graph Transformer for DTI Prediction
    
    Combines:
    1. Neighborhood view: Heterogeneous GNN (GraphSAGE-based)
    2. Meta-path view: Graph Transformer for higher-order relations
    
    Reference: "DHGT-DTI: Advancing Drug-Target Interaction Prediction through 
    a Dual-View Heterogeneous Network with GraphSAGE and Graph Transformer"
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        num_drug_features: int = 256,
        num_protein_features: int = 256,
        dropout: float = 0.3,
        use_residual: bool = True,
        edge_types: Optional[List[EdgeType]] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        # Feature projection layers
        self.drug_encoder = nn.Sequential(
            nn.Linear(num_drug_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.protein_encoder = nn.Sequential(
            nn.Linear(num_protein_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Neighborhood view: Heterogeneous GNN layers
        self.neighborhood_layers = nn.ModuleList([
            HeterogeneousGNNLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_attention=False  # Use SAGEConv
            ) for _ in range(num_layers)
        ])
        
        # Meta-path view: Graph Transformer layers
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout * 0.5,  # Lower dropout for transformer
                use_residual=use_residual
            ) for _ in range(num_layers)
        )
        
        # Cross-view attention fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def encode(
        self,
        data: HeteroData
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Encode nodes using both neighborhood and transformer views.
        
        Returns:
            neighborhood_embeddings: Dict of node embeddings from GNN view
            transformer_embeddings: Dict of node embeddings from transformer view
        """
        # Initial feature encoding
        x_dict = {
            'drug': self.drug_encoder(data['drug'].x),
            'protein': self.protein_encoder(data['protein'].x)
        }
        
        # Add more node types if present
        for node_type in data.node_types:
            if node_type not in x_dict and 'x' in data[node_type]:
                x_dict[node_type] = data[node_type].x
        
        # Neighborhood view
        neigh_x = x_dict.copy()
        edge_index_dict = {k: v.edge_index for k, v in data.edge_items()}
        
        for layer in self.neighborhood_layers:
            neigh_x = layer(neigh_x, edge_index_dict)
        
        # Meta-path view (Graph Transformer)
        # Aggregate all nodes for global attention
        all_nodes = []
        node_types = []
        for node_type, x in x_dict.items():
            all_nodes.append(x)
            node_types.extend([node_type] * x.size(0))
        
        transformer_x = torch.cat(all_nodes, dim=0)
        
        for layer in self.transformer_layers:
            transformer_x = layer(transformer_x)
        
        # Split back into node types
        transformer_dict = {}
        idx = 0
        for node_type, x in x_dict.items():
            n_nodes = x.size(0)
            transformer_dict[node_type] = transformer_x[idx:idx + n_nodes]
            idx += n_nodes
        
        return neigh_x, transformer_dict
    
    def fuse_embeddings(
        self,
        neigh_emb: Dict[str, torch.Tensor],
        trans_emb: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Fuse embeddings from both views using cross-attention."""
        fused = {}
        
        for node_type in neigh_emb.keys():
            if node_type not in trans_emb:
                fused[node_type] = neigh_emb[node_type]
                continue
            
            n = neigh_emb[node_type]
            t = trans_emb[node_type]
            
            # Concatenate for cross-attention
            combined = torch.stack([n, t], dim=1)  # [N, 2, D]
            
            # Self-attention over the two views
            attended, _ = self.cross_attention(combined, combined, combined)
            
            # Mean pooling over views
            fused[node_type] = attended.mean(dim=1)
        
        return fused
    
    def predict(
        self,
        drug_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict DTI scores for given edges.
        
        Args:
            drug_emb: Drug node embeddings [N_drugs, hidden_dim]
            protein_emb: Protein node embeddings [N_proteins, hidden_dim]
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            scores: Predicted interaction scores [num_edges]
        """
        src, dst = edge_index
        
        # Get embeddings for edge endpoints
        drug_features = drug_emb[src]
        protein_features = protein_emb[dst]
        
        # Concatenate and predict
        combined = torch.cat([drug_features, protein_features], dim=-1)
        scores = self.predictor(combined).squeeze(-1)
        
        return scores
    
    def forward(
        self,
        data: HeteroData,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Full forward pass.
        
        Args:
            data: Heterogeneous graph data
            edge_index: DTI edges to predict [2, num_edges]
        
        Returns:
            scores: Predicted interaction scores
        """
        # Encode nodes with dual views
        neigh_emb, trans_emb = self.encode(data)
        
        # Fuse embeddings
        fused_emb = self.fuse_embeddings(neigh_emb, trans_emb)
        
        # Predict
        scores = self.predict(
            fused_emb['drug'],
            fused_emb['protein'],
            edge_index
        )
        
        return scores


class HGANDTI(nn.Module):
    """
    HGAN: Heterogeneous Graph Attention Network for DTI Prediction
    
    Uses enhanced graph attention diffusion to capture long-range dependencies.
    
    Reference: "Heterogeneous Graph Attention Network for Drug-Target Interaction Prediction"
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        num_drug_features: int = 256,
        num_protein_features: int = 256,
        dropout: float = 0.3,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature encoders
        self.drug_encoder = nn.Sequential(
            nn.Linear(num_drug_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.protein_encoder = nn.Sequential(
            nn.Linear(num_protein_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Enhanced Graph Attention Diffusion Layers
        self.attention_layers = nn.ModuleList([
            HeterogeneousGNNLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=attention_dropout,
                use_attention=True
            ) for _ in range(num_layers)
        ])
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        data: HeteroData,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        # Encode features
        x_dict = {
            'drug': self.drug_encoder(data['drug'].x),
            'protein': self.protein_encoder(data['protein'].x)
        }
        
        # Get edge indices
        edge_index_dict = {k: v.edge_index for k, v in data.edge_items()}
        
        # Apply attention layers
        for layer in self.attention_layers:
            x_dict = layer(x_dict, edge_index_dict)
        
        # Predict
        src, dst = edge_index
        drug_features = x_dict['drug'][src]
        protein_features = x_dict['protein'][dst]
        
        combined = torch.cat([drug_features, protein_features], dim=-1)
        scores = self.predictor(combined).squeeze(-1)
        
        return scores


class SAGEBaseline(nn.Module):
    """
    Simple GraphSAGE baseline for DTI prediction.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_drug_features: int = 256,
        num_protein_features: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.drug_encoder = nn.Sequential(
            nn.Linear(num_drug_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.protein_encoder = nn.Sequential(
            nn.Linear(num_protein_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.layers = nn.ModuleList([
            HeterogeneousGNNLayer(
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_attention=False
            ) for _ in range(num_layers)
        ])
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        data: HeteroData,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        x_dict = {
            'drug': self.drug_encoder(data['drug'].x),
            'protein': self.protein_encoder(data['protein'].x)
        }
        
        edge_index_dict = {k: v.edge_index for k, v in data.edge_items()}
        
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        
        src, dst = edge_index
        combined = torch.cat([x_dict['drug'][src], x_dict['protein'][dst]], dim=-1)
        
        return self.predictor(combined).squeeze(-1)


def create_model(
    model_name: str,
    **kwargs
) -> nn.Module:
    """Factory function to create models by name."""
    models = {
        'dhgt': DHGTDTI,
        'hgan': HGANDTI,
        'sage': SAGEBaseline
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name.lower()](**kwargs)
