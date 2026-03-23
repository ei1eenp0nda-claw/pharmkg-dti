"""
PharmKG-DTI: Data Loading and Preprocessing Module

Handles loading of heterogeneous biomedical knowledge graphs including:
- Drug-Target Interactions (DTI)
- Drug-Drug Interactions (DDI)
- Protein-Protein Interactions (PPI)
- Disease associations
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected, remove_self_loops
from sklearn.model_selection import train_test_split
import networkx as nx


class PharmKGDataset:
    """
    Unified dataset handler for heterogeneous biomedical knowledge graphs.
    
    Supports multiple data sources:
    - DrugBank: Drug-target interactions with detailed drug/protein info
    - BioKG: Large-scale biomedical knowledge graph
    - OpenBioLink: Benchmark dataset for link prediction
    - STITCH: Chemical-protein interactions
    - STRING: Protein-protein interactions
    """
    
    def __init__(
        self,
        root: str,
        dataset_name: str = "drugbank",
        use_structure: bool = True,
        use_sequence: bool = True,
        preload: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            root: Root directory for data storage
            dataset_name: Name of the dataset to load
            use_structure: Whether to include drug molecular structure features
            use_sequence: Whether to include protein sequence features
            preload: Whether to preload data into memory
        """
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.use_structure = use_structure
        self.use_sequence = use_sequence
        
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.drug_ids: List[str] = []
        self.protein_ids: List[str] = []
        self.disease_ids: List[str] = []
        
        self.drug_features: Optional[torch.Tensor] = None
        self.protein_features: Optional[torch.Tensor] = None
        
        # Interaction edges
        self.dti_edges: Optional[torch.Tensor] = None  # Drug-Target Interactions
        self.ddi_edges: Optional[torch.Tensor] = None  # Drug-Drug Interactions
        self.ppi_edges: Optional[torch.Tensor] = None  # Protein-Protein Interactions
        
        # Labels for DTI prediction
        self.dti_labels: Optional[torch.Tensor] = None
        
        if preload:
            self.load_data()
    
    def load_data(self):
        """Load or process dataset."""
        processed_file = self.processed_dir / f"{self.dataset_name}_processed.pt"
        
        if processed_file.exists():
            print(f"Loading processed data from {processed_file}")
            data = torch.load(processed_file)
            self._load_from_cache(data)
        else:
            print(f"Processing raw data for {self.dataset_name}")
            self._process_raw_data()
            self._save_processed_data(processed_file)
    
    def _load_from_cache(self, data: Dict):
        """Load data from cached processed file."""
        self.drug_ids = data['drug_ids']
        self.protein_ids = data['protein_ids']
        self.disease_ids = data.get('disease_ids', [])
        self.drug_features = data.get('drug_features')
        self.protein_features = data.get('protein_features')
        self.dti_edges = data['dti_edges']
        self.dti_labels = data['dti_labels']
        self.ddi_edges = data.get('ddi_edges')
        self.ppi_edges = data.get('ppi_edges')
    
    def _process_raw_data(self):
        """Process raw data files based on dataset type."""
        if self.dataset_name == "drugbank":
            self._process_drugbank()
        elif self.dataset_name == "biokg":
            self._process_biokg()
        elif self.dataset_name == "openbiolink":
            self._process_openbiolink()
        elif self.dataset_name == "custom":
            self._process_custom()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _process_drugbank(self):
        """
        Process DrugBank dataset.
        
        Expected files:
        - drug_links.csv: Drug information
        - protein_links.csv: Target protein information
        - dti_edges.csv: Drug-target interaction edges
        """
        # Load drug information
        drug_file = self.raw_dir / "drugbank" / "drug_links.csv"
        if drug_file.exists():
            drug_df = pd.read_csv(drug_file)
            self.drug_ids = drug_df['drugbank_id'].tolist()
            if self.use_structure and 'smiles' in drug_df.columns:
                self.drug_features = self._compute_drug_features(drug_df['smiles'].tolist())
        else:
            # Generate synthetic data for testing
            print("Warning: DrugBank data not found, generating synthetic data")
            self._generate_synthetic_data()
            return
        
        # Load protein information
        protein_file = self.raw_dir / "drugbank" / "protein_links.csv"
        if protein_file.exists():
            protein_df = pd.read_csv(protein_file)
            self.protein_ids = protein_df['uniprot_id'].tolist()
            if self.use_sequence and 'sequence' in protein_df.columns:
                self.protein_features = self._compute_protein_features(protein_df['sequence'].tolist())
        
        # Load DTI edges
        dti_file = self.raw_dir / "drugbank" / "dti_edges.csv"
        if dti_file.exists():
            dti_df = pd.read_csv(dti_file)
            self.dti_edges = self._build_edge_index(dti_df, self.drug_ids, self.protein_ids)
            self.dti_labels = torch.ones(self.dti_edges.size(1))
    
    def _process_biokg(self):
        """Process BioKG dataset."""
        # BioKG has multiple entity and relation types
        kg_file = self.raw_dir / "biokg" / "train.tsv"
        if kg_file.exists():
            triples = pd.read_csv(kg_file, sep='\t', header=None, names=['head', 'relation', 'tail'])
            self._parse_kg_triples(triples)
        else:
            print("Warning: BioKG data not found, generating synthetic data")
            self._generate_synthetic_data()
    
    def _process_openbiolink(self):
        """Process OpenBioLink benchmark dataset."""
        edges_file = self.raw_dir / "openbiolink" / "edges.csv"
        if edges_file.exists():
            edges_df = pd.read_csv(edges_file)
            self._parse_openbiolink_edges(edges_df)
        else:
            print("Warning: OpenBioLink data not found, generating synthetic data")
            self._generate_synthetic_data()
    
    def _process_custom(self):
        """Process custom dataset format."""
        # Expects files: entities.csv, relations.csv, triples.csv
        triples_file = self.raw_dir / "custom" / "triples.csv"
        if triples_file.exists():
            triples = pd.read_csv(triples_file)
            self._parse_kg_triples(triples)
        else:
            self._generate_synthetic_data()
    
    def _parse_kg_triples(self, triples: pd.DataFrame):
        """Parse knowledge graph triples and extract entities/relations."""
        # Extract unique entities by type
        drug_entities = set()
        protein_entities = set()
        disease_entities = set()
        
        for _, row in triples.iterrows():
            head, rel, tail = row['head'], row['relation'], row['tail']
            # Classify entities based on ID patterns or relation type
            if 'DRUG' in str(head).upper() or 'COMPOUND' in str(head).upper():
                drug_entities.add(head)
            if 'PROTEIN' in str(head).upper() or 'GENE' in str(head).upper():
                protein_entities.add(head)
            if 'DISEASE' in str(head).upper():
                disease_entities.add(head)
        
        self.drug_ids = list(drug_entities)
        self.protein_ids = list(protein_entities)
        self.disease_ids = list(disease_entities)
        
        # Build edge indices for different relation types
        self._build_heterogeneous_edges(triples)
    
    def _build_heterogeneous_edges(self, triples: pd.DataFrame):
        """Build edge indices for different relation types."""
        dti_edges_list = []
        ddi_edges_list = []
        ppi_edges_list = []
        
        drug_to_idx = {d: i for i, d in enumerate(self.drug_ids)}
        protein_to_idx = {p: i for i, p in enumerate(self.protein_ids)}
        
        for _, row in triples.iterrows():
            head, rel, tail = row['head'], row['relation'], row['tail']
            rel_lower = str(rel).lower()
            
            # DTI edges
            if 'target' in rel_lower or 'interact' in rel_lower:
                if head in drug_to_idx and tail in protein_to_idx:
                    dti_edges_list.append([drug_to_idx[head], protein_to_idx[tail]])
            # DDI edges
            elif 'drug-drug' in rel_lower or 'interaction' in rel_lower:
                if head in drug_to_idx and tail in drug_to_idx:
                    ddi_edges_list.append([drug_to_idx[head], drug_to_idx[tail]])
            # PPI edges
            elif 'protein-protein' in rel_lower:
                if head in protein_to_idx and tail in protein_to_idx:
                    ppi_edges_list.append([protein_to_idx[head], protein_to_idx[tail]])
        
        if dti_edges_list:
            self.dti_edges = torch.tensor(dti_edges_list, dtype=torch.long).t()
            self.dti_labels = torch.ones(self.dti_edges.size(1))
        if ddi_edges_list:
            self.ddi_edges = torch.tensor(ddi_edges_list, dtype=torch.long).t()
        if ppi_edges_list:
            self.ppi_edges = torch.tensor(ppi_edges_list, dtype=torch.long).t()
    
    def _compute_drug_features(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Compute molecular fingerprints from SMILES strings.
        
        Uses Morgan fingerprints (ECFP) as default.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            fingerprints = []
            for smiles in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
                        fingerprints.append(list(fp))
                    else:
                        fingerprints.append([0] * 256)
                except:
                    fingerprints.append([0] * 256)
            
            return torch.tensor(fingerprints, dtype=torch.float)
        except ImportError:
            print("RDKit not available, using random drug features")
            return torch.randn(len(smiles_list), 256)
    
    def _compute_protein_features(self, sequences: List[str]) -> torch.Tensor:
        """
        Compute protein sequence features.
        
        Uses simple k-mer encoding as baseline.
        """
        # Simple amino acid composition features
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        features = []
        
        for seq in sequences:
            if pd.isna(seq) or not isinstance(seq, str):
                features.append([0] * 20)
                continue
            
            seq = seq.upper()
            aa_counts = [seq.count(aa) / max(len(seq), 1) for aa in amino_acids]
            features.append(aa_counts)
        
        return torch.tensor(features, dtype=torch.float)
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for testing when real data is unavailable."""
        print("Generating synthetic data for testing...")
        
        n_drugs = 500
        n_proteins = 1000
        n_interactions = 2000
        
        self.drug_ids = [f"DRUG_{i}" for i in range(n_drugs)]
        self.protein_ids = [f"PROT_{i}" for i in range(n_proteins)]
        
        # Generate random features
        self.drug_features = torch.randn(n_drugs, 256)
        self.protein_features = torch.randn(n_proteins, 256)
        
        # Generate random DTI edges
        drug_indices = torch.randint(0, n_drugs, (n_interactions,))
        protein_indices = torch.randint(0, n_proteins, (n_interactions,))
        self.dti_edges = torch.stack([drug_indices, protein_indices])
        self.dti_labels = torch.ones(n_interactions)
        
        # Generate some negative samples
        n_negatives = 2000
        neg_drug_indices = torch.randint(0, n_drugs, (n_negatives,))
        neg_protein_indices = torch.randint(0, n_proteins, (n_negatives,))
        
        # Combine positive and negative
        self.dti_edges = torch.cat([self.dti_edges, torch.stack([neg_drug_indices, neg_protein_indices])], dim=1)
        self.dti_labels = torch.cat([self.dti_labels, torch.zeros(n_negatives)])
    
    def _build_edge_index(self, df: pd.DataFrame, src_ids: List[str], dst_ids: List[str]) -> torch.Tensor:
        """Build edge index tensor from DataFrame."""
        src_to_idx = {id_: i for i, id_ in enumerate(src_ids)}
        dst_to_idx = {id_: i for i, id_ in enumerate(dst_ids)}
        
        edges = []
        for _, row in df.iterrows():
            src = row.get('drug_id', row.get('head'))
            dst = row.get('protein_id', row.get('tail'))
            if src in src_to_idx and dst in dst_to_idx:
                edges.append([src_to_idx[src], dst_to_idx[dst]])
        
        return torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long)
    
    def _save_processed_data(self, path: Path):
        """Save processed data to disk."""
        data = {
            'drug_ids': self.drug_ids,
            'protein_ids': self.protein_ids,
            'disease_ids': self.disease_ids,
            'drug_features': self.drug_features,
            'protein_features': self.protein_features,
            'dti_edges': self.dti_edges,
            'dti_labels': self.dti_labels,
            'ddi_edges': self.ddi_edges,
            'ppi_edges': self.ppi_edges
        }
        torch.save(data, path)
        print(f"Saved processed data to {path}")
    
    def to_pyg_hetero_data(self) -> HeteroData:
        """
        Convert dataset to PyTorch Geometric HeteroData format.
        
        Returns:
            HeteroData object containing the heterogeneous graph
        """
        data = HeteroData()
        
        # Add node features
        if self.drug_features is not None:
            data['drug'].x = self.drug_features
        else:
            data['drug'].x = torch.randn(len(self.drug_ids), 256)
        
        if self.protein_features is not None:
            data['protein'].x = self.protein_features
        else:
            data['protein'].x = torch.randn(len(self.protein_ids), 256)
        
        # Add edges
        if self.dti_edges is not None and self.dti_edges.size(1) > 0:
            data['drug', 'interacts', 'protein'].edge_index = self.dti_edges
        
        if self.ddi_edges is not None and self.ddi_edges.size(1) > 0:
            data['drug', 'interacts', 'drug'].edge_index = self.ddi_edges
        
        if self.ppi_edges is not None and self.ppi_edges.size(1) > 0:
            data['protein', 'interacts', 'protein'].edge_index = self.ppi_edges
        
        return data
    
    def get_train_val_test_split(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split DTI edges into train/val/test sets.
        
        Returns:
            train_edges, train_labels, val_edges, val_labels, test_edges, test_labels
        """
        if self.dti_edges is None or self.dti_labels is None:
            raise ValueError("No DTI data available for splitting")
        
        n_edges = self.dti_edges.size(1)
        indices = np.arange(n_edges)
        
        # Stratified split to maintain positive/negative ratio
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=val_ratio + test_ratio,
            stratify=self.dti_labels.numpy(),
            random_state=random_state
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=self.dti_labels[temp_idx].numpy(),
            random_state=random_state
        )
        
        train_edges = self.dti_edges[:, train_idx]
        train_labels = self.dti_labels[train_idx]
        val_edges = self.dti_edges[:, val_idx]
        val_labels = self.dti_labels[val_idx]
        test_edges = self.dti_edges[:, test_idx]
        test_labels = self.dti_labels[test_idx]
        
        return train_edges, train_labels, val_edges, val_labels, test_edges, test_labels
    
    def __len__(self) -> int:
        """Return number of DTI edges."""
        return self.dti_edges.size(1) if self.dti_edges is not None else 0
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset={self.dataset_name}, "
            f"n_drugs={len(self.drug_ids)}, "
            f"n_proteins={len(self.protein_ids)}, "
            f"n_dti={len(self)})"
        )
