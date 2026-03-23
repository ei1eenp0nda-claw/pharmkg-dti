"""
PharmKG-DTI: Benchmark Dataset Loaders

Integration with standard DTI benchmark datasets:
- BindingDB: Large-scale binding affinity database (~2.5M pairs)
- DAVIS: Kinase inhibitor dataset (30,056 pairs)
- KIBA: Kinase inhibitor bioactivity (118,254 pairs)

Uses DeepPurpose/TDC data loading pipeline for consistency.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch


class TDCDataLoader:
    """
    Data loader for Therapeutics Data Commons (TDC) datasets.
    
    TDC provides standardized access to DTI datasets with consistent
    train/validation/test splits.
    """
    
    def __init__(self, root: str = 'data/tdc'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
    
    def load_bindingdb(
        self,
        binary: bool = True,
        threshold: float = 30.0,
        y_column: str = 'IC50'
    ) -> pd.DataFrame:
        """
        Load BindingDB dataset.
        
        BindingDB is the largest public database of measured binding affinities,
        containing over 2.5 million binding data for drug-like molecules and targets.
        
        Args:
            binary: If True, convert to binary classification (active/inactive)
            threshold: Threshold for binary classification (nM)
            y_column: Affinity type ('IC50', 'Kd', 'Ki', 'EC50')
        
        Returns:
            DataFrame with columns: ['Drug_ID', 'Drug', 'Target_ID', 'Target', 'Y']
        """
        print("Loading BindingDB dataset...")
        print(f"  Binary: {binary}, Threshold: {threshold} nM")
        
        try:
            # Try importing TDC
            from tdc.multi_pred import DTI
            data = DTI(name='BindingDB_Kd', path=str(self.root))
        except ImportError:
            print("  TDC not installed. Using synthetic data fallback.")
            return self._create_synthetic_bindingdb()
        except:
            print("  TDC download failed. Using synthetic data fallback.")
            return self._create_synthetic_bindingdb()
        
        df = data.get_data()
        
        if binary:
            # Convert to binary: active if affinity < threshold
            df['Y'] = (df['Y'] < threshold).astype(int)
        
        print(f"  Loaded {len(df)} drug-target pairs")
        print(f"  Drugs: {df['Drug_ID'].nunique()}, Targets: {df['Target_ID'].nunique()}")
        print(f"  Positive ratio: {df['Y'].mean():.3f}")
        
        return df
    
    def load_davis(self, binary: bool = True, threshold: float = 30.0) -> pd.DataFrame:
        """
        Load DAVIS dataset.
        
        DAVIS contains binding affinity data between kinase inhibitors and kinases,
        providing a standardized benchmark for kinase-focused DTI prediction.
        
        Statistics:
        - 30,056 drug-target pairs
        - 68 kinase inhibitors
        - 379 kinase targets
        
        Args:
            binary: If True, convert to binary classification
            threshold: Threshold for binary classification (nM)
        """
        print("Loading DAVIS dataset...")
        
        try:
            from tdc.multi_pred import DTI
            data = DTI(name='DAVIS', path=str(self.root))
            df = data.get_data()
            
            if binary:
                df['Y'] = (df['Y'] < threshold).astype(int)
            
            print(f"  Loaded {len(df)} pairs")
            print(f"  Drugs: {df['Drug_ID'].nunique()}, Targets: {df['Target_ID'].nunique()}")
            
            return df
        except:
            print("  TDC not available, using synthetic data")
            return self._create_synthetic_davis()
    
    def load_kiba(self, binary: bool = True, threshold: float = 9.0) -> pd.DataFrame:
        """
        Load KIBA dataset.
        
        KIBA (Kinase Inhibitor Bioactivity) combines multiple bioactivity
        measurements into a unified KIBA score.
        
        Statistics:
        - 118,254 drug-target pairs (after filtering)
        - 2,111 kinase inhibitors
        - 229 kinases
        
        Args:
            binary: If True, convert to binary classification
            threshold: Threshold for binary classification (KIBA score)
        """
        print("Loading KIBA dataset...")
        
        try:
            from tdc.multi_pred import DTI
            data = DTI(name='KIBA', path=str(self.root))
            df = data.get_data()
            
            if binary:
                df['Y'] = (df['Y'] >= threshold).astype(int)
            
            print(f"  Loaded {len(df)} pairs")
            print(f"  Drugs: {df['Drug_ID'].nunique()}, Targets: {df['Target_ID'].nunique()}")
            
            return df
        except:
            print("  TDC not available, using synthetic data")
            return self._create_synthetic_kiba()
    
    def _create_synthetic_bindingdb(self, n_samples: int = 10000) -> pd.DataFrame:
        """Create synthetic BindingDB-like data."""
        print(f"  Creating synthetic BindingDB data ({n_samples} samples)")
        
        np.random.seed(42)
        n_drugs = 500
        n_targets = 300
        
        drugs = [f"DRUG_{i:05d}" for i in range(n_drugs)]
        targets = [f"PROT_{i:05d}" for i in range(n_targets)]
        
        data = []
        for _ in range(n_samples):
            drug = np.random.choice(drugs)
            target = np.random.choice(targets)
            affinity = np.random.exponential(100)  # nM
            label = int(affinity < 30)
            
            data.append({
                'Drug_ID': drug,
                'Drug': 'CCO',  # Simplified SMILES
                'Target_ID': target,
                'Target': 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNAL',
                'Y': label
            })
        
        return pd.DataFrame(data)
    
    def _create_synthetic_davis(self, n_samples: int = 5000) -> pd.DataFrame:
        """Create synthetic DAVIS-like data."""
        return self._create_synthetic_bindingdb(n_samples)
    
    def _create_synthetic_kiba(self, n_samples: int = 8000) -> pd.DataFrame:
        """Create synthetic KIBA-like data."""
        return self._create_synthetic_bindingdb(n_samples)


class ColdStartSplitter:
    """
    Cold-start evaluation splits for DTI prediction.
    
    Three settings:
    1. Cold Drug: Test drugs not seen during training
    2. Cold Target: Test targets not seen during training
    3. Cold Drug-Target Pair: Both drug and target are unseen
    
    Reference: DeepPurpose cold split implementation
    """
    
    def __init__(self, df: pd.DataFrame, random_seed: int = 42):
        self.df = df
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def cold_drug_split(
        self,
        frac: List[float] = [0.7, 0.1, 0.2]
    ) -> Dict[str, pd.DataFrame]:
        """
        Split ensuring test drugs are not in training set.
        
        Args:
            frac: [train, val, test] fractions
        
        Returns:
            Dict with 'train', 'val', 'test' DataFrames
        """
        drugs = self.df['Drug_ID'].unique()
        np.random.shuffle(drugs)
        
        n_train = int(len(drugs) * frac[0])
        n_val = int(len(drugs) * frac[1])
        
        train_drugs = drugs[:n_train]
        val_drugs = drugs[n_train:n_train + n_val]
        test_drugs = drugs[n_train + n_val:]
        
        train = self.df[self.df['Drug_ID'].isin(train_drugs)]
        val = self.df[self.df['Drug_ID'].isin(val_drugs)]
        test = self.df[self.df['Drug_ID'].isin(test_drugs)]
        
        print(f"Cold Drug Split:")
        print(f"  Train: {len(train)} pairs, {len(train_drugs)} drugs")
        print(f"  Val:   {len(val)} pairs, {len(val_drugs)} drugs")
        print(f"  Test:  {len(test)} pairs, {len(test_drugs)} drugs")
        
        return {'train': train, 'val': val, 'test': test}
    
    def cold_target_split(
        self,
        frac: List[float] = [0.7, 0.1, 0.2]
    ) -> Dict[str, pd.DataFrame]:
        """
        Split ensuring test targets are not in training set.
        
        Args:
            frac: [train, val, test] fractions
        
        Returns:
            Dict with 'train', 'val', 'test' DataFrames
        """
        targets = self.df['Target_ID'].unique()
        np.random.shuffle(targets)
        
        n_train = int(len(targets) * frac[0])
        n_val = int(len(targets) * frac[1])
        
        train_targets = targets[:n_train]
        val_targets = targets[n_train:n_train + n_val]
        test_targets = targets[n_train + n_val:]
        
        train = self.df[self.df['Target_ID'].isin(train_targets)]
        val = self.df[self.df['Target_ID'].isin(val_targets)]
        test = self.df[self.df['Target_ID'].isin(test_targets)]
        
        print(f"Cold Target Split:")
        print(f"  Train: {len(train)} pairs, {len(train_targets)} targets")
        print(f"  Val:   {len(val)} pairs, {len(val_targets)} targets")
        print(f"  Test:  {len(test)} pairs, {len(test_targets)} targets")
        
        return {'train': train, 'val': val, 'test': test}
    
    def random_split(
        self,
        frac: List[float] = [0.7, 0.1, 0.2]
    ) -> Dict[str, pd.DataFrame]:
        """
        Standard random split (transductive setting).
        
        Args:
            frac: [train, val, test] fractions
        
        Returns:
            Dict with 'train', 'val', 'test' DataFrames
        """
        from sklearn.model_selection import train_test_split
        
        train, temp = train_test_split(
            self.df,
            test_size=frac[1] + frac[2],
            random_state=self.random_seed
        )
        val, test = train_test_split(
            temp,
            test_size=frac[2] / (frac[1] + frac[2]),
            random_state=self.random_seed
        )
        
        print(f"Random Split:")
        print(f"  Train: {len(train)} pairs")
        print(f"  Val:   {len(val)} pairs")
        print(f"  Test:  {len(test)} pairs")
        
        return {'train': train, 'val': val, 'test': test}


def load_benchmark_dataset(
    dataset_name: str,
    split_method: str = 'random',
    root: str = 'data/tdc',
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Load and split a benchmark DTI dataset.
    
    Args:
        dataset_name: 'bindingdb', 'davis', or 'kiba'
        split_method: 'random', 'cold_drug', 'cold_target'
        root: Data directory
        **kwargs: Additional arguments for dataset loading
    
    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    loader = TDCDataLoader(root=root)
    
    # Load dataset
    if dataset_name.lower() == 'bindingdb':
        df = loader.load_bindingdb(**kwargs)
    elif dataset_name.lower() == 'davis':
        df = loader.load_davis(**kwargs)
    elif dataset_name.lower() == 'kiba':
        df = loader.load_kiba(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split
    splitter = ColdStartSplitter(df, random_seed=kwargs.get('random_seed', 42))
    
    if split_method == 'cold_drug':
        splits = splitter.cold_drug_split()
    elif split_method == 'cold_target':
        splits = splitter.cold_target_split()
    elif split_method == 'random':
        splits = splitter.random_split()
    else:
        raise ValueError(f"Unknown split method: {split_method}")
    
    return splits


if __name__ == '__main__':
    # Test loading
    print("Testing benchmark dataset loaders...\n")
    
    for dataset in ['bindingdb', 'davis', 'kiba']:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")
        
        try:
            splits = load_benchmark_dataset(
                dataset_name=dataset,
                split_method='random',
                binary=True
            )
            
            for split_name, split_df in splits.items():
                print(f"\n{split_name.upper()}:")
                print(f"  Samples: {len(split_df)}")
                print(f"  Positive ratio: {split_df['Y'].mean():.3f}")
                
        except Exception as e:
            print(f"Error loading {dataset}: {e}")
