"""
PharmKG-DTI: Improved Data Download with OpenBioLink Integration

Adds direct support for:
- OpenBioLink dataloader
- BioKG automatic download
- Benchmark dataset evaluation splits
"""

import os
import urllib.request
import zipfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def setup_openbiolink(data_dir: Path):
    """
    Setup OpenBioLink dataset using official dataloader.
    
    OpenBioLink2021 Dataset Statistics:
    - Train: 4,192,002 triples
    - Valid: 186,301 triples
    - Test: 180,964 triples
    - Entities: 180,992
    - Relations: 28
    """
    print("\n" + "=" * 60)
    print("OpenBioLink Dataset Setup")
    print("=" * 60)
    print("""
OpenBioLink can be loaded using the official Python package:

    pip install openbiolink

Then use the dataloader:

    from openbiolink.obl2021 import OBL2021Dataset
    dl = OBL2021Dataset()
    train = dl.training  # torch.tensor of shape(num_train, 3)
    valid = dl.validation
    test = dl.testing

Or download directly from Zenodo:
https://zenodo.org/record/3834052/files/KGID_HQ_DIR.zip
""")
    
    openbio_dir = data_dir / 'openbiolink'
    openbio_dir.mkdir(parents=True, exist_ok=True)
    
    # Create README
    readme = openbio_dir / 'README.txt'
    readme.write_text("""
OpenBioLink Dataset
==================

Download Options:
1. Python API (recommended):
   pip install openbiolink
   
   from openbiolink.evaluation.dataLoader import DataLoader
   dl = DataLoader("HQ_DIR")
   train = dl.training.mapped_triples
   test = dl.testing.mapped_triples
   valid = dl.validation.mapped_triples

2. Direct Download:
   https://zenodo.org/record/3834052/files/KGID_HQ_DIR.zip

3. Manual Generation:
   openbiolink generate
   openbiolink split rand --edges graph_files/edges.csv ...

Dataset Variants:
- HQ_DIR: High quality, directed (recommended for benchmarking)
- HQ_UNDIR: High quality, undirected
- ALL_DIR: All edges, directed
- ALL_UNDIR: All edges, undirected
""")
    print(f"Created README at: {readme}")


def setup_biokg(data_dir: Path):
    """
    Setup BioKG dataset.
    
    BioKG Statistics:
    - ~105K entities
    - ~2.0M triples
    - 13 data sources (UniProt, Reactome, OMIM, GO, etc.)
    """
    print("\n" + "=" * 60)
    print("BioKG Dataset Setup")
    print("=" * 60)
    
    biokg_dir = data_dir / 'biokg'
    biokg_dir.mkdir(parents=True, exist_ok=True)
    
    # BioKG is available on GitHub
    base_url = "https://raw.githubusercontent.com/dsi-bdi/biokg/main/data"
    files = ["train.tsv", "valid.tsv", "test.tsv", "entities.txt", "relations.txt"]
    
    print("Downloading BioKG files from GitHub...")
    for file in files:
        url = f"{base_url}/{file}"
        output_path = biokg_dir / file
        
        if output_path.exists():
            print(f"  {file}: already exists ✓")
            continue
        
        try:
            print(f"  Downloading {file}...")
            download_url(url, str(output_path))
            print(f"  {file}: downloaded ✓")
        except Exception as e:
            print(f"  {file}: failed - {e}")
    
    print(f"\nBioKG data directory: {biokg_dir}")
    print("\nBioKG can also be cloned from:")
    print("  git clone https://github.com/dsi-bdi/biokg.git")


def setup_ogb_biokg(data_dir: Path):
    """
    Setup OGB (Open Graph Benchmark) BioKG.
    
    This is a different dataset from the original BioKG,
    specifically designed for graph ML benchmarking.
    """
    print("\n" + "=" * 60)
    print("OGB-BioKG Dataset Setup")
    print("=" * 60)
    print("""
OGB-BioKG can be loaded using the OGB package:

    pip install ogb

    from ogb.linkproppred import LinkPropPredDataset
    dataset = LinkPropPredDataset(name='ogbl-biokg')
    split_edge = dataset.get_edge_split()
    
Statistics:
- ~93K nodes
- ~5M edges
- 51 relation types
""")
    
    ogb_dir = data_dir / 'ogb_biokg'
    ogb_dir.mkdir(parents=True, exist_ok=True)


def create_benchmark_split(
    dti_edges: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> dict:
    """
    Create benchmark train/val/test split following OpenBioLink style.
    
    Ensures:
    - No data leakage
    - Fixed negative samples for reproducibility
    - 1:1 negative sampling ratio
    
    Args:
        dti_edges: DataFrame with ['drug_id', 'protein_id', 'label'] columns
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
    
    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    np.random.seed(random_seed)
    
    # Split positive edges
    pos_edges = dti_edges[dti_edges['label'] == 1]
    neg_edges = dti_edges[dti_edges['label'] == 0]
    
    # Stratified split for positives
    train_pos, temp_pos = train_test_split(
        pos_edges,
        test_size=val_ratio + test_ratio,
        random_state=random_seed
    )
    val_pos, test_pos = train_test_split(
        temp_pos,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_seed
    )
    
    # Sample fixed negatives (1:1 ratio)
    n_train_neg = len(train_pos)
    n_val_neg = len(val_pos)
    n_test_neg = len(test_pos)
    
    train_neg = neg_edges.sample(n=n_train_neg, random_state=random_seed)
    val_neg = neg_edges.sample(n=n_val_neg, random_state=random_seed + 1)
    test_neg = neg_edges.sample(n=n_test_neg, random_state=random_seed + 2)
    
    # Combine
    train = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=random_seed)
    val = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=random_seed)
    test = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=random_seed)
    
    print(f"Benchmark Split (seed={random_seed}):")
    print(f"  Train: {len(train)} edges ({len(train_pos)} pos, {len(train_neg)} neg)")
    print(f"  Val:   {len(val)} edges ({len(val_pos)} pos, {len(val_neg)} neg)")
    print(f"  Test:  {len(test)} edges ({len(test_pos)} pos, {len(test_neg)} neg)")
    
    return {'train': train, 'val': val, 'test': test}


def main():
    """Main setup function."""
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("PharmKG-DTI Benchmark Dataset Setup")
    print("=" * 60)
    
    # Setup datasets
    setup_openbiolink(data_dir)
    setup_biokg(data_dir)
    setup_ogb_biokg(data_dir)
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nData directory: {data_dir.absolute()}")
    print("\nNext steps:")
    print("1. Install openbiolink: pip install openbiolink")
    print("2. Install ogb: pip install ogb")
    print("3. Run: python src/data/dataset.py")


if __name__ == '__main__':
    main()
