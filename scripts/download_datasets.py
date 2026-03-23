#!/usr/bin/env python3
"""
PharmKG-DTI: Dataset Download Script

Downloads standard DTI benchmark datasets:
- DrugBank
- BioKG
- OpenBioLink
"""

import os
import urllib.request
import zipfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_drugbank(data_dir: Path):
    """Download DrugBank dataset (requires manual download due to licensing)."""
    print("=" * 60)
    print("DrugBank Dataset")
    print("=" * 60)
    print("""
DrugBank requires registration and manual download due to licensing restrictions.

Please follow these steps:
1. Visit: https://go.drugbank.com/releases/latest
2. Register for an account
3. Download the 'DrugBank Open Data' (or full data if you have academic access)
4. Extract to: {data_dir}/drugbank/

Expected files:
- drug_links.csv
- protein_links.csv  
- dti_edges.csv
""".format(data_dir=data_dir))
    
    drugbank_dir = data_dir / 'drugbank'
    drugbank_dir.mkdir(parents=True, exist_ok=True)


def download_biokg(data_dir: Path):
    """Download BioKG dataset from GitHub."""
    print("\n" + "=" * 60)
    print("BioKG Dataset")
    print("=" * 60)
    
    biokg_dir = data_dir / 'biokg'
    biokg_dir.mkdir(parents=True, exist_ok=True)
    
    # BioKG is available on GitHub
    base_url = "https://raw.githubusercontent.com/dsi-bdi/biokg/main/data"
    files = [
        "train.tsv",
        "valid.tsv",
        "test.tsv",
        "entities.txt",
        "relations.txt"
    ]
    
    for file in files:
        url = f"{base_url}/{file}"
        output_path = biokg_dir / file
        
        if output_path.exists():
            print(f"{file} already exists, skipping...")
            continue
        
        try:
            print(f"Downloading {file}...")
            download_url(url, str(output_path))
            print(f"Downloaded {file}")
        except Exception as e:
            print(f"Failed to download {file}: {e}")
            print("BioKG may require manual download from: https://github.com/dsi-bdi/biokg")


def download_openbiolink(data_dir: Path):
    """Download OpenBioLink benchmark dataset."""
    print("\n" + "=" * 60)
    print("OpenBioLink Dataset")
    print("=" * 60)
    
    openbio_dir = data_dir / 'openbiolink'
    openbio_dir.mkdir(parents=True, exist_ok=True)
    
    print("""
OpenBioLink dataset can be downloaded from:
https://github.com/OpenBioLink/OpenBioLink

The dataset will be generated using their pipeline. Alternatively,
you can use their pre-generated benchmark files.

Expected files:
- edges.csv
- nodes.csv
- train.csv
- val.csv
- test.csv
""")


def create_synthetic_dataset(data_dir: Path):
    """Create a small synthetic dataset for testing."""
    print("\n" + "=" * 60)
    print("Creating Synthetic Dataset")
    print("=" * 60)
    
    import pandas as pd
    import numpy as np
    
    synthetic_dir = data_dir / 'synthetic'
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic drugs
    n_drugs = 100
    n_proteins = 200
    n_interactions = 500
    
    drugs = pd.DataFrame({
        'drug_id': [f"DRUG_{i:04d}" for i in range(n_drugs)],
        'name': [f"Drug_{i}" for i in range(n_drugs)],
        'smiles': ['CCO'] * n_drugs  # Placeholder
    })
    
    proteins = pd.DataFrame({
        'protein_id': [f"PROT_{i:04d}" for i in range(n_proteins)],
        'name': [f"Protein_{i}" for i in range(n_proteins)],
        'sequence': ['MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNAL'] * n_proteins
    })
    
    # Generate random interactions
    np.random.seed(42)
    drug_indices = np.random.randint(0, n_drugs, n_interactions)
    protein_indices = np.random.randint(0, n_proteins, n_interactions)
    
    interactions = pd.DataFrame({
        'drug_id': [f"DRUG_{i:04d}" for i in drug_indices],
        'protein_id': [f"PROT_{i:04d}" for i in protein_indices],
        'interaction_type': ['inhibition'] * n_interactions,
        'confidence': np.random.uniform(0.5, 1.0, n_interactions)
    })
    
    # Save
    drugs.to_csv(synthetic_dir / 'drug_links.csv', index=False)
    proteins.to_csv(synthetic_dir / 'protein_links.csv', index=False)
    interactions.to_csv(synthetic_dir / 'dti_edges.csv', index=False)
    
    print(f"Created synthetic dataset:")
    print(f"  - {n_drugs} drugs")
    print(f"  - {n_proteins} proteins")
    print(f"  - {n_interactions} interactions")
    print(f"  Saved to: {synthetic_dir}")


def main():
    """Main download function."""
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("PharmKG-DTI Dataset Download Script")
    print("=" * 60)
    
    # Download/Setup datasets
    download_drugbank(data_dir)
    download_biokg(data_dir)
    download_openbiolink(data_dir)
    
    # Create synthetic dataset for immediate testing
    create_synthetic_dataset(data_dir)
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print(f"\nData directory: {data_dir.absolute()}")
    print("\nNext steps:")
    print("1. For real datasets: Follow manual download instructions above")
    print("2. For testing: Synthetic dataset is ready to use")
    print("3. Run: python src/data/dataset.py to verify data loading")


if __name__ == '__main__':
    main()
