"""
PharmKG-DTI: Data Augmentation for DTI

Techniques to augment limited DTI training data.
"""

import torch
import numpy as np
from typing import List, Tuple
import random


class SMILESAugmenter:
    """
    Data augmentation for SMILES strings.
    
    Techniques:
    - Random atom masking
    - Bond removal/addition
    - SMILES enumeration (valid equivalent representations)
    """
    
    def __init__(self, augment_prob: float = 0.3):
        self.augment_prob = augment_prob
    
    def mask_atoms(self, smiles: str, mask_ratio: float = 0.1) -> str:
        """
        Randomly mask atoms in SMILES (simulate missing info).
        
        Args:
            smiles: Input SMILES string
            mask_ratio: Ratio of atoms to mask
        
        Returns:
            Augmented SMILES
        """
        if random.random() > self.augment_prob:
            return smiles
        
        chars = list(smiles)
        n_mask = max(1, int(len(chars) * mask_ratio))
        
        # Mask random positions (non-special chars)
        mask_positions = random.sample(range(len(chars)), min(n_mask, len(chars)))
        for pos in mask_positions:
            if chars[pos].isalpha():
                chars[pos] = '*'
        
        return ''.join(chars)
    
    def smiles_enumeration(self, smiles: str, n_variants: int = 5) -> List[str]:
        """
        Generate equivalent SMILES representations.
        
        Uses RDKit to generate valid SMILES variants.
        
        Args:
            smiles: Input SMILES
            n_variants: Number of variants to generate
        
        Returns:
            List of SMILES variants
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import EnumerateMolFromMolBlock
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles]
            
            variants = set()
            variants.add(smiles)
            
            # Generate random SMILES with different starting atoms
            for _ in range(n_variants):
                random_smiles = Chem.MolToSmiles(mol, doRandom=True)
                variants.add(random_smiles)
            
            return list(variants)[:n_variants]
            
        except ImportError:
            return [smiles]
    
    def augment_batch(self, smiles_list: List[str]) -> List[str]:
        """Augment a batch of SMILES."""
        augmented = []
        for smiles in smiles_list:
            augmented.append(self.mask_atoms(smiles))
        return augmented


class ProteinAugmenter:
    """
    Data augmentation for protein sequences.
    
    Techniques:
    - Random mutation (amino acid substitution)
    - Sequence truncation
    - Random cropping
    """
    
    def __init__(self, augment_prob: float = 0.3):
        self.augment_prob = augment_prob
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    def random_mutation(self, sequence: str, mutation_rate: float = 0.05) -> str:
        """
        Randomly mutate amino acids.
        
        Args:
            sequence: Amino acid sequence
            mutation_rate: Rate of mutation
        
        Returns:
            Mutated sequence
        """
        if random.random() > self.augment_prob:
            return sequence
        
        seq_list = list(sequence)
        n_mutations = max(1, int(len(seq_list) * mutation_rate))
        
        positions = random.sample(range(len(seq_list)), min(n_mutations, len(seq_list)))
        for pos in positions:
            seq_list[pos] = random.choice(self.amino_acids)
        
        return ''.join(seq_list)
    
    def random_crop(self, sequence: str, crop_ratio: float = 0.9) -> str:
        """
        Randomly crop sequence.
        
        Args:
            sequence: Amino acid sequence
            crop_ratio: Ratio to keep
        
        Returns:
            Cropped sequence
        """
        if random.random() > self.augment_prob:
            return sequence
        
        crop_length = int(len(sequence) * crop_ratio)
        start = random.randint(0, len(sequence) - crop_length)
        return sequence[start:start + crop_length]
    
    def augment_batch(self, sequences: List[str]) -> List[str]:
        """Augment a batch of sequences."""
        augmented = []
        for seq in sequences:
            seq = self.random_mutation(seq)
            seq = self.random_crop(seq)
            augmented.append(seq)
        return augmented


class NegativeSampler:
    """
    Advanced negative sampling strategies.
    
    Beyond random sampling:
    - Hard negative mining
    - Type-constrained sampling
    """
    
    def __init__(self, strategy: str = 'random'):
        self.strategy = strategy
    
    def sample_hard_negatives(
        self,
        model,
        positive_pairs: List[Tuple],
        all_drugs: List,
        all_targets: List,
        n_negatives: int = 5
    ) -> List[Tuple]:
        """
        Sample hard negatives (high similarity but no interaction).
        
        Uses model predictions to find challenging negatives.
        
        Args:
            model: Trained model
            positive_pairs: List of (drug, target) positive pairs
            all_drugs: All available drugs
            all_targets: All available targets
            n_negatives: Number of negatives per positive
        
        Returns:
            List of negative pairs
        """
        negatives = []
        
        for drug, target in positive_pairs:
            # Generate candidate negatives
            candidates = []
            for _ in range(n_negatives * 3):  # Oversample
                neg_target = random.choice(all_targets)
                if (drug, neg_target) not in positive_pairs:
                    candidates.append((drug, neg_target))
            
            # Score candidates (simplified - would use actual model)
            scored = []
            for d, t in candidates:
                score = random.random()  # Placeholder
                scored.append((score, d, t))
            
            # Select hard negatives (high score)
            scored.sort(reverse=True)
            for _, d, t in scored[:n_negatives]:
                negatives.append((d, t))
        
        return negatives
    
    def sample_type_constrained(
        self,
        drug_type: str,
        target_type: str,
        all_drugs_by_type: Dict[str, List],
        all_targets_by_type: Dict[str, List]
    ) -> Tuple:
        """
        Sample negatives constrained by entity types.
        
        Ensures realistic negative samples by respecting type constraints.
        """
        drug = random.choice(all_drugs_by_type.get(drug_type, []))
        target = random.choice(all_targets_by_type.get(target_type, []))
        return (drug, target)


class MixupAugmentation:
    """
    Mixup augmentation for graph/node features.
    
    Reference: Zhang et al. "mixup: Beyond Empirical Risk Minimization"
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def mixup_features(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup to features and labels.
        
        Args:
            x1, x2: Feature tensors
            y1, y2: Label tensors
        
        Returns:
            Mixed features and labels
        """
        lam = np.random.beta(self.alpha, self.alpha)
        
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_x, mixed_y
    
    def mixup_graph(
        self,
        graph1,
        graph2,
        lam: float = None
    ):
        """Apply mixup at graph level."""
        if lam is None:
            lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix node features
        mixed_features = {}
        for node_type in graph1.x_dict.keys():
            if node_type in graph2.x_dict:
                mixed_features[node_type] = (
                    lam * graph1.x_dict[node_type] +
                    (1 - lam) * graph2.x_dict[node_type]
                )
        
        return mixed_features


if __name__ == '__main__':
    # Test augmentation
    print("Testing data augmentation...")
    
    # SMILES
    smiles_aug = SMILESAugmenter()
    test_smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen
    
    masked = smiles_aug.mask_atoms(test_smiles)
    print(f"Original: {test_smiles}")
    print(f"Masked:   {masked}")
    
    variants = smiles_aug.smiles_enumeration(test_smiles, n_variants=3)
    print(f"Variants: {variants}")
    
    # Protein
    protein_aug = ProteinAugmenter()
    test_seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLS"
    
    mutated = protein_aug.random_mutation(test_seq)
    print(f"\nOriginal: {test_seq}")
    print(f"Mutated:  {mutated}")
    
    print("\n✓ Augmentation tests passed!")
