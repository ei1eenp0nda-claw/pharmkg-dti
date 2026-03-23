"""
PharmKG-DTI: Distributed Training Support

Multi-GPU and distributed training utilities using PyTorch DDP.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Initialize distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def get_distributed_model(model: torch.nn.Module, rank: int) -> DDP:
    """
    Wrap model for distributed training.
    
    Args:
        model: Model to wrap
        rank: Process rank
    
    Returns:
        DDP wrapped model
    """
    model = model.to(rank)
    return DDP(model, device_ids=[rank], find_unused_parameters=True)


def get_distributed_dataloader(
    dataset,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Create dataloader with distributed sampler.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size per GPU
        rank: Process rank
        world_size: Total number of processes
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader with DistributedSampler
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )


class DistributedTrainer:
    """
    Trainer with distributed training support.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rank: int,
        world_size: int
    ):
        self.rank = rank
        self.world_size = world_size
        self.model = get_distributed_model(model, rank)
        self.optimizer = optimizer
    
    def train_step(self, batch) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward
        loss = self.model(batch)
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """Save checkpoint (only from rank 0)."""
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                **kwargs
            }
            torch.save(checkpoint, path)
            print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=f'cuda:{self.rank}')
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0)


if __name__ == '__main__':
    print("Distributed training utilities ready")
