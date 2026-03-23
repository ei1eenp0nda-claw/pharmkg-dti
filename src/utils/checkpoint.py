"""
PharmKG-DTI: Model Checkpoint Manager

Utilities for saving, loading, and exporting models.
Supports PyTorch checkpoints, ONNX export, and TorchScript.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union
import shutil
import time

import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages model checkpoints with versioning and metadata.
    
    Features:
    - Automatic versioning
    - Metadata tracking (metrics, config, timestamp)
    - Best model selection
    - Checkpoint cleanup
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        metric_name: str = "val_auc",
        mode: str = "max"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode
        
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.checkpoints = []
        
        # Load existing checkpoints
        self._scan_existing_checkpoints()
    
    def _scan_existing_checkpoints(self):
        """Scan existing checkpoints in directory."""
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime
        )
        self.checkpoints = [str(f) for f in checkpoint_files]
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict] = None,
        config: Optional[Dict] = None,
        is_best: bool = False
    ) -> str:
        """
        Save a model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state (optional)
            epoch: Current epoch
            metrics: Dictionary of metrics (optional)
            config: Model configuration (optional)
            is_best: Whether this is the best model so far
        
        Returns:
            Path to saved checkpoint
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch{epoch:03d}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model_architecture': model.__class__.__name__,
            'timestamp': timestamp,
            'metrics': metrics or {},
            'config': config or {}
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(str(checkpoint_path))
        
        # Save as best model if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy(checkpoint_path, best_path)
            print(f"✓ Saved best model to {best_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"✓ Saved checkpoint: {checkpoint_name}")
        return str(checkpoint_path)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if exceeding max."""
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            old_path = Path(old_checkpoint)
            if old_path.exists() and old_path.name != 'best_model.pt':
                old_path.unlink()
                print(f"  Removed old checkpoint: {old_path.name}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Dict:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            device: Device to load to
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"  Metrics: {checkpoint.get('metrics', {})}")
        
        return checkpoint
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return str(best_path)
        return None
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        return sorted(self.checkpoints)


class ModelExporter:
    """
    Export models to different formats for deployment.
    
    Supports:
    - PyTorch checkpoint (.pt)
    - TorchScript (.ts)
    - ONNX (.onnx)
    """
    
    def __init__(self, output_dir: str = "exported_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_torchscript(
        self,
        model: nn.Module,
        example_inputs: tuple,
        filename: str = "model.ts"
    ) -> str:
        """
        Export model to TorchScript format.
        
        TorchScript models can be loaded in C++ and are optimized for inference.
        
        Args:
            model: Model to export
            example_inputs: Example inputs for tracing
            filename: Output filename
        
        Returns:
            Path to exported model
        """
        model.eval()
        output_path = self.output_dir / filename
        
        try:
            # Try scripting first
            scripted_model = torch.jit.script(model)
        except:
            # Fallback to tracing
            scripted_model = torch.jit.trace(model, example_inputs)
        
        scripted_model.save(str(output_path))
        print(f"✓ Exported TorchScript model to {output_path}")
        
        return str(output_path)
    
    def export_onnx(
        self,
        model: nn.Module,
        example_inputs: tuple,
        filename: str = "model.onnx",
        input_names: Optional[list] = None,
        output_names: Optional[list] = None
    ) -> str:
        """
        Export model to ONNX format.
        
        ONNX models are framework-agnostic and can run on various runtimes.
        
        Args:
            model: Model to export
            example_inputs: Example inputs
            filename: Output filename
            input_names: Names for inputs
            output_names: Names for outputs
        
        Returns:
            Path to exported model
        """
        model.eval()
        output_path = self.output_dir / filename
        
        torch.onnx.export(
            model,
            example_inputs,
            str(output_path),
            input_names=input_names or ['input'],
            output_names=output_names or ['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=11
        )
        
        print(f"✓ Exported ONNX model to {output_path}")
        return str(output_path)
    
    def export_production_bundle(
        self,
        model: nn.Module,
        config: Dict,
        filename: str = "production_bundle"
    ) -> str:
        """
        Export a complete production bundle with model, config, and metadata.
        
        Args:
            model: Model to export
            config: Model configuration
            filename: Base filename (without extension)
        
        Returns:
            Path to bundle directory
        """
        bundle_dir = self.output_dir / filename
        bundle_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = bundle_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save config
        config_path = bundle_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save metadata
        metadata = {
            'model_class': model.__class__.__name__,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        metadata_path = bundle_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Exported production bundle to {bundle_dir}")
        return str(bundle_dir)


def create_model_card(
    model_name: str,
    description: str,
    architecture: str,
    performance_metrics: Dict,
    output_path: str = "MODEL_CARD.md"
):
    """
    Create a model card documenting the model.
    
    Model cards improve transparency and reproducibility.
    
    Args:
        model_name: Name of the model
        description: Description of the model
        architecture: Architecture details
        performance_metrics: Dictionary of performance metrics
        output_path: Where to save the model card
    """
    card_content = f"""# Model Card: {model_name}

## Model Description

{description}

## Architecture

```
{architecture}
```

## Performance Metrics

| Metric | Value |
|--------|-------|
"""
    
    for metric, value in performance_metrics.items():
        if isinstance(value, float):
            card_content += f"| {metric} | {value:.4f} |\n"
        else:
            card_content += f"| {metric} | {value} |\n"
    
    card_content += f"""

## Intended Use

This model is designed for predicting drug-target interactions.

## Limitations

- Performance may vary on unseen drug/target types
- Requires valid SMILES strings and protein sequences
- Not intended for clinical decision-making without validation

## Training Data

See training documentation for dataset details.

## Citation

If you use this model, please cite:

```
@software{{pharmkg_dti,
  title = {{PharmKG-DTI: Heterogeneous Graph Neural Networks for Drug-Target Interaction Prediction}},
  year = {{2026}}
}}
```

---
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    with open(output_path, 'w') as f:
        f.write(card_content)
    
    print(f"✓ Created model card: {output_path}")


if __name__ == '__main__':
    # Example usage
    print("Testing Checkpoint Manager...")
    
    # Create dummy model
    dummy_model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    # Test checkpoint manager
    manager = CheckpointManager(checkpoint_dir="test_checkpoints")
    
    # Save checkpoint
    path = manager.save_checkpoint(
        model=dummy_model,
        epoch=1,
        metrics={'auc': 0.95, 'aupr': 0.88},
        config={'lr': 0.001, 'batch_size': 32},
        is_best=True
    )
    
    print(f"Checkpoint saved to: {path}")
    print(f"Checkpoints: {manager.list_checkpoints()}")
    
    # Test exporter
    print("\nTesting Model Exporter...")
    exporter = ModelExporter(output_dir="test_exports")
    
    example_input = torch.randn(1, 128)
    ts_path = exporter.export_torchscript(
        dummy_model,
        (example_input,),
        filename="test_model.ts"
    )
    
    bundle_path = exporter.export_production_bundle(
        dummy_model,
        config={'hidden_dim': 128, 'num_layers': 2},
        filename="test_bundle"
    )
    
    print(f"\n✓ All tests passed!")
