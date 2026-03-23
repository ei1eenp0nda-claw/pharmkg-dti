"""
PharmKG-DTI: Configuration Management

Centralized configuration with validation.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
from dataclasses import dataclass, asdict, field


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "dhgt_dti"
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.3
    use_kge: bool = True
    kge_model: str = "RotatE"


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 200
    early_stopping_patience: int = 20
    gradient_clip: float = 1.0
    warmup_epochs: int = 5
    scheduler: str = "cosine"  # cosine, plateau, step


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "bindingdb"  # bindingdb, davis, kiba
    split_method: str = "random"  # random, cold_drug, cold_target
    split_ratio: list = field(default_factory=lambda: [0.7, 0.1, 0.2])
    negative_sampling_ratio: float = 1.0
    use_augmentation: bool = False
    augmentation_prob: float = 0.3


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: list = field(default_factory=lambda: ["auc", "aupr", "f1", "mcc"])
    eval_every: int = 5
    save_predictions: bool = True
    n_negatives_eval: int = 100


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str = "pharmkg_dti_experiment"
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    output_dir: str = "outputs"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Load config from dictionary."""
        return cls(
            experiment_name=config_dict.get('experiment_name', 'experiment'),
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'auto'),
            output_dir=config_dict.get('output_dir', 'outputs'),
            log_dir=config_dict.get('log_dir', 'logs'),
            checkpoint_dir=config_dict.get('checkpoint_dir', 'checkpoints'),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {}))
        )
    
    def to_yaml(self, path: str):
        """Save config to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate configuration."""
        assert self.model.hidden_dim > 0, "hidden_dim must be positive"
        assert self.model.num_layers > 0, "num_layers must be positive"
        assert 0 <= self.model.dropout < 1, "dropout must be in [0, 1)"
        assert self.training.batch_size > 0, "batch_size must be positive"
        assert self.training.learning_rate > 0, "learning_rate must be positive"
        assert self.training.epochs > 0, "epochs must be positive"
        return True


# Default configurations for different scenarios

def get_default_config() -> ExperimentConfig:
    """Get default configuration."""
    return ExperimentConfig()


def get_fast_config() -> ExperimentConfig:
    """Get fast config for quick experiments."""
    return ExperimentConfig(
        experiment_name="fast_experiment",
        model=ModelConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4
        ),
        training=TrainingConfig(
            batch_size=512,
            epochs=50,
            early_stopping_patience=10
        )
    )


def get_production_config() -> ExperimentConfig:
    """Get production-ready configuration."""
    return ExperimentConfig(
        experiment_name="production_run",
        model=ModelConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            dropout=0.2
        ),
        training=TrainingConfig(
            batch_size=256,
            epochs=500,
            early_stopping_patience=50,
            warmup_epochs=10
        ),
        evaluation=EvaluationConfig(
            save_predictions=True,
            eval_every=1
        )
    )


if __name__ == '__main__':
    print("Testing configuration management...")
    
    # Test default config
    config = get_default_config()
    print(f"\nDefault config: {config.to_dict()}")
    
    # Test validation
    assert config.validate()
    print("✓ Config validation passed")
    
    # Test save/load
    config.to_yaml("test_config.yaml")
    loaded = ExperimentConfig.from_yaml("test_config.yaml")
    print(f"✓ Config saved and loaded")
    
    # Test fast config
    fast = get_fast_config()
    print(f"\nFast config epochs: {fast.training.epochs}")
    
    print("\n✓ Configuration tests passed!")
