"""
PharmKG-DTI: Logging and Monitoring

Structured logging and experiment tracking.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


class ExperimentLogger:
    """
    Comprehensive experiment logging.
    
    Logs to console, file, and optionally to external trackers (WandB, MLflow).
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        config: Dict[str, Any] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # External trackers
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self._setup_external_trackers()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup file and console logging."""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.experiment_name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_external_trackers(self):
        """Initialize external tracking tools."""
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="pharmkg-dti",
                    name=self.experiment_name,
                    config=self.config
                )
                self.wandb = wandb
            except ImportError:
                self.logger.warning("WandB not installed. Skipping.")
                self.use_wandb = False
        
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard" / self.experiment_name
                self.tb_writer = SummaryWriter(tb_dir)
            except ImportError:
                self.logger.warning("TensorBoard not installed. Skipping.")
                self.use_tensorboard = False
    
    def log(self, message: str, level: str = "info"):
        """Log message."""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to all trackers.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        # Log to console
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.log(f"Metrics (step={step}): {metrics_str}")
        
        # Log to WandB
        if self.use_wandb:
            self.wandb.log(metrics, step=step)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, step)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        self.log(f"Hyperparameters: {json.dumps(params, indent=2)}")
        
        if self.use_wandb:
            self.wandb.config.update(params)
    
    def log_model_graph(self, model, input_sample):
        """Log model architecture."""
        if self.use_tensorboard:
            self.tb_writer.add_graph(model, input_sample)
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """Log artifact (model checkpoint, etc.)."""
        if self.use_wandb:
            artifact = self.wandb.Artifact(
                name=f"{self.experiment_name}_{artifact_type}",
                type=artifact_type
            )
            artifact.add_file(artifact_path)
            self.wandb.log_artifact(artifact)
    
    def finish(self):
        """Finish logging and close trackers."""
        if self.use_wandb:
            self.wandb.finish()
        
        if self.use_tensorboard:
            self.tb_writer.close()
        
        self.log("Experiment logging finished.")


class MetricsTracker:
    """
    Track metrics over training epochs.
    """
    
    def __init__(self):
        self.metrics = {}
        self.best_metrics = {}
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """
        Update metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append((epoch, value))
    
    def get_best(self, metric_name: str, mode: str = "max") -> tuple:
        """
        Get best value for a metric.
        
        Args:
            metric_name: Name of the metric
            mode: 'max' or 'min'
        
        Returns:
            (epoch, best_value)
        """
        if metric_name not in self.metrics:
            return None, None
        
        values = self.metrics[metric_name]
        if mode == "max":
            best = max(values, key=lambda x: x[1])
        else:
            best = min(values, key=lambda x: x[1])
        
        return best
    
    def get_history(self, metric_name: str) -> list:
        """Get full history for a metric."""
        return self.metrics.get(metric_name, [])
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            name: [v for _, v in values]
            for name, values in self.metrics.items()
        }
    
    def save(self, path: str):
        """Save metrics to JSON."""
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)


class ProgressLogger:
    """
    Simple progress logging with ETA estimation.
    """
    
    def __init__(self, total_steps: int, desc: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.desc = desc
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        """Update progress."""
        self.current_step += n
        
        if self.current_step % max(1, self.total_steps // 10) == 0:
            self._print_progress()
    
    def _print_progress(self):
        """Print progress bar."""
        percent = 100 * self.current_step / self.total_steps
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.current_step > 0:
            eta = elapsed * (self.total_steps - self.current_step) / self.current_step
            eta_str = f"ETA: {int(eta)}s"
        else:
            eta_str = "ETA: --"
        
        bar_length = 30
        filled = int(bar_length * self.current_step / self.total_steps)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r{self.desc}: |{bar}| {percent:.1f}% {eta_str}", end='', flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # New line at end


if __name__ == '__main__':
    print("Testing logging module...")
    
    # Test experiment logger
    logger = ExperimentLogger(
        experiment_name="test_run",
        log_dir="test_logs",
        config={'lr': 0.001, 'batch_size': 256}
    )
    
    logger.log("Starting experiment...")
    logger.log_metrics({'train_loss': 0.5, 'val_auc': 0.85}, step=1)
    logger.log_metrics({'train_loss': 0.4, 'val_auc': 0.88}, step=2)
    
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update(1, {'loss': 0.5, 'auc': 0.85})
    tracker.update(2, {'loss': 0.4, 'auc': 0.88})
    
    best_epoch, best_auc = tracker.get_best('auc', mode='max')
    print(f"Best AUC: {best_auc} at epoch {best_epoch}")
    
    logger.finish()
    print("\n✓ Logging tests passed!")
