"""
PharmKG-DTI: Model Compression and Quantization

Utilities for optimizing models for deployment:
- Pruning
- Quantization (INT8)
- Knowledge Distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import copy


class ModelPruner:
    """
    Structured pruning for GNN models.
    
    Removes less important weights/channels based on magnitude.
    """
    
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.3):
        self.model = model
        self.pruning_ratio = pruning_ratio
    
    def prune_by_magnitude(self) -> nn.Module:
        """
        Prune weights with smallest absolute magnitude.
        
        Returns:
            Pruned model
        """
        pruned_model = copy.deepcopy(self.model)
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                # Compute weight magnitude
                weight = module.weight.data.abs()
                
                # Find threshold
                flat_weight = weight.view(-1)
                k = int(self.pruning_ratio * flat_weight.numel())
                if k > 0:
                    threshold = torch.kthvalue(flat_weight, k)[0]
                    
                    # Create mask
                    mask = weight > threshold
                    module.weight.data *= mask
        
        return pruned_model
    
    def count_parameters(self, model: nn.Module = None) -> Dict:
        """Count total and non-zero parameters."""
        model = model or self.model
        
        total = 0
        nonzero = 0
        
        for param in model.parameters():
            total += param.numel()
            nonzero += (param != 0).sum().item()
        
        return {
            'total': total,
            'nonzero': nonzero,
            'sparsity': 1 - (nonzero / total)
        }


class ModelQuantizer:
    """
    Post-training quantization for inference optimization.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def quantize_dynamic(self) -> torch.jit.ScriptModule:
        """
        Apply dynamic quantization (INT8 weights).
        
        Returns:
            Quantized model
        """
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model
    
    def quantize_static(self, calibration_data) -> torch.jit.ScriptModule:
        """
        Apply static quantization (requires calibration).
        
        Args:
            calibration_data: Representative data for calibration
        
        Returns:
            Quantized model
        """
        model_fp32 = copy.deepcopy(self.model)
        model_fp32.eval()
        
        # Fuse modules
        model_fp32 = torch.quantization.fuse_modules(
            model_fp32,
            [['linear', 'relu']] if hasattr(model_fp32, 'relu') else []
        )
        
        # Specify quantization config
        model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare
        model_prepared = torch.quantization.prepare(model_fp32)
        
        # Calibrate
        with torch.no_grad():
            for data in calibration_data:
                model_prepared(data)
        
        # Convert
        model_quantized = torch.quantization.convert(model_prepared)
        
        return model_quantized
    
    def compare_model_sizes(self, original_model: nn.Module, quantized_model) -> Dict:
        """Compare model sizes before and after quantization."""
        import io
        
        # Original size
        buffer = io.BytesIO()
        torch.save(original_model.state_dict(), buffer)
        original_size = buffer.tell() / 1024 / 1024  # MB
        
        # Quantized size
        buffer = io.BytesIO()
        torch.save(quantized_model.state_dict(), buffer)
        quantized_size = buffer.tell() / 1024 / 1024  # MB
        
        return {
            'original_mb': original_size,
            'quantized_mb': quantized_size,
            'reduction': (original_size - quantized_size) / original_size * 100
        }


class KnowledgeDistillation:
    """
    Knowledge distillation from teacher to student model.
    
    Reference: Hinton et al. "Distilling the Knowledge in a Neural Network"
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Loss = alpha * KL(soft_teacher, soft_student) + (1-alpha) * CE(student, labels)
        """
        # Soft targets from teacher
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence loss
        distillation_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard target loss
        student_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return loss
    
    def train_student(
        self,
        train_loader,
        optimizer,
        epochs: int = 50
    ):
        """Train student model with knowledge distillation."""
        self.teacher.eval()
        self.student.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Get teacher predictions (no grad)
                with torch.no_grad():
                    teacher_logits = self.teacher(batch)
                
                # Get student predictions
                student_logits = self.student(batch)
                
                # Compute loss
                loss = self.distillation_loss(
                    student_logits,
                    teacher_logits,
                    batch.labels
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


class ModelOptimizer:
    """
    High-level interface for model optimization pipeline.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def optimize_for_deployment(
        self,
        optimization_type: str = 'prune_quantize',
        **kwargs
    ) -> nn.Module:
        """
        Run optimization pipeline.
        
        Args:
            optimization_type: 'prune', 'quantize', 'prune_quantize', 'distill'
        
        Returns:
            Optimized model
        """
        if optimization_type == 'prune':
            pruner = ModelPruner(self.model, kwargs.get('pruning_ratio', 0.3))
            return pruner.prune_by_magnitude()
        
        elif optimization_type == 'quantize':
            quantizer = ModelQuantizer(self.model)
            return quantizer.quantize_dynamic()
        
        elif optimization_type == 'prune_quantize':
            # First prune
            pruner = ModelPruner(self.model, kwargs.get('pruning_ratio', 0.3))
            pruned = pruner.prune_by_magnitude()
            
            # Then quantize
            quantizer = ModelQuantizer(pruned)
            return quantizer.quantize_dynamic()
        
        elif optimization_type == 'distill':
            # Requires teacher and student models
            raise NotImplementedError("Distillation requires separate teacher/student setup")
        
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")
    
    def benchmark_inference(
        self,
        model: nn.Module,
        input_shape: tuple,
        n_runs: int = 100
    ) -> Dict:
        """Benchmark inference speed."""
        import time
        
        model.eval()
        dummy_input = torch.randn(*input_shape)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.time()
                _ = model(dummy_input)
                times.append(time.time() - start)
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000
        }


if __name__ == '__main__':
    # Test optimization
    print("Testing Model Optimization...")
    
    # Create dummy model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    
    # Pruning
    pruner = ModelPruner(model, pruning_ratio=0.3)
    pruned = pruner.prune_by_magnitude()
    stats = pruner.count_parameters(pruned)
    print(f"Pruning stats: {stats}")
    
    # Quantization
    quantizer = ModelQuantizer(pruned)
    quantized = quantizer.quantize_dynamic()
    size_info = quantizer.compare_model_sizes(model, quantized)
    print(f"Size comparison: {size_info}")
    
    print("\n✓ Model optimization tests passed!")
