"""
Utilities Module

Helper functions for the mechanistic interpretability assessment.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
import json
from datetime import datetime


def setup_device() -> str:
    """
    Set up and return the best available device (MPS for Apple Silicon, CUDA for NVIDIA, else CPU).
    
    Returns:
        Device string ('mps', 'cuda', or 'cpu')
    """
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal Performance Shaders) for Apple Silicon")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("Using CPU (MPS and CUDA not available)")
    
    return device


def print_gpu_info():
    """Print detailed GPU information if available."""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    else:
        print("No GPU available")


def create_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)


def save_experiment_results(results: Dict, filename: str):
    """
    Save experiment results to JSON file.
    
    Args:
        results: Dictionary containing experiment results
        filename: Name of the file to save
    """
    create_results_dir()
    
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    
    filepath = os.path.join("results", filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {filepath}")


def load_experiment_results(filename: str) -> Dict:
    """
    Load experiment results from JSON file.
    
    Args:
        filename: Name of the file to load
        
    Returns:
        Dictionary containing experiment results
    """
    filepath = os.path.join("results", filename)
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def plot_training_curves(history: Dict, save_path: Optional[str] = None):
    """
    Plot training curves for model training.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training curves
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train')
        axes[0, 0].plot(history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    if 'train_acc' in history and 'val_acc' in history:
        axes[0, 1].plot(history['train_acc'], label='Train')
        axes[0, 1].plot(history['val_acc'], label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two tensors.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Cosine similarity
    """
    return F.cosine_similarity(a, b, dim=-1)


def normalize_activations(activations: torch.Tensor) -> torch.Tensor:
    """
    Normalize activations to zero mean and unit variance.
    
    Args:
        activations: Input activations
        
    Returns:
        Normalized activations
    """
    mean = activations.mean(dim=0, keepdim=True)
    std = activations.std(dim=0, keepdim=True)
    return (activations - mean) / (std + 1e-8)


def compute_sparsity(activations: torch.Tensor) -> float:
    """
    Compute sparsity of activations.
    
    Args:
        activations: Input activations
        
    Returns:
        Sparsity ratio (fraction of zero elements)
    """
    return (activations == 0).float().mean().item()


def find_most_active_neurons(activations: torch.Tensor, top_k: int = 10) -> List[int]:
    """
    Find the most active neurons.
    
    Args:
        activations: Input activations
        top_k: Number of top neurons to return
        
    Returns:
        List of neuron indices
    """
    mean_activations = activations.mean(dim=0)
    _, indices = torch.topk(mean_activations, top_k)
    return indices.tolist()


def find_least_active_neurons(activations: torch.Tensor, top_k: int = 10) -> List[int]:
    """
    Find the least active neurons.
    
    Args:
        activations: Input activations
        top_k: Number of bottom neurons to return
        
    Returns:
        List of neuron indices
    """
    mean_activations = activations.mean(dim=0)
    _, indices = torch.topk(-mean_activations, top_k)
    return indices.tolist()


def create_activation_dataset(activations: torch.Tensor, 
                            batch_size: int = 256) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader from activations.
    
    Args:
        activations: Input activations
        batch_size: Batch size for the DataLoader
        
    Returns:
        DataLoader
    """
    dataset = torch.utils.data.TensorDataset(activations)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def save_model(model: torch.nn.Module, filepath: str):
    """
    Save a PyTorch model.
    
    Args:
        model: Model to save
        filepath: Path to save the model
    """
    create_results_dir()
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model: torch.nn.Module, filepath: str):
    """
    Load a PyTorch model.
    
    Args:
        model: Model to load weights into
        filepath: Path to the saved model
    """
    model.load_state_dict(torch.load(filepath))
    print(f"Model loaded from {filepath}")


def print_model_summary(model: torch.nn.Module):
    """
    Print a summary of the model architecture.
    
    Args:
        model: Model to summarize
    """
    print("Model Summary:")
    print("=" * 50)
    
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape} - {param.numel()} parameters")
            total_params += param.numel()
    
    print("=" * 50)
    print(f"Total trainable parameters: {total_params:,}")


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print(f"Random seed set to {seed}")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_assessment_progress(current_step: int, total_steps: int, step_name: str):
    """
    Print progress for the assessment.
    
    Args:
        current_step: Current step number
        total_steps: Total number of steps
        step_name: Name of the current step
    """
    progress = (current_step / total_steps) * 100
    print(f"\n{'='*60}")
    print(f"ASSESSMENT PROGRESS: {current_step}/{total_steps} ({progress:.1f}%)")
    print(f"Current Step: {step_name}")
    print(f"{'='*60}\n") 