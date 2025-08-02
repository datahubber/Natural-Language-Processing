"""
Sparse Autoencoder (SAE) Module

This module implements Sparse Autoencoders for feature extraction and analysis.
Addresses the key questions:
- What is the purpose of training the auxiliary SAE network?
- Why does the SAE hidden layer have higher dimensionality than the original network?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for feature extraction from neural activations.
    
    This addresses the question: "What is the purpose of training the auxiliary SAE network?"
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, sparsity_target: float = 0.01,
                 sparsity_weight: float = 0.1, l1_weight: float = 1e-5):
        """
        Initialize the Sparse Autoencoder.
        
        Args:
            input_dim: Dimension of input activations
            hidden_dim: Dimension of hidden layer (features)
            sparsity_target: Target sparsity level
            sparsity_weight: Weight for sparsity penalty
            l1_weight: Weight for L1 regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.l1_weight = l1_weight
        
        # Encoder: input_dim -> hidden_dim
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Decoder: hidden_dim -> input_dim
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        # Xavier initialization for better training
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations [batch_size, input_dim]
            
        Returns:
            Tuple of (encoded_features, reconstructed_input)
        """
        # Encode
        encoded = self.encoder(x)
        
        # Apply ReLU activation for sparsity
        encoded = F.relu(encoded)
        
        # Decode
        reconstructed = self.decoder(encoded)
        
        return encoded, reconstructed
    
    def compute_loss(self, x: torch.Tensor, encoded: torch.Tensor, 
                    reconstructed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss including reconstruction, sparsity, and L1 penalties.
        
        Args:
            x: Original input
            encoded: Encoded features
            reconstructed: Reconstructed input
            
        Returns:
            Dictionary containing all loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, x)
        
        # Sparsity penalty (KL divergence from target sparsity)
        sparsity_loss = self._compute_sparsity_loss(encoded)
        
        # L1 regularization on weights
        l1_loss = self._compute_l1_loss()
        
        # Total loss
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss + self.l1_weight * l1_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'sparsity': sparsity_loss,
            'l1': l1_loss
        }
    
    def _compute_sparsity_loss(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence sparsity penalty.
        
        Args:
            encoded: Encoded features
            
        Returns:
            Sparsity loss
        """
        # Compute average activation across batch
        avg_activation = encoded.mean(dim=0)
        
        # KL divergence from target sparsity
        kl_div = (self.sparsity_target * torch.log(self.sparsity_target / (avg_activation + 1e-8)) +
                 (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target) / (1 - avg_activation + 1e-8)))
        
        return kl_div.sum()
    
    def _compute_l1_loss(self) -> torch.Tensor:
        """Compute L1 regularization on weights."""
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_loss


class SAETrainer:
    """
    Trainer for Sparse Autoencoders.
    
    This addresses the question: "Why does the SAE hidden layer have higher dimensionality 
    than the hidden layer of the original network?"
    """
    
    def __init__(self, sae: SparseAutoencoder, device: str = "cuda"):
        """
        Initialize the SAE trainer.
        
        Args:
            sae: Sparse Autoencoder model
            device: Device to train on
        """
        self.sae = sae.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.sae.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
    def train(self, activations: torch.Tensor, epochs: int = 100, 
              batch_size: int = 256, val_split: float = 0.1) -> Dict:
        """
        Train the Sparse Autoencoder.
        
        Args:
            activations: Input activations [num_samples, input_dim]
            epochs: Number of training epochs
            batch_size: Batch size for training
            val_split: Fraction of data to use for validation
            
        Returns:
            Training history
        """
        # Prepare data
        dataset = TensorDataset(activations)
        
        # Split into train/val
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon': [],
            'val_recon': [],
            'train_sparsity': [],
            'val_sparsity': []
        }
        
        print(f"Training SAE for {epochs} epochs...")
        print(f"Input dim: {self.sae.input_dim}, Hidden dim: {self.sae.hidden_dim}")
        
        for epoch in tqdm(range(epochs)):
            # Training
            train_losses = self._train_epoch(train_loader)
            
            # Validation
            val_losses = self._validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_losses['total'])
            
            # Record history
            for key in history:
                if key.startswith('train_'):
                    history[key].append(train_losses[key[6:]])
                elif key.startswith('val_'):
                    history[key].append(val_losses[key[4:]])
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_losses['total']:.6f}, Val Loss: {val_losses['total']:.6f}")
                print(f"Train Recon: {train_losses['reconstruction']:.6f}, Val Recon: {val_losses['reconstruction']:.6f}")
                print(f"Train Sparsity: {train_losses['sparsity']:.6f}, Val Sparsity: {val_losses['sparsity']:.6f}")
                print("-" * 50)
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.sae.train()
        total_losses = {'total': 0, 'reconstruction': 0, 'sparsity': 0, 'l1': 0}
        num_batches = 0
        
        for batch in train_loader:
            x = batch[0].to(self.device)
            
            # Forward pass
            encoded, reconstructed = self.sae(x)
            
            # Compute loss
            losses = self.sae.compute_loss(x, encoded, reconstructed)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            self.optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1
        
        # Average losses
        return {key: total_losses[key] / num_batches for key in total_losses}
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.sae.eval()
        total_losses = {'total': 0, 'reconstruction': 0, 'sparsity': 0, 'l1': 0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                
                # Forward pass
                encoded, reconstructed = self.sae(x)
                
                # Compute loss
                losses = self.sae.compute_loss(x, encoded, reconstructed)
                
                # Accumulate losses
                for key in total_losses:
                    total_losses[key] += losses[key].item()
                num_batches += 1
        
        # Average losses
        return {key: total_losses[key] / num_batches for key in total_losses}
    
    def extract_features(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from activations using the trained SAE.
        
        Args:
            activations: Input activations
            
        Returns:
            Extracted features
        """
        self.sae.eval()
        with torch.no_grad():
            features, _ = self.sae(activations.to(self.device))
        return features.cpu()


def explain_sae_purpose():
    """
    Explain the purpose of training auxiliary SAE networks.
    
    This addresses the assessment question about SAE purpose.
    """
    explanation = """
    WHAT IS THE PURPOSE OF TRAINING AUXILIARY SAE NETWORKS?
    
    Sparse Autoencoders (SAEs) serve several key purposes in mechanistic interpretability:
    
    1. FEATURE DECOMPOSITION:
       - Neural network activations are often "superposed" - multiple features mixed together
       - SAEs help decompose these mixed activations into individual, interpretable features
       - Each SAE neuron learns to respond to a specific, meaningful pattern
    
    2. SPARSITY INDUCTION:
       - Natural neural networks often have dense, overlapping representations
       - SAEs encourage sparse, non-overlapping feature representations
       - This makes features more interpretable and easier to understand
    
    3. DIMENSIONALITY EXPANSION:
       - SAEs typically have larger hidden layers than input layers
       - This allows the network to "unpack" superposed features
       - More features = more specific, interpretable patterns
    
    4. FEATURE EXTRACTION:
       - SAEs learn to extract meaningful features from raw activations
       - These features can be analyzed independently
       - Helps understand what the original model has learned
    
    WHY IS THIS IMPORTANT?
    
    - Makes neural networks more interpretable
    - Helps identify what features the model uses for decision-making
    - Enables better understanding of model behavior
    - Supports development of more trustworthy AI systems
    """
    
    print(explanation)
    return explanation


def explain_sae_architecture():
    """
    Explain why SAE hidden layers have higher dimensionality.
    
    This addresses the assessment question about SAE architecture.
    """
    explanation = """
    WHY DOES THE SAE HIDDEN LAYER HAVE HIGHER DIMENSIONALITY THAN THE ORIGINAL NETWORK?
    
    This is a key design choice that addresses the "superposition" problem:
    
    1. THE SUPERPOSITION PROBLEM:
       - Neural networks often represent multiple features in the same neurons
       - For example, one neuron might respond to both "cat" and "dog" concepts
       - This makes interpretation difficult
    
    2. DIMENSIONALITY EXPANSION SOLUTION:
       - SAEs use more hidden neurons than input neurons
       - This allows the network to "unpack" superposed features
       - Each superposed feature gets its own dedicated neuron
    
    3. EXAMPLE:
       - Original layer: 768 neurons (mixed features)
       - SAE hidden layer: 2048 neurons (individual features)
       - Result: Each SAE neuron responds to one specific feature
    
    4. BENEFITS:
       - More interpretable: each neuron has a clear, specific function
       - Better analysis: can study individual features in isolation
       - Improved understanding: see what the model actually learned
    
    5. TRADE-OFFS:
       - Higher computational cost
       - More parameters to train
       - Need for sparsity to prevent overfitting
    
    This architecture choice is fundamental to the success of SAE-based interpretability.
    """
    
    print(explanation)
    return explanation


def visualize_sae_training(history: Dict, save_path: Optional[str] = None):
    """
    Visualize SAE training progress.
    
    Args:
        history: Training history from SAETrainer
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Reconstruction loss
    axes[0, 1].plot(history['train_recon'], label='Train')
    axes[0, 1].plot(history['val_recon'], label='Validation')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Sparsity loss
    axes[1, 0].plot(history['train_sparsity'], label='Train')
    axes[1, 0].plot(history['val_sparsity'], label='Validation')
    axes[1, 0].set_title('Sparsity Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate (if available)
    if 'lr' in history:
        axes[1, 1].plot(history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 