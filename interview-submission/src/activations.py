"""
Activations Analysis Module

This module handles the extraction and analysis of neural network activations,
addressing the key question: "What are activations and how do I find them?"
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ActivationExtractor:
    """
    Extracts activations from transformer models for interpretability analysis.
    
    This addresses the question: "What are activations? How do I find the activations 
    on a particular token on a given piece of text?"
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        """
        Initialize the activation extractor.
        
        Args:
            model_name: Name of the model to analyze
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        
    def get_activations(self, text: str, layer_idx: int = -1) -> Dict:
        """
        Extract activations for a given text.
        
        Args:
            text: Input text to analyze
            layer_idx: Layer to extract activations from (-1 for last layer)
            
        Returns:
            Dictionary containing activations, tokens, and metadata
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Register hooks to capture activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks for the specified layer
        if hasattr(self.model, 'transformer'):
            # GPT-2 style architecture
            target_layer = self.model.transformer.h[layer_idx]
        elif hasattr(self.model, 'layers'):
            # Other architectures
            target_layer = self.model.layers[layer_idx]
        else:
            raise ValueError(f"Unknown model architecture for {self.model_name}")
            
        hook = target_layer.register_forward_hook(hook_fn(f"layer_{layer_idx}"))
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Remove hook
        hook.remove()
        
        # Get tokens for reference
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return {
            'activations': activations[f"layer_{layer_idx}"],
            'tokens': tokens,
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
    
    def get_token_activations(self, text: str, token_idx: int, layer_idx: int = -1) -> torch.Tensor:
        """
        Get activations for a specific token.
        
        Args:
            text: Input text
            token_idx: Index of the token to analyze
            layer_idx: Layer to extract from
            
        Returns:
            Activation vector for the specified token
        """
        result = self.get_activations(text, layer_idx)
        activations = result['activations'][0]  # Remove batch dimension
        
        # Get activation for specific token
        if token_idx < activations.shape[0]:
            return activations[token_idx]
        else:
            raise ValueError(f"Token index {token_idx} out of range")
    
    def analyze_activation_patterns(self, text: str, layer_idx: int = -1) -> Dict:
        """
        Analyze activation patterns across tokens.
        
        Args:
            text: Input text
            layer_idx: Layer to analyze
            
        Returns:
            Dictionary with activation statistics
        """
        result = self.get_activations(text, layer_idx)
        activations = result['activations'][0]
        
        # Calculate statistics
        stats = {
            'mean_activation': activations.mean(dim=0),
            'std_activation': activations.std(dim=0),
            'max_activation': activations.max(dim=0)[0],
            'min_activation': activations.min(dim=0)[0],
            'sparsity': (activations == 0).float().mean(),
            'tokens': result['tokens']
        }
        
        return stats
    
    def visualize_activations(self, text: str, layer_idx: int = -1, 
                            save_path: Optional[str] = None):
        """
        Create a heatmap visualization of activations.
        
        Args:
            text: Input text
            layer_idx: Layer to visualize
            save_path: Optional path to save the plot
        """
        result = self.get_activations(text, layer_idx)
        activations = result['activations'][0].cpu().numpy()
        tokens = result['tokens']
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(activations.T, 
                   xticklabels=tokens,
                   yticklabels=False,
                   cmap='viridis',
                   cbar_kws={'label': 'Activation Value'})
        
        plt.title(f'Activation Heatmap - Layer {layer_idx}')
        plt.xlabel('Tokens')
        plt.ylabel('Neurons')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def explain_activations():
    """
    Educational function to explain what activations are.
    
    This addresses the assessment question about understanding activations.
    """
    explanation = """
    WHAT ARE ACTIVATIONS?
    
    Activations are the output values of neurons in a neural network layer.
    In transformer models like GPT-2:
    
    1. Each token in the input text gets converted to a vector representation
    2. This vector flows through the model's layers
    3. At each layer, the input gets transformed by:
       - Self-attention mechanisms
       - Feed-forward networks (MLPs)
    4. The output of each layer is called the "activation"
    5. Activations represent the model's internal representation of the input
    
    WHY ARE ACTIVATIONS IMPORTANT FOR INTERPRETABILITY?
    
    - Activations show how the model "thinks" about each token
    - Different neurons may respond to different features (e.g., syntax, semantics)
    - By analyzing activations, we can understand what the model has learned
    - This helps us build more interpretable and trustworthy AI systems
    
    HOW TO FIND ACTIVATIONS:
    
    1. Load a pre-trained model
    2. Tokenize your input text
    3. Run a forward pass through the model
    4. Extract the output of specific layers
    5. Analyze the activation patterns
    """
    
    print(explanation)
    return explanation 