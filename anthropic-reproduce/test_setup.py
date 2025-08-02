#!/usr/bin/env python3
"""
Test script to verify M2 Mac setup for mechanistic interpretability assessment.
"""

import torch
import transformers
import matplotlib.pyplot as plt
import numpy as np

def test_setup():
    """Test that all components are working correctly."""
    print("🧪 Testing M2 Mac Setup for Mechanistic Interpretability")
    print("=" * 60)
    
    # Test PyTorch and device
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ MPS (Metal Performance Shaders) is available!")
        print("✅ Using Apple Silicon acceleration")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"✅ CUDA is available: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("⚠️  Using CPU (no GPU acceleration)")
    
    # Test basic tensor operations
    print(f"\n🔧 Testing tensor operations on {device}...")
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = torch.mm(x, y)
    print(f"✅ Matrix multiplication successful: {z.shape}")
    
    # Test transformers
    print(f"\n🤖 Testing transformers library...")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        model = transformers.AutoModel.from_pretrained("gpt2").to(device)
        print("✅ GPT-2 model loaded successfully")
        
        # Test inference
        text = "Hello world"
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✅ Inference successful: {outputs.last_hidden_state.shape}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    # Test visualization
    print(f"\n📊 Testing visualization libraries...")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x_plot = np.linspace(0, 10, 100)
        y_plot = np.sin(x_plot)
        ax.plot(x_plot, y_plot)
        ax.set_title("Test Plot")
        plt.close(fig)
        print("✅ Matplotlib working correctly")
    except Exception as e:
        print(f"❌ Error with matplotlib: {e}")
    
    print(f"\n🎉 Setup test completed!")
    print(f"Device: {device}")
    print(f"Ready for mechanistic interpretability analysis!")
    
    return device

if __name__ == "__main__":
    test_setup() 