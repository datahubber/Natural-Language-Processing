#!/bin/bash

echo "Setting up Mechanistic Interpretability Assessment for M2 Mac..."

# Check if we're on Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    echo "Detected Apple Silicon (M1/M2) Mac"
    DEVICE="mps"  # Use Metal Performance Shaders
else
    echo "Detected Intel Mac"
    DEVICE="cpu"
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch for Apple Silicon
if [[ $DEVICE == "mps" ]]; then
    echo "Installing PyTorch for Apple Silicon..."
    pip install torch torchvision torchaudio
else
    echo "Installing PyTorch for Intel..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other requirements..."
pip install transformers datasets einops matplotlib seaborn plotly tqdm jupyter ipywidgets

echo "Setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To start Jupyter, run: jupyter notebook"
echo "Device will be: $DEVICE" 