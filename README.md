# Natural-Language-Processing

NLP and Large Language Model applications.

## About

This repository contains various NLP and LLM projects including:

### Projects

- **ChatbotArena**: Chatbot evaluation and ranking systems
- **FineTuningMistral**: Fine-tuning experiments with Mistral models
- **MechanisticInterpretability**: Neural network interpretability analysis

## Mechanistic Interpretability Assessment

The `MechanisticInterpretability/` folder contains a comprehensive implementation of mechanistic interpretability analysis focusing on neural network activations and sparse autoencoders.

### Key Features

- **Activation Analysis**: Complete extraction and visualization of neural network activations
- **Sparse Autoencoders (SAE)**: Feature extraction and monosemanticity analysis
- **M2 Mac Optimization**: Metal Performance Shaders (MPS) acceleration support
- **Self-contained Demo**: No network connection required for core functionality

### Quick Start

```bash
cd MechanisticInterpretability
chmod +x setup_local.sh
./setup_local.sh
source venv/bin/activate
jupyter notebook
```

### Core Questions Answered

1. **What are activations?** - Neural network outputs that represent internal states
2. **How to find activations on specific tokens?** - Using PyTorch hooks for extraction
3. **Purpose of Sparse Autoencoders (SAE)?** - Feature extraction and monosemanticity analysis

### Project Structure

```
MechanisticInterpretability/
├── notebooks/02_simple_activations_demo.ipynb  # Main demonstration
├── src/                                        # Source code modules
├── docs/                                       # Documentation
├── requirements.txt                            # Dependencies
└── setup_local.sh                             # Setup script
```

For detailed documentation, see the `MechanisticInterpretability/` folder.

## Resources

- **ChatbotArena**: Advanced chatbot evaluation and ranking
- **FineTuningMistral**: Model fine-tuning experiments
- **MechanisticInterpretability**: Neural network interpretability research

## Contributing

Each project folder contains its own documentation and setup instructions.
