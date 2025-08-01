# Local M2 Mac Setup Guide

Congratulations! Your M2 Mac has been successfully set up to run the mechanistic interpretability assessment.

## ✅ Setup Status

- **PyTorch**: ✅ Installed (version 2.7.1)
- **MPS Acceleration**: ✅ Available (Apple Silicon acceleration)
- **Transformers**: ✅ Installed
- **Visualization Libraries**: ✅ Installed (Matplotlib, Seaborn, Plotly)
- **Jupyter**: ✅ Installed

## 🚀 Quick Start

### 1. Start Environment

```bash
# Activate virtual environment
source venv/bin/activate

# Start Jupyter notebook
jupyter notebook
```

### 2. Access Jupyter

Open browser and visit: `http://localhost:8888`

### 3. Run First Notebook

In Jupyter, open: `notebooks/01_activations_analysis_local.ipynb`

## 📚 Assessment Content

### Core Question Answers

1. **What are Activations?**
   - Output values of neurons after processing input data
   - Capture internal representations of the model
   - Crucial for interpretability

2. **How to find activations for specific tokens?**
   - Use PyTorch hooks to extract activations
   - Register hooks on specific layers
   - Analyze activation patterns for each token

3. **Purpose of Sparse Autoencoders (SAE)?**
   - Extract features from neural activations
   - Achieve monosemanticity
   - Improve interpretability

### Experimental Content

1. **Activation Analysis** (`01_activations_analysis_local.ipynb`)
   - Extract GPT-2 activations
   - Visualize activation heatmaps
   - Compare activation patterns across different tokens

2. **Sparse Autoencoders** (coming soon)
   - Train SAE networks
   - Feature extraction and analysis
   - Monosemanticity research

3. **Bonus: Roleplay Models** (coming soon)
   - Extend to models like Stheno-8B
   - Multi-layer MLP analysis

## 🔧 Technical Details

### M2 Mac Optimization

- **MPS Acceleration**: Use Metal Performance Shaders
- **Memory Optimization**: Suitable for medium-scale models
- **Performance**: 3-5x faster than CPU

### Model Selection

- **GPT-2**: Suitable for beginners and experiments
- **GPT-2 Medium**: More complex analysis
- **Stheno-8B**: Bonus task (requires more memory)

## 📊 Expected Results

### Activation Analysis
- Unique activation patterns for different tokens
- Similar activations for semantically similar words
- Positional effects analysis

### SAE Analysis
- Feature extraction and visualization
- Monosemanticity metrics
- Sparsity analysis

## 🎯 Assessment Submission

### Required Content
1. **Written Report** (PDF/Markdown)
   - Experimental setup and results
   - Analysis and insights
   - Research proposal

2. **Code and Notebooks**
   - All Jupyter notebooks
   - Source code files
   - Experimental results

### Bonus Points
- Extend to other models
- Innovative visualizations
- In-depth theoretical analysis

## 🆘 Troubleshooting

### Common Issues

1. **Model Download Failure**
   ```bash
   # Use mirror or proxy
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. **Insufficient Memory**
   - Use smaller models
   - Reduce batch size
   - Use gradient checkpointing

3. **MPS Errors**
   - Fallback to CPU: `device = "cpu"`
   - Check PyTorch version

## 📈 Performance Benchmarks

Expected performance on M2 Mac:

| Task | Model | Time | Memory |
|------|------|------|------|
| Activation Extraction | GPT-2 | ~2s | ~2GB |
| SAE Training | GPT-2 | ~10min | ~4GB |
| Visualization | - | ~1s | ~1GB |

## 🎉 Start Your Analysis!

Now you can begin your mechanistic interpretability analysis. Remember:

1. **Understand Concepts**: First understand the basic concepts of activations and SAE
2. **Experiment First**: Run many experiments and observe results
3. **Record Discoveries**: Document your observations and insights
4. **Innovate**: Propose your own research ideas

Good luck with your assessment! 🚀 