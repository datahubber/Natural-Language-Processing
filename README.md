# Mechanistic Interpretability Assessment

This repository contains the implementation and analysis for the mechanistic interpretability assessment, focusing on reproducing and extending Jake Ward's monosemanticity research.

## Background

This assessment explores mechanistic interpretability concepts including:
- **Toy Models of Superposition**: Understanding how features are represented in hidden layers
- **Monosemanticity**: The concept of neurons responding to specific, interpretable features
- **Sparse Autoencoders (SAE)**: Auxiliary networks for feature extraction and analysis

## Key Resources

1. [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
2. [Monosemanticity at Home](https://jakeward.substack.com/p/monosemanticity-at-home-my-attempt)
3. [Monosemanticity Reproduction Repository](https://github.com/jnward/monosemanticity-repro)

## Project Structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_activations_analysis.ipynb
│   ├── 02_sae_training.ipynb
│   ├── 03_feature_analysis.ipynb
│   └── 04_stheno_experiment.ipynb
├── src/
│   ├── __init__.py
│   ├── activations.py
│   ├── sae.py
│   ├── feature_extraction.py
│   └── utils.py
├── data/
├── models/
└── results/
```

## Assessment Tasks

### 1. Reproduce Jake's Experiment
- Set up cloud GPU environment (RunPod recommended)
- Navigate versioning challenges
- Successfully run the original experiment

### 2. Understand and Discuss Results
- Analyze activations and their meaning
- Explain SAE purpose and architecture
- Discuss feature extraction results

### 3. Bonus: Extend to Stheno-8B
- Apply techniques to a roleplay model
- Handle multi-layer MLP architecture
- Focus on last layer MLP analysis

## Key Questions to Address

1. **What are activations?** How to find activations on particular tokens for given text?
2. **SAE Purpose:** Why train auxiliary Sparse Autoencoder networks?
3. **SAE Architecture:** Why does SAE hidden layer have higher dimensionality than original network?

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up cloud GPU environment (RunPod):
   - Create account and add credits
   - Launch Jupyter notebook instance
   - Upload project files

3. Run experiments:
   - Start with `01_activations_analysis.ipynb`
   - Progress through notebooks sequentially
   - Document results and insights

## Expected Deliverables

- Written report (PDF/Markdown) with analysis and results
- Code and Jupyter notebooks
- Experimental setup documentation
- Research proposal for extensions

## Notes

- AI tools are encouraged for this assessment
- Up to $50 GPU credits available for RunPod experiments
- Focus on understanding and practical implementation 