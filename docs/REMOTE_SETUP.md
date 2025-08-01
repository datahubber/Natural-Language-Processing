# Remote GPU Server Setup Guide

This guide will help you set up and run the mechanistic interpretability assessment on the remote GPU server.

## Server Information

- **Server**: a6000.ecailab.org
- **Username**: xqiu7
- **Password**: 2427696419
- **Environment**: Kaggle (conda environment)

## Step 1: Initial SSH Connection

First, connect to the remote server:

```bash
ssh xqiu7@a6000.ecailab.org
```

When prompted, enter the password: `2427696419`

## Step 2: Activate Environment and Start Jupyter

Once connected to the server, activate the Kaggle environment and start Jupyter:

```bash
# Activate the conda environment
. ~/Kaggle/bin/activate

# Start Jupyter notebook server
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 --allow-root
```

**Important**: Keep this terminal session open! The Jupyter server needs to keep running.

## Step 3: Create SSH Tunnel (New Terminal)

Open a **new terminal window** on your local machine and create an SSH tunnel:

```bash
ssh -L 8888:localhost:8888 xqiu7@a6000.ecailab.org
```

This creates a tunnel from your local port 8888 to the remote port 8888.

## Step 4: Upload Project Files

In another terminal, upload the project files to the server:

```bash
# Create remote directory
ssh xqiu7@a6000.ecailab.org "mkdir -p ~/mechanistic_interpretability"

# Upload files (from your local project directory)
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    ./ xqiu7@a6000.ecailab.org:~/mechanistic_interpretability/
```

## Step 5: Access Jupyter Notebook

1. Open your web browser
2. Go to: `http://localhost:8888`
3. Look for the token in the remote server output (it will look like: `?token=abc123...`)
4. Enter the token to access Jupyter

## Step 6: Navigate to Project Directory

In Jupyter:
1. Navigate to `~/mechanistic_interpretability`
2. Open the notebooks in order:
   - `01_activations_analysis.ipynb`
   - `02_sae_training.ipynb`
   - `03_feature_analysis.ipynb`
   - `04_stheno_experiment.ipynb`

## Step 7: Install Dependencies

In the first notebook cell, install the required packages:

```python
!pip install torch transformers datasets accelerate einops sentencepiece tokenizers
!pip install numpy pandas matplotlib seaborn plotly jupyter ipywidgets tqdm
!pip install requests huggingface-hub scikit-learn scipy
```

## Alternative: Using the Setup Scripts

You can also use the provided setup scripts:

### Option 1: Manual Setup
```bash
# Terminal 1: Start remote server
./setup_remote.sh

# Terminal 2: Create tunnel
./tunnel.sh

# Terminal 3: Upload files
./upload_to_server.sh
```

### Option 2: Automated Setup
```bash
# Make scripts executable
chmod +x setup_remote.sh tunnel.sh upload_to_server.sh

# Run setup
./setup_remote.sh
```

## Troubleshooting

### Connection Issues
- Make sure you're using the correct server address
- Check if the port 8888 is available
- Try a different port if needed: `--port=8889`

### Environment Issues
- If the Kaggle environment doesn't exist, create it:
  ```bash
  conda create -n Kaggle python=3.8
  conda activate Kaggle
  ```

### Jupyter Issues
- If Jupyter doesn't start, try:
  ```bash
  jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token=''
  ```

### File Upload Issues
- If rsync is not available, use scp:
  ```bash
  scp -r ./ xqiu7@a6000.ecailab.org:~/mechanistic_interpretability/
  ```

## Running the Assessment

1. **Start with Notebook 01**: Understand activations
2. **Progress through notebooks**: Each builds on the previous
3. **Save results**: Use the provided save functions
4. **Document findings**: Take notes on your observations

## Assessment Tasks

### Task 1: Reproduce Jake's Experiment
- Run the activation analysis
- Train the SAE network
- Analyze feature extraction results

### Task 2: Understand and Discuss
- Answer the key questions about activations and SAEs
- Analyze your experimental results
- Prepare discussion points

### Task 3: Bonus - Stheno-8B Extension
- Apply techniques to a roleplay model
- Handle multi-layer architecture
- Compare results with original experiment

## Expected Deliverables

1. **Written Report**: Analysis and findings
2. **Code/Notebooks**: All modified notebooks
3. **Results**: Saved plots and data
4. **Documentation**: Setup and experimental notes

## Tips for Success

- **Start early**: GPU training can take time
- **Save frequently**: Use the provided save functions
- **Document everything**: Take notes on your process
- **Ask questions**: Use the educational functions in the code
- **Experiment**: Try different parameters and approaches

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the error messages carefully
3. Try restarting the Jupyter server
4. Contact your instructor if needed

Good luck with your assessment! 