#!/bin/bash

echo "🚀 GitHub Repository Setup Script"
echo "=================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git repository not initialized. Please run 'git init' first."
    exit 1
fi

# Check if we have commits
if ! git rev-parse HEAD >/dev/null 2>&1; then
    echo "❌ No commits found. Please commit your files first."
    exit 1
fi

echo "✅ Git repository is ready"
echo ""

# Ask for GitHub repository URL
echo "📝 Please provide your GitHub repository URL:"
echo "   Example: https://github.com/yourusername/mechanistic-interpretability.git"
echo "   Or: git@github.com:yourusername/mechanistic-interpretability.git"
echo ""
read -p "GitHub URL: " github_url

if [ -z "$github_url" ]; then
    echo "❌ No URL provided. Exiting."
    exit 1
fi

# Add remote repository
echo ""
echo "🔗 Adding remote repository..."
git remote add origin "$github_url"

# Verify remote
echo ""
echo "📋 Current remotes:"
git remote -v

# Push to GitHub
echo ""
echo "📤 Pushing to GitHub..."
echo "   This will create the 'main' branch on GitHub"
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Success! Your files have been uploaded to GitHub."
    echo ""
    echo "📁 Repository structure:"
    echo "   - interview-submission/ (main submission folder)"
    echo "   - docs/ (documentation)"
    echo "   - notebooks/ (Jupyter notebooks)"
    echo "   - src/ (source code)"
    echo ""
    echo "🔗 Your repository is available at: $github_url"
    echo ""
    echo "📋 Files ready for interview submission:"
    echo "   ✅ interview-submission/README_EN.md"
    echo "   ✅ interview-submission/notebooks/02_simple_activations_demo.ipynb"
    echo "   ✅ interview-submission/src/ (all Python modules)"
    echo "   ✅ interview-submission/requirements.txt"
    echo "   ✅ interview-submission/setup_local.sh"
    echo "   ✅ interview-submission/test_setup.py"
    echo "   ✅ interview-submission/docs/ (all documentation)"
else
    echo ""
    echo "❌ Failed to push to GitHub. Please check:"
    echo "   1. Your GitHub URL is correct"
    echo "   2. You have write access to the repository"
    echo "   3. Your SSH keys are set up (if using SSH)"
    echo "   4. Your GitHub credentials are configured"
fi 