#!/bin/bash

echo "=========================================="
echo "Version 7 - Git Repository Setup"
echo "=========================================="
echo

# Check if already a git repo
if [ -d ".git" ]; then
    echo "⚠️  Git repository already exists"
    echo "   To start fresh: rm -rf .git"
    exit 1
fi

# Initialize git
echo "[1/5] Initializing Git repository..."
git init
git branch -M main
echo "✅ Git initialized"
echo

# Add all files
echo "[2/5] Adding files to Git..."
git add .
echo "✅ Files staged"
echo

# Show status
echo "[3/5] Repository status:"
git status --short
echo

# Create initial commit
echo "[4/5] Creating initial commit..."
git commit -m "feat: Initial release of CTMS Activity Pattern Analysis v7.0.0

- Experiment 1: Violin plot dimensional analysis (Task p=0.0604)
- Experiment 2: Daily temporal patterns (CN/CI 5h shift)
- Experiment 3: Multi-classifier evaluation (75% accuracy)
- Experiment 4: Medical correlation (r=0.701***)

Features:
- Unified experiment scripts with correct imports
- Comprehensive English documentation (700+ lines)
- Automated setup and validation scripts
- MIT License
- Publication-ready structure

Tech stack: Python, PyTorch, scikit-learn, Ridge Regression
Data: 68 subjects, 22 activity classes, 80+ engineered features

From Version 5 (Exp 1-3) + Version 6 (Exp 4)
"
echo "✅ Initial commit created"
echo

# Show commit
echo "[5/5] Commit details:"
git log --oneline -1
echo

echo "=========================================="
echo "✅ Git repository ready!"
echo
echo "Next steps:"
echo "  1. Create GitHub repository"
echo "  2. Add remote: git remote add origin <your-repo-url>"
echo "  3. Push: git push -u origin main"
echo
echo "  4. Upload model file to GitHub Releases:"
echo "     - File: ctms_model_medium.pth (15.1 MB)"
echo "     - Location: ../../ctms_model_medium.pth"
echo "     - Tag: v7.0.0"
echo
echo "  5. Update README.md with your repo URL"
echo "=========================================="
