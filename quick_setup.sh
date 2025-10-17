#!/bin/bash

echo "=========================================="
echo "CTMS Activity Pattern Analysis"
echo "Quick Setup Script"
echo "=========================================="
echo

# Create virtual environment
echo "[1/4] Creating virtual environment..."
python3 -m venv venv
echo "✅ Virtual environment created"
echo

# Activate virtual environment
echo "[2/4] Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo

# Install dependencies
echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn scipy matplotlib seaborn pyyaml
echo "✅ Dependencies installed"
echo

# Validate setup
echo "[4/4] Validating installation..."
python setup_validate.py
echo

echo "=========================================="
echo "Setup complete!"
echo
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run experiments: bash run_all_experiments.sh"
echo "  3. Or run individual: cd exp4_medical && python run_enhanced.py"
echo "=========================================="
