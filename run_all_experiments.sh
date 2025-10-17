#!/bin/bash

# CTMS Activity Pattern Analysis - Run All Experiments
# Version 7.0.0

set -e  # Exit on error

echo "========================================"
echo "CTMS Analysis - Running All Experiments"
echo "========================================"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "   Recommended: source venv/bin/activate"
    echo ""
fi

# Check if model exists
if [ ! -f "../../../ctms_model_medium.pth" ]; then
    echo "❌ Error: Model file not found"
    echo "   Please download ctms_model_medium.pth"
    exit 1
fi

echo "✅ Model file found"
echo ""

# Create output directory
mkdir -p all_results
echo "📁 Created output directory: all_results/"
echo ""

# Experiment 1: Violin Plots
echo "======================================"
echo "Experiment 1: Violin Plot Analysis"
echo "======================================"
cd exp1_violin
python run_violin.py
cp outputs/*.png ../all_results/exp1_violin.png 2>/dev/null || true
cp outputs/*.csv ../all_results/exp1_statistics.csv 2>/dev/null || true
cd ..
echo "✅ Experiment 1 completed"
echo ""

# Experiment 2: Daily Patterns
echo "======================================"
echo "Experiment 2: Daily Temporal Patterns"
echo "======================================"
cd exp2_daily
python run_daily_pattern.py
cp outputs/*.png ../all_results/exp2_daily.png 2>/dev/null || true
cp outputs/*.npz ../all_results/exp2_data.npz 2>/dev/null || true
cd ..
echo "✅ Experiment 2 completed"
echo ""

# Experiment 3: Classification
echo "======================================"
echo "Experiment 3: Classification Analysis"
echo "======================================"
cd exp3_classification
python run_classification.py
cp outputs/*.png ../all_results/ 2>/dev/null || true
cp outputs/*.csv ../all_results/exp3_results.csv 2>/dev/null || true
cd ..
echo "✅ Experiment 3 completed"
echo ""

# Experiment 4: Clinical Correlation
echo "======================================"
echo "Experiment 4: Clinical Correlation"
echo "======================================"
cd exp4_medical
python run_enhanced.py
python create_correlation_table.py
cp outputs/enhanced_analysis.png ../all_results/exp4_enhanced.png 2>/dev/null || true
cp outputs/table4_enhanced.png ../all_results/exp4_table.png 2>/dev/null || true
cp outputs/enhanced_results.json ../all_results/exp4_results.json 2>/dev/null || true
cd ..
echo "✅ Experiment 4 completed"
echo ""

echo "========================================"
echo "🎉 All Experiments Completed!"
echo "========================================"
echo ""
echo "📊 Results saved to: all_results/"
echo ""
echo "Generated files:"
ls -lh all_results/
echo ""
echo "To view results:"
echo "  - Experiment 1: open all_results/exp1_violin.png"
echo "  - Experiment 2: open all_results/exp2_daily.png"  
echo "  - Experiment 3: open all_results/exp3_*.png"
echo "  - Experiment 4: open all_results/exp4_enhanced.png"
echo ""
echo "✅ Done!"
