#!/usr/bin/env python3
"""
Run all Exp_Full experiments and validate results.

This script executes all four experiments for both strategies:
- Without Personalization
- With Personalization

Usage:
    python run_all_experiments.py
"""

import subprocess
import sys
from pathlib import Path
import time

SCRIPT_DIR = Path(__file__).parent
WITHOUT_DIR = SCRIPT_DIR / 'Without_Personalization'
WITH_DIR = SCRIPT_DIR / 'With_Personalization'

experiments = [
    ('Exp1: CTMS Embedding', 'exp1_embedding_viz.py'),
    ('Exp2: Daily Patterns', 'exp2_daily_patterns.py'),
    ('Exp3: Classification', 'exp3_classification.py'),
    ('Exp4: Medical Correlations', 'exp4_medical_correlations.py'),
]

def run_experiment(name, script_path):
    """Run a single experiment script."""
    print("\n" + "="*80)
    print(f"Running: {name}")
    print(f"Script: {script_path}")
    print("="*80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ {name} completed successfully ({elapsed:.1f}s)")
            return True
        else:
            print(f"✗ {name} failed!")
            print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {name} timed out after 1 hour!")
        return False
    except Exception as e:
        print(f"✗ {name} failed with error: {e}")
        return False


def main():
    print("="*80)
    print("RUNNING ALL EXP_FULL EXPERIMENTS")
    print("="*80)
    
    results = {
        'Without Personalization': {},
        'With Personalization': {}
    }
    
    # Run Without Personalization experiments
    print("\n" + "#"*80)
    print("# WITHOUT PERSONALIZATION")
    print("#"*80)
    
    for exp_name, script_name in experiments:
        script_path = WITHOUT_DIR / script_name
        if script_path.exists():
            success = run_experiment(exp_name, script_path)
            results['Without Personalization'][exp_name] = success
        else:
            print(f"⚠ Skipping {exp_name}: {script_path} not found")
            results['Without Personalization'][exp_name] = None
    
    # Run With Personalization experiments
    print("\n" + "#"*80)
    print("# WITH PERSONALIZATION")
    print("#"*80)
    
    for exp_name, script_name in experiments:
        script_path = WITH_DIR / script_name
        if script_path.exists():
            success = run_experiment(exp_name, script_path)
            results['With Personalization'][exp_name] = success
        else:
            print(f"⚠ Skipping {exp_name}: {script_path} not found")
            results['With Personalization'][exp_name] = None
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for strategy in results:
        print(f"\n{strategy}:")
        for exp_name, success in results[strategy].items():
            if success is None:
                status = "⚠ Not found"
            elif success:
                status = "✓ Success"
            else:
                status = "✗ Failed"
            print(f"  {status}: {exp_name}")
    
    # Overall status
    total_ran = sum(1 for s in results.values() for r in s.values() if r is not None)
    total_success = sum(1 for s in results.values() for r in s.values() if r is True)
    total_failed = sum(1 for s in results.values() for r in s.values() if r is False)
    
    print(f"\nOverall: {total_success}/{total_ran} succeeded, {total_failed} failed")
    
    if total_failed == 0 and total_ran > 0:
        print("\n🎉 All experiments completed successfully!")
        return 0
    else:
        print("\n⚠ Some experiments had issues. Check logs above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
