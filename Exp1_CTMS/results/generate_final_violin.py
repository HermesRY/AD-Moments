#!/usr/bin/env python3
"""Reproduce the final Experiment 1 violin plot with the fixed best configuration."""

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    results_dir = Path(__file__).resolve().parent
    exp_dir = results_dir.parent
    config_path = results_dir / "final_config.yaml"

    env = os.environ.copy()
    env["EXP1_CTMS_CONFIG"] = str(config_path)

    subprocess.run([sys.executable, str(exp_dir / "run_exp1_ctms.py")], check=True, env=env, cwd=exp_dir)


if __name__ == "__main__":
    main()
