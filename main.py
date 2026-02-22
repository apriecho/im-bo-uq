#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lightweight entry point — runs the complete IM-BO-UQ pipeline.

Reads data.csv from the data/ folder and produces all results
under outputs/.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import CompletePipeline


def main():
    """Run the full pipeline with default settings."""
    print("=" * 70)
    print("IM-BO-UQ: Interval Mining – Bayesian Optimization –")
    print("          Uncertainty Quantification for N-C/PMS Catalysts")
    print("=" * 70)
    print("\nDefault configuration:")
    print("  Data file  : data/sample_data.csv")
    print("  n_trials   : 50")
    print("  Features   : raw only (no constructed features)")
    print("  n_jobs     : -1 (all CPUs)")
    print("=" * 70)

    pipeline = CompletePipeline(
        data_path='data/sample_data.csv',
        use_constructed_features=False,
        n_jobs=-1
    )

    pipeline.run_all(n_trials=50)

    print("\n" + "=" * 70)
    print("[Done] Pipeline completed!")
    print(f"[Output] {Path('outputs').absolute()}")
    print("=" * 70)


if __name__ == '__main__':
    main()
