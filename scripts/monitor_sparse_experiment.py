#!/usr/bin/env python3
"""Monitor progress of sparse ground truth experiment."""

import json
from pathlib import Path
import time

output_dir = Path("results/sparse_ground_truth")

print("Monitoring sparse ground truth experiment...")
print("="*80)

while True:
    # Check transformer checkpoints
    transformer_dir = output_dir / "transformer"
    if transformer_dir.exists():
        checkpoints = list(transformer_dir.glob("*.pt"))
        print(f"\n[{time.strftime('%H:%M:%S')}] Transformer checkpoints: {len(checkpoints)}")
        for cp in sorted(checkpoints):
            size_kb = cp.stat().st_size / 1024
            print(f"  {cp.name}: {size_kb:.1f} KB")

    # Check SAE checkpoints
    sae_files = list(output_dir.glob("sae_seed_*.pt"))
    if sae_files:
        print(f"\n[{time.strftime('%H:%M:%S')}] SAE checkpoints: {len(sae_files)}")
        for sae in sorted(sae_files):
            size_kb = sae.stat().st_size / 1024
            print(f"  {sae.name}: {size_kb:.1f} KB")

    # Check fourier validation
    fourier_file = output_dir / "fourier_validation.json"
    if fourier_file.exists():
        with open(fourier_file) as f:
            fourier_data = json.load(f)
        print(f"\n[{time.strftime('%H:%M:%S')}] Fourier Validation:")
        print(f"  Status: {fourier_data.get('validation_status', 'unknown')}")
        print(f"  Primary R²: {fourier_data.get('primary_r2', 0):.4f}")

    # Check final results
    results_file = output_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        print(f"\n[{time.strftime('%H:%M:%S')}] ✅ EXPERIMENT COMPLETE!")
        print(f"  Sparse PWMCC: {results['pwmcc_results']['mean']:.4f}")
        print(f"  Validation: {results['validation_result']}")
        break

    print("\n" + "-"*80)
    time.sleep(30)  # Check every 30 seconds

print("\nExperiment monitoring complete.")
