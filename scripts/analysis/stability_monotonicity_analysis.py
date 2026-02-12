"""
Critical analysis of stability vs L0 monotonicity claim across architectures.
"""

import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Load data
BASE_DIR = Path(__file__).resolve().parents[2]
data_path = BASE_DIR / "results" / "multi_architecture_stability" / "multi_architecture_results.json"
with open(data_path, "r") as f:
    data = json.load(f)

# Organize by architecture
architectures = {}
for entry in data:
    arch = entry['architecture']
    if arch not in architectures:
        architectures[arch] = []
    architectures[arch].append({
        'l0': entry['l0'],
        'pwmcc': entry['pwmcc_mean'],
        'config': entry['config']
    })

print("="*80)
print("CRITICAL ANALYSIS: Stability vs L0 Monotonicity")
print("="*80)
print()

for arch_name, arch_data in sorted(architectures.items()):
    print(f"\n{'='*80}")
    print(f"ARCHITECTURE: {arch_name}")
    print(f"{'='*80}")

    # Sort by L0
    arch_data_sorted = sorted(arch_data, key=lambda x: x['l0'])

    l0_values = np.array([d['l0'] for d in arch_data_sorted])
    pwmcc_values = np.array([d['pwmcc'] for d in arch_data_sorted])

    print(f"\nData points (n={len(l0_values)}):")
    print(f"{'L0':<12} {'PWMCC':<12} {'Config':<15}")
    print("-" * 40)
    for d in arch_data_sorted:
        print(f"{d['l0']:<12.2f} {d['pwmcc']:<12.6f} {d['config']:<15}")

    # Correlation analysis
    pearson_r, pearson_p = pearsonr(l0_values, pwmcc_values)
    spearman_r, spearman_p = spearmanr(l0_values, pwmcc_values)

    print(f"\n--- Correlation Analysis ---")
    print(f"Pearson correlation:  r = {pearson_r:.4f}, p = {pearson_p:.4e}")
    print(f"Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.4e}")

    # Check monotonicity
    is_strictly_decreasing = all(pwmcc_values[i] > pwmcc_values[i+1] for i in range(len(pwmcc_values)-1))
    is_weakly_decreasing = all(pwmcc_values[i] >= pwmcc_values[i+1] for i in range(len(pwmcc_values)-1))

    print(f"\n--- Monotonicity Check ---")
    print(f"Strictly monotonic decreasing: {is_strictly_decreasing}")
    print(f"Weakly monotonic decreasing:   {is_weakly_decreasing}")

    # Check for violations
    violations = []
    for i in range(len(pwmcc_values)-1):
        if pwmcc_values[i] < pwmcc_values[i+1]:
            violations.append((i, l0_values[i], pwmcc_values[i], l0_values[i+1], pwmcc_values[i+1]))

    if violations:
        print(f"\nMonotonicity VIOLATIONS found ({len(violations)}):")
        for idx, l0_a, pwmcc_a, l0_b, pwmcc_b in violations:
            delta_l0 = l0_b - l0_a
            delta_pwmcc = pwmcc_b - pwmcc_a
            print(f"  L0: {l0_a:.2f} → {l0_b:.2f} (+{delta_l0:.2f})")
            print(f"  PWMCC: {pwmcc_a:.6f} → {pwmcc_b:.6f} (+{delta_pwmcc:.6f})")

    # Compute total change
    total_l0_change = l0_values[-1] - l0_values[0]
    total_pwmcc_change = pwmcc_values[-1] - pwmcc_values[0]
    pct_change = (total_pwmcc_change / pwmcc_values[0]) * 100

    print(f"\n--- Overall Trend ---")
    print(f"L0 range: {l0_values[0]:.2f} → {l0_values[-1]:.2f} (Δ = {total_l0_change:.2f})")
    print(f"PWMCC range: {pwmcc_values[0]:.6f} → {pwmcc_values[-1]:.6f}")
    print(f"PWMCC change: {total_pwmcc_change:.6f} ({pct_change:+.2f}%)")

    # Effect size
    if len(l0_values) > 1:
        l0_std = np.std(l0_values, ddof=1)
        pwmcc_std = np.std(pwmcc_values, ddof=1)
        if l0_std > 0:
            slope = np.cov(l0_values, pwmcc_values)[0, 1] / np.var(l0_values, ddof=1)
            print(f"Slope (PWMCC per unit L0): {slope:.6f}")

    # Statistical significance interpretation
    print(f"\n--- Interpretation ---")
    if pearson_p < 0.001:
        sig = "highly significant (p < 0.001)"
    elif pearson_p < 0.01:
        sig = "significant (p < 0.01)"
    elif pearson_p < 0.05:
        sig = "marginally significant (p < 0.05)"
    else:
        sig = "NOT significant (p ≥ 0.05)"

    if abs(pearson_r) > 0.9:
        strength = "very strong"
    elif abs(pearson_r) > 0.7:
        strength = "strong"
    elif abs(pearson_r) > 0.5:
        strength = "moderate"
    elif abs(pearson_r) > 0.3:
        strength = "weak"
    else:
        strength = "very weak"

    print(f"Correlation is {strength} and {sig}")

    if is_strictly_decreasing:
        print(f"✓ Monotonicity claim SUPPORTED: Strict monotonic decrease")
    elif is_weakly_decreasing:
        print(f"⚠ Monotonicity claim WEAKLY SUPPORTED: Weak monotonic decrease")
    else:
        print(f"✗ Monotonicity claim REJECTED: Non-monotonic relationship")

print("\n" + "="*80)
print("OVERALL ASSESSMENT")
print("="*80)

# Count architectures
total_archs = len(architectures)
strictly_monotonic = 0
weakly_monotonic = 0
non_monotonic = 0

for arch_name, arch_data in architectures.items():
    arch_data_sorted = sorted(arch_data, key=lambda x: x['l0'])
    pwmcc_values = np.array([d['pwmcc'] for d in arch_data_sorted])

    is_strict = all(pwmcc_values[i] > pwmcc_values[i+1] for i in range(len(pwmcc_values)-1))
    is_weak = all(pwmcc_values[i] >= pwmcc_values[i+1] for i in range(len(pwmcc_values)-1))

    if is_strict:
        strictly_monotonic += 1
    elif is_weak:
        weakly_monotonic += 1
    else:
        non_monotonic += 1

print(f"\nTotal architectures analyzed: {total_archs}")
print(f"  Strictly monotonic decreasing: {strictly_monotonic} ({strictly_monotonic/total_archs*100:.1f}%)")
print(f"  Weakly monotonic decreasing:   {weakly_monotonic} ({weakly_monotonic/total_archs*100:.1f}%)")
print(f"  Non-monotonic:                 {non_monotonic} ({non_monotonic/total_archs*100:.1f}%)")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if strictly_monotonic == total_archs:
    print("✓ CLAIM VERIFIED: Stability decreases monotonically with L0 across ALL architectures")
elif strictly_monotonic + weakly_monotonic == total_archs:
    print("⚠ CLAIM MOSTLY ACCURATE: All architectures show monotonic decrease,")
    print("  but some are only weakly monotonic (with flat regions)")
else:
    print("✗ CLAIM OVERSTATED: Not all architectures show monotonic decrease")
    print(f"  {non_monotonic}/{total_archs} architectures violate monotonicity")

print("\n" + "="*80)
print("KEY CAVEATS AND NUANCES")
print("="*80)

# Check data quality issues
print("\n1. DATA QUALITY:")
for arch_name, arch_data in architectures.items():
    if len(arch_data) <= 2:
        print(f"   ⚠ {arch_name}: Only {len(arch_data)} data points - insufficient for robust monotonicity claim")

    # Check for duplicates
    l0_values = [d['l0'] for d in arch_data]
    if len(l0_values) != len(set(l0_values)):
        print(f"   ⚠ {arch_name}: Duplicate L0 values detected")

print("\n2. CORRELATION STRENGTH:")
print("   While relationships may be monotonic, correlation strength varies by architecture")

print("\n3. STATISTICAL SIGNIFICANCE:")
print("   Sample sizes are small - results should be interpreted cautiously")

print("\n" + "="*80)
