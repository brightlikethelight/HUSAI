#!/usr/bin/env python3
"""Statistical comparison of PWMCC between tasks."""

import numpy as np
from scipy import stats

# Data from experiments
copy_task_pwmcc = 0.306
copy_task_std = 0.001
copy_task_n = 10  # 5 SAEs, 10 pairwise comparisons

modular_pwmcc = 0.309
modular_std = 0.023
modular_n = 10  # 5 SAEs, 10 pairwise comparisons

print("="*60)
print("STATISTICAL COMPARISON: COPY VS MODULAR ARITHMETIC")
print("="*60)

print(f"\nCopy Task:")
print(f"  PWMCC: {copy_task_pwmcc:.3f} ± {copy_task_std:.3f}")
print(f"  N: {copy_task_n} pairwise comparisons")

print(f"\nModular Arithmetic:")
print(f"  PWMCC: {modular_pwmcc:.3f} ± {modular_std:.3f}")
print(f"  N: {modular_n} pairwise comparisons")

# Compute difference
diff = abs(copy_task_pwmcc - modular_pwmcc)
print(f"\nAbsolute Difference: {diff:.3f}")

# Two-sample t-test (assuming normal distribution)
# Standard error of difference
se_diff = np.sqrt((copy_task_std**2 / copy_task_n) + (modular_std**2 / modular_n))
print(f"Standard Error of Difference: {se_diff:.4f}")

# Z-score (for large samples, use z-test)
z_score = diff / se_diff
print(f"Z-score: {z_score:.2f}")

# P-value (two-tailed)
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
print(f"P-value (two-tailed): {p_value:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((copy_task_std**2 + modular_std**2) / 2)
cohens_d = diff / pooled_std
print(f"Cohen's d (effect size): {cohens_d:.2f}")

# 95% confidence interval for the difference
ci_95 = 1.96 * se_diff
print(f"\n95% Confidence Interval for Difference:")
print(f"  [{diff - ci_95:.4f}, {diff + ci_95:.4f}]")

# Interpretation
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if p_value < 0.05:
    print(f"✅ STATISTICALLY SIGNIFICANT DIFFERENCE (p = {p_value:.4f} < 0.05)")
    print(f"   The two tasks have different PWMCC values.")
else:
    print(f"❌ NOT STATISTICALLY SIGNIFICANT (p = {p_value:.4f} > 0.05)")
    print(f"   Cannot reject null hypothesis - tasks may have similar PWMCC.")

if cohens_d < 0.2:
    effect = "NEGLIGIBLE"
elif cohens_d < 0.5:
    effect = "SMALL"
elif cohens_d < 0.8:
    effect = "MEDIUM"
else:
    effect = "LARGE"

print(f"\nEffect Size: {effect} (Cohen's d = {cohens_d:.2f})")

# Practical significance
print("\n" + "="*60)
print("PRACTICAL SIGNIFICANCE")
print("="*60)

baseline = 0.30
threshold = 0.70

both_near_baseline = (abs(copy_task_pwmcc - baseline) < 0.05) and (abs(modular_pwmcc - baseline) < 0.05)
both_below_threshold = (copy_task_pwmcc < threshold) and (modular_pwmcc < threshold)

if both_near_baseline:
    print(f"🔴 CRITICAL: Both tasks near baseline ({baseline:.2f})")
    print(f"   Copy task: {copy_task_pwmcc:.3f} (within 0.05)")
    print(f"   Modular: {modular_pwmcc:.3f} (within 0.05)")
    print(f"   This suggests UNIVERSAL instability across tasks!")

if both_below_threshold:
    print(f"\n⚠️  WARNING: Both tasks below stability threshold ({threshold:.2f})")
    print(f"   This indicates a REPRODUCIBILITY CRISIS in SAE training.")
    print(f"   Different seeds learn fundamentally different features.")

# Final conclusion
print("\n" + "="*60)
print("FINAL CONCLUSION")
print("="*60)

if not (p_value < 0.05) and both_near_baseline:
    print("✅ GENERALIZATION CONFIRMED:")
    print("   1. No significant difference between tasks (p > 0.05)")
    print("   2. Both tasks show PWMCC ≈ 0.30")
    print("   3. The finding is UNIVERSAL, not task-specific")
    print("   4. SAEs show consistent instability across different tasks")
elif p_value < 0.05:
    print("❌ TASK-SPECIFIC FINDING:")
    print("   1. Significant difference between tasks (p < 0.05)")
    print("   2. The PWMCC baseline may be task-dependent")
    print("   3. Different tasks may have different stability properties")
else:
    print("🤔 INCONCLUSIVE:")
    print("   1. No significant difference statistically")
    print("   2. But values not both near baseline")
    print("   3. Need more investigation")

print("\n" + "="*60)
