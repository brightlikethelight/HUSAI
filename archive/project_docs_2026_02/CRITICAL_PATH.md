# Critical Path Code (Top 10 Files)

These files are the true implementation path for the research claim "SAE feature stability across seeds".

1. `src/data/modular_arithmetic.py`
- Generates modular arithmetic data and dataloaders.
- Defines sequence tokenization and vocabulary semantics used by downstream training.

2. `src/utils/config.py`
- Pydantic schemas and cross-component validation.
- Central contract for dataset/transformer/SAE dimensional compatibility.

3. `src/models/transformer.py`
- TransformerLens wrapper, activation hook extraction, checkpoint load/save.
- Source of model activations used for SAE training.

4. `scripts/training/train_baseline.py`
- Main baseline transformer training entrypoint.
- Produces checkpoints consumed by activation extraction and SAE training.

5. `scripts/analysis/extract_activations.py`
- Extracts target activations by layer and token position.
- Creates the activation tensors used as SAE training data.

6. `src/models/sae.py`
- SAELens-based SAE creation and wrapper API.
- Currently contains version-drift breakage against installed SAELens API.

7. `src/training/train_sae.py`
- Core SAE optimization loop (losses, decoder normalization, dead neuron tracking, checkpointing).
- Contains path/shape/API assumptions that currently break execution.

8. `scripts/training/train_sae.py`
- Intended full SAE pipeline CLI (extract activations, create SAE, train, save summary).
- Currently broken due import path mismatch.

9. `src/analysis/feature_matching.py`
- Implements PWMCC and overlap matrices for seed stability evaluation.
- This is the core metric backing the paper-style claims.

10. `scripts/analysis/analyze_feature_stability.py`
- Orchestrates loading multiple SAE checkpoints and computing stability summaries/plots.
- Primary analysis entrypoint for cross-seed and architecture comparisons.

## Why this is the critical path
- If files 1-5 fail, data-to-activation pipeline is broken.
- If files 6-8 fail, SAE training cannot run reliably.
- If files 9-10 are wrong, research conclusions are not trustworthy.
