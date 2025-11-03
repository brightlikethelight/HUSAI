#!/bin/bash
# Wrapper script to run training with OpenMP fix

# Fix OpenMP duplicate library issue on macOS
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Run the training command
python scripts/train_baseline.py "$@"
