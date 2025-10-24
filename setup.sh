#!/bin/bash
# HUSAI Project Setup Script
# Run with: bash setup.sh

set -e  # Exit on error

echo "🚀 HUSAI Project Setup"
echo "======================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}⚠️  conda not found. Please install miniconda/anaconda first.${NC}"
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${BLUE}📦 Step 1: Creating conda environment...${NC}"
if conda env list | grep -q "^husai "; then
    echo "Environment 'husai' already exists. Updating..."
    conda env update -f environment.yml --prune
else
    echo "Creating new environment 'husai'..."
    conda env create -f environment.yml
fi

echo ""
echo -e "${BLUE}📦 Step 2: Activating environment...${NC}"
echo "To activate the environment, run:"
echo -e "${GREEN}conda activate husai${NC}"
echo ""

# Check if we're in a conda environment
if [[ "$CONDA_DEFAULT_ENV" == "husai" ]]; then
    echo -e "${BLUE}📦 Step 3: Installing development tools...${NC}"
    pip install -r requirements-dev.txt

    echo ""
    echo -e "${BLUE}🔧 Step 4: Setting up pre-commit hooks...${NC}"
    pre-commit install
    echo "Pre-commit hooks installed!"

    echo ""
    echo -e "${BLUE}✅ Step 5: Verifying installation...${NC}"

    echo "Python version:"
    python --version

    echo ""
    echo "PyTorch and CUDA:"
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

    echo ""
    echo "Key packages:"
    python -c "import transformer_lens; import sae_lens; print(f'✓ TransformerLens: {transformer_lens.__version__}'); print(f'✓ SAELens: {sae_lens.__version__}')"

    echo ""
    echo -e "${GREEN}✅ Setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Read docs/00-Foundations/mission.md"
    echo "2. Run tests: make test"
    echo "3. Start Jupyter: make run-notebooks"
    echo "4. Check environment: make env-info"
    echo ""
    echo "For W&B setup:"
    echo "  wandb login"
    echo ""
    echo "Happy researching! 🎯"

else
    echo ""
    echo -e "${YELLOW}⚠️  Environment not activated. Please run:${NC}"
    echo -e "${GREEN}conda activate husai${NC}"
    echo ""
    echo "Then complete setup with:"
    echo "  pip install -r requirements-dev.txt"
    echo "  pre-commit install"
fi
