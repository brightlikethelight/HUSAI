.PHONY: help install install-dev setup test test-cov smoke lint format typecheck clean run-notebooks reproduce-phase4a ablate-core benchmark-slice

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install core dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements-dev.txt
	pre-commit install

setup: ## Full project setup (conda environment + dependencies)
	@echo "Creating conda environment from environment.yml..."
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo ""
	@echo "Environment 'husai' created/updated successfully!"
	@echo "Activate with: conda activate husai"
	@echo "Then run: make install-dev"

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

test-gpu: ## Run GPU-specific tests
	pytest tests/ -v -m gpu

test-fast: ## Run fast tests only (skip slow tests)
	pytest tests/ -v -m "not slow"

smoke: ## Run fail-fast smoke pipeline
	scripts/ci/smoke_pipeline.sh

lint: ## Run all linters
	@echo "Running flake8..."
	flake8 src/ tests/
	@echo "Running isort check..."
	isort --check-only src/ tests/
	@echo "Running black check..."
	black --check src/ tests/

format: ## Auto-format code
	@echo "Running isort..."
	isort src/ tests/
	@echo "Running black..."
	black src/ tests/
	@echo "Code formatted successfully!"

typecheck: ## Run type checking
	mypy src/

quality: lint typecheck test-fast ## Run all quality checks

clean: ## Clean up generated files
	@echo "Cleaning up..."
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	@echo "Cleanup complete!"

clean-data: ## Clean generated data (careful!)
	@echo "⚠️  This will delete all generated data, results, and checkpoints!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/* results/* checkpoints/* wandb/*; \
		echo "Data cleaned!"; \
	fi

run-notebooks: ## Start Jupyter Lab
	jupyter lab --port=8888

# Training commands
train-baseline: ## Train baseline transformer on modular arithmetic
	python -m scripts.training.train_baseline --config configs/examples/baseline_relu.yaml

train-sae-relu: ## Train ReLU SAE
	python -m scripts.training.train_sae --transformer-checkpoint results/transformer_5000ep/transformer_best.pt --config configs/sae/relu_8x.yaml --layer 1

train-sae-topk: ## Train TopK SAE
	python -m scripts.training.train_sae --transformer-checkpoint results/transformer_5000ep/transformer_best.pt --config configs/sae/topk_8x_k32.yaml --layer 1

train-sae-batchtopk: ## Train BatchTopK SAE
	python -m scripts.training.train_sae --transformer-checkpoint results/transformer_5000ep/transformer_best.pt --config configs/examples/batchtopk_32x.yaml --layer 1

train-all: ## Run full experiment sweep (expensive!)
	python -m scripts.experiments.comprehensive_stability_analysis

# Analysis commands
analyze-consistency: ## Run feature consistency analysis
	python -m scripts.analysis.analyze_feature_stability --sae-dir results/saes --architecture topk --output results/analysis/feature_stability.pkl

analyze-geometry: ## Run geometric structure analysis
	python -m scripts.analysis.hungarian_matching_analysis

analyze-circuits: ## Run circuit discovery analysis
	python -m scripts.analysis.analyze_transformer_fourier

# Documentation
docs-serve: ## Serve documentation locally
	mkdocs serve

docs-build: ## Build documentation
	mkdocs build

# Git shortcuts
commit: quality ## Run quality checks before committing
	@echo "✅ Quality checks passed! Ready to commit."
	@echo "Run: git add . && git commit -m 'your message'"

# Environment info
env-info: ## Show environment information
	@echo "Python version:"
	@python --version
	@echo ""
	@echo "PyTorch version and CUDA availability:"
	@python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
	@echo ""
	@echo "Key packages:"
	@python -c "import transformer_lens; import sae_lens; import wandb; print(f'TransformerLens: {transformer_lens.__version__}'); print(f'SAELens: {sae_lens.__version__}'); print(f'W&B: {wandb.__version__}')" || echo "Some packages not installed"

# Cloud setup helpers
cloud-setup-aws: ## Display AWS setup instructions
	@echo "AWS GPU Instance Setup:"
	@echo "1. Launch g4dn.xlarge instance (NVIDIA T4 GPU, ~\$$0.50/hr)"
	@echo "2. Use Deep Learning AMI (Ubuntu 20.04)"
	@echo "3. Security group: allow SSH (22), Jupyter (8888)"
	@echo "4. SSH: ssh -i your-key.pem ubuntu@<instance-ip>"
	@echo "5. Clone repo and run: make setup"

cloud-setup-gcp: ## Display GCP setup instructions
	@echo "GCP GPU Instance Setup:"
	@echo "1. Create n1-standard-4 with 1x NVIDIA T4 (~\$$0.35/hr)"
	@echo "2. Use Deep Learning VM image"
	@echo "3. Enable Jupyter in Notebooks API"
	@echo "4. SSH via cloud console or: gcloud compute ssh <instance>"
	@echo "5. Clone repo and run: make setup"

wandb-login: ## Setup W&B authentication
	wandb login

wandb-init: ## Initialize W&B project
	wandb init -p husai -e brightliu

# Quick start for new team members
onboard: ## Quick onboarding for new team members
	@echo "🚀 Welcome to HUSAI!"
	@echo ""
	@echo "Onboarding checklist:"
	@echo "1. Read docs/00-Foundations/mission.md"
	@echo "2. Set up environment: make setup"
	@echo "3. Activate environment: conda activate husai"
	@echo "4. Install dev tools: make install-dev"
	@echo "5. Run tests: make test"
	@echo "6. Check environment: make env-info"
	@echo "7. Start Jupyter: make run-notebooks"
	@echo "8. Pick an issue from GitHub!"
	@echo ""
	@echo "Need help? Email: brightliu@college.harvard.edu"

# Reproduction and benchmark helpers
reproduce-phase4a: ## Run Phase 4a trained-vs-random reproduction
	python scripts/experiments/run_phase4a_reproduction.py

ablate-core: ## Run core k/d_sae ablations with confidence intervals
	python scripts/experiments/run_core_ablations.py --device cpu --epochs 20

benchmark-slice: ## Build SAEBench/CE-Bench-aligned benchmark slice
	python scripts/experiments/run_external_benchmark_slice.py
