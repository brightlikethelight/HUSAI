# Comprehensive SAE Training Guide for HUSAI
**Date:** November 3, 2025

## Quick Reference

### What You Need to Know

**SAEs (Sparse Autoencoders)** decompose neural network activations into interpretable features by projecting to a larger, sparse space. Think: 128-dim → 1024-dim with only ~32 active features.

### Architecture Comparison

| Architecture | Sparsity Control | Best For | Dead Neurons |
|--------------|------------------|----------|--------------|
| **ReLU** | L1 penalty (indirect) | Baselines, research | High (~20%) |
| **TopK** | Explicit (k active) | Production, stability | Low (~5%) |
| **BatchTopK** | Batch-level | Research comparisons | Medium (~10%) |
| **JumpReLU** | Learnable threshold | SOTA performance | Very low (~3%) |

**Recommendation:** Start with **TopK** - used by Anthropic, Google, OpenAI.

### Key Hyperparameters

```python
# For your modular arithmetic (d_model=128):
expansion_factor = 8  # 128 → 1024 features
k = 32  # TopK sparsity (3.1%)
learning_rate = 3e-4
batch_size = 4096  # tokens
warmup_steps = 10000
```

## Existing Repos & Resources

### 1. SAELens (Your Current Tool) ✅
- **Use this!** Industry standard
- HuggingFace integration
- TransformerLens compatibility
- `pip install sae-lens`

### 2. Pre-trained SAEs on HuggingFace

**Gemma Scope** (Google): 400+ SAEs
```python
from sae_lens import SAE
sae = SAE.from_pretrained("google/gemma-scope-2b-pt-res")
```

**Llama Scope** (OpenMOSS): 256 SAEs on Llama-3.1-8B
```python
# https://huggingface.co/fnlp/Llama-Scope
# 32K and 128K feature SAEs, every layer
```

**Llama 3.2 SAE** (PaulPauls): Educational implementation
```python
# https://github.com/PaulPauls/llama3_interpretability_sae
# Pure PyTorch, detailed training analysis
```

## Training on Modern LLMs

### Llama 3.1/3.2 Best Practices

**From Llama Scope paper (Oct 2024):**
- TopK architecture with k=32-128
- Expansion factor: 16-32× for 8B models
- Data: 4-8B tokens from diverse sources
- Training: 5-7 days on 8× A100 GPUs
- Batch size: 8192 tokens
- Learning rate: 3e-4 with warmup

**Budget version (for you):**
- Expansion: 8×
- Data: 1B tokens, multi-epoch (5 epochs)
- Hardware: 2-4× RTX 4090 or 1× A6000
- Time: 2-3 days

### DeepSeek / Qwen Considerations

**DeepSeek (MoE):** Complex routing, train on shared layers first
**Qwen 2.5:** Standard transformer, same as Llama training

### Together API for Activation Extraction

```python
from together import Together
client = Together(api_key="your-key")

# Some models support hidden state extraction
response = client.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    prompt=prompt,
    return_hidden_states=True,  # Check if supported
    hidden_layer_ids=[12]
)
```

**Note:** Verify Together API supports hidden state extraction for your target model.

## Critical Implementation Details

### 1. Decoder Weight Normalization (REQUIRED!)
```python
# After EVERY training step:
def normalize_decoder_weights(self):
    self.decoder.weight.data = F.normalize(
        self.decoder.weight.data, dim=1
    )
```

### 2. TopK Auxiliary Loss
```python
# For dead neuron revival:
error = (activations - reconstructed).abs()
pre_act = sae.encode_pre_activation(activations)
aux_loss = (error * pre_act.abs()).topk(k)[0].sum()
total_loss = mse_loss + (1/32) * aux_loss
```

### 3. Initialization
```python
# Encoder/Decoder: Orthogonal
nn.init.orthogonal_(encoder.weight)
decoder.weight.data = encoder.weight.data.T

# Normalize decoder
decoder.weight.data = F.normalize(decoder.weight.data, dim=1)
```

## Implementation Strategy for HUSAI

### Phase 1: Modular Arithmetic (Weeks 2-4)
**Goal:** Validate SAE stability on ground-truth task

```bash
# Train baseline transformer
./run_training.sh --config configs/examples/baseline_relu.yaml \
  --epochs 5000 --batch-size 256

# Train SAEs with different seeds
for seed in 42 123 456 789 1011; do
  python -m scripts.training.train_sae --seed $seed --layer 1
done

# Analyze feature overlap
python scripts/analyze_feature_overlap.py
```

**Expected time:** 1-2 hours per SAE on single GPU

### Phase 2: Architecture Comparison (Weeks 4-6)
```python
architectures = ["relu", "topk", "batchtopk"]
for arch in architectures:
    for seed in [42, 123, 456]:
        train_sae(architecture=arch, seed=seed)
```

### Phase 3: Scale to GPT-2 (Weeks 7-10)
```python
# Validate methods on real language
model = AutoModelForCausalLM.from_pretrained("gpt2")
train_sae_on_gpt2(model, layer=6, config=config)
```

## Evaluation Metrics

```python
# 1. Reconstruction
explained_variance = 1 - (mse / original.var())  # Target: >0.90

# 2. Sparsity
l0 = (latents != 0).float().sum(dim=-1).mean()  # Target: 10-50

# 3. Feature Health
dead_ratio = (latents.sum(dim=0) == 0).float().mean()  # Target: <0.10

# 4. Ground Truth (Your advantage!)
fourier_recovery_mcc = compare_features_to_fourier_basis(sae, ground_truth)
```

## Pro Tips

1. **Start small, iterate fast**: 10min validation runs before full training
2. **Log everything to W&B**: Track dead neurons, L0, explained variance
3. **Use pre-trained SAEs as sanity checks**: Compare your metrics
4. **Checkpoint frequently**: Every 1000 steps initially
5. **Master modular arithmetic first**: Don't jump to Llama immediately

## Common Pitfalls

❌ Forgetting decoder normalization → unstable training
❌ Wrong L1 coefficient → either too sparse or too dense
❌ No warmup → dead neurons never recover
❌ Batch size too small → noisy gradients
❌ Training on full LLM immediately → hard to debug

## Resources

**Papers:**
- Scaling Monosemanticity (Anthropic, 2024)
- Gemma Scope (Google, 2024)
- Llama Scope (OpenMOSS, 2024)

**Code:**
- SAELens: https://github.com/jbloomAus/SAELens
- Llama 3.2 SAE: https://github.com/PaulPauls/llama3_interpretability_sae

**Interactive:**
- Neuronpedia: https://neuronpedia.org
- Gemma Scope Explorer: https://neuronpedia.org/gemma-scope

**Your advantage:** Ground truth Fourier circuits for validation!
