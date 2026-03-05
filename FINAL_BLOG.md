# What HUSAI Actually Proved (And What It Did Not)

Date: 2026-03-05

---

Sparse Autoencoders are supposed to crack open neural networks and show us what is inside. The pitch is compelling: train an encoder-decoder pair on a model's internal activations, enforce sparsity, and out come "features" -- interpretable units that correspond to human-understandable concepts. Safety researchers want to use these features to verify that models are not deceptive. Circuit analysts want to trace how information flows through feature-level pathways. The entire mechanistic interpretability research program rests on the assumption that SAE features are real things, not artifacts of random initialization.

HUSAI was built to test that assumption with strict standards.

## The Setup

We trained a 2-layer transformer on modular arithmetic (mod 113) until it achieved 100% accuracy via grokking. Then we trained SAEs on its internal representations -- 5 independent runs per architecture (TopK and ReLU), varying only the random seed. Same data, same hyperparameters, same training duration. The only difference was initialization.

If SAE features reflect genuine structure in the model, different seeds should converge to similar decompositions. If they do not, those features are arbitrary -- one of many equally valid ways to slice up the activation space, with no special claim to correctness.

We measured feature similarity using PWMCC (Pairwise Maximum Cosine Correlation), following the protocol established by Paulo & Belrose (2025) and Song et al. (2025). And we included a control that, surprisingly, prior work had not emphasized: randomly initialized, completely untrained SAEs.

## The Core Finding: PWMCC = Random Baseline

The result was stark. Trained SAEs achieved a cross-seed PWMCC of 0.309. Untrained random SAEs scored 0.300. The difference -- 0.009, less than 3% -- is statistically significant but practically meaningless.

Standard SAE training produces feature representations that are indistinguishable from random initialization in terms of cross-seed consistency.

This is not a failure of training. The SAEs work perfectly well at their stated job: reconstruction error is 4-8x better than random, with explained variance above 0.91. They learn useful representations. They just learn *different* useful representations every time, with no convergence toward any canonical feature basis.

## The Paradox: Great Reconstruction, Zero Stability

This creates a paradox that cuts to the heart of current SAE evaluation practices. Every standard metric says these SAEs are working: low MSE, high explained variance, controlled sparsity. A practitioner evaluating any single SAE would conclude it has discovered meaningful features. But train another one and you get completely different features that reconstruct equally well.

The reconstruction task is fundamentally underconstrained. Many different sparse decompositions achieve the same error, and random initialization breaks symmetry, sending each run to a different equally-valid solution. The sparsity constraint narrows the space but does not uniquely determine it.

This parallels a known phenomenon in sparse coding for computer vision (Olshausen & Field, 1996): many dictionaries can represent natural images with similar fidelity. SAEs inherit the same ambiguity.

## It Is Not About the Architecture

Both TopK (PWMCC = 0.302) and ReLU (PWMCC = 0.300) show identical random-baseline behavior. This is not an architectural artifact. It is a property of the reconstruction objective itself, which admits multiple solutions regardless of how sparsity is enforced.

Paulo & Belrose (2025) found that TopK SAEs were more seed-dependent than ReLU on large language models. We find no difference on algorithmic tasks. The discrepancy likely reflects the richer semantic structure in LLM activations -- features that correspond to interpretable concepts like "mentions of Paris" provide anchors that partially stabilize across seeds. Our modular arithmetic transformer has no such structure (feature correlations with input variables are essentially zero), so there is nothing for different seeds to converge on.

## The Effective Rank Story

We ran a comprehensive parameterization study varying d_sae from 16 to 1024. A clear tradeoff emerged:

- **Underparameterized** (d_sae < effective rank ~80): stability up to 2.87x random, but poor reconstruction.
- **Matched** (d_sae near effective rank): 1.2-1.6x random stability with good reconstruction.
- **Overparameterized** (d_sae >> effective rank): stability collapses to random baseline.

This makes theoretical sense. When the SAE has fewer dimensions than the activation subspace, it is forced to prioritize the most important directions, and different seeds agree on what those are. When it has excess capacity, there are too many ways to tile the space, and seeds diverge.

Cui et al. (2025) formalized the conditions for SAE identifiability -- extreme sparsity of ground truth, sparse SAE activation, and sufficient hidden dimensions. Our setup violates the first condition: our transformer's activations occupy a dense ~80-dimensional subspace (effective rank 80 out of 128). Under these conditions, identifiability theory predicts PWMCC in the 0.25-0.35 range. Our measured 0.309 matches precisely.

## Task Generalization

We validated the finding on a second task: sequence copying (input [a,b,c,SEP], output copy [a,b,c]). SAEs trained on this transformer achieved PWMCC = 0.300, exactly matching random baseline despite perfect reconstruction (explained variance 0.98). The random-baseline phenomenon is not specific to modular arithmetic. It appears to be a general property of SAE training on algorithmic tasks.

## What This Means for Interpretability

If you analyze the features of a single SAE and conclude that "feature 42 detects modular addition by 7," that interpretation is seed-dependent. A different initialization would assign that semantic (if it exists at all) to a different feature, or distribute it across multiple features, or not discover it. The interpretation is a property of that particular run, not of the model being analyzed.

This threatens several pillars of the interpretability research program:

**Circuit analysis** built on SAE features may be analyzing artifacts. If the features change with seeds, the circuits do too.

**Safety verification** using SAEs ("no deception features detected") is unreliable when a different seed might surface features the first run missed.

**Cumulative progress** requires building on prior findings. If each SAE run starts from a different arbitrary decomposition, studies cannot build on each other.

These concerns are strongest for our setting (algorithmic tasks with dense activations). LLMs with richer semantic structure show higher baseline stability (~65% shared features per Paulo & Belrose), though still far below what would be needed for reliable feature-level claims.

## The Reliability-First Approach

HUSAI did not stop at the stability finding. We built a full release pipeline with explicit gates:

1. **Internal gate**: Does the SAE show trained-vs-random improvement?
2. **Stress gates**: Does it survive random-model, transcoder, and out-of-distribution controls?
3. **External gate**: Does it achieve positive deltas on SAEBench and CE-Bench benchmarks?

All three must pass for a release claim. Internal and stress gates pass. External gates do not -- the SAEBench delta is -0.029 (lower confidence bound) and the CE-Bench interpretability delta is -43.7 vs baseline. The strict release verdict is `pass_all=false`.

We report this honestly. The negative external result is the actual finding, not a failure of the project. Most SAE papers would not have surfaced this because they do not gate on external benchmarks with strict positivity thresholds.

## Engineering Reliability

Along the way, we fixed several code-path defects that could distort research conclusions:

- TopK auxiliary loss was not being optimized (silently dropped from the backward pass).
- Training crashed when dataset size was smaller than batch size.
- The official benchmark harness executed commands via `shell=True`, creating a command-injection risk.
- Feature stability statistics crashed on single-model inputs.

These are the kinds of bugs that quietly corrupt results without generating error messages. Finding and fixing them is part of doing reliability-first research.

## Why Negative Results Matter

The interpretability community needs more projects like this -- not because negative results are inherently valuable, but because the alternative is building on unverified assumptions.

If SAE features are unstable, we need to know that before building safety-critical systems on top of them. If external benchmarks do not confirm internal improvements, we need that feedback loop to guide training objective research. If the reconstruction task is underconstrained, we need stability-aware training methods (Song et al. 2025 show 0.80 PWMCC is achievable) before treating features as ground truth.

The gap between 0.30 (random baseline) and 0.80 (demonstrated achievable) is the research frontier. Closing it requires moving beyond standard reconstruction loss toward objectives that explicitly reward reproducible decompositions.

## Follow-Up Experiments (Implemented)

We implemented and ran six follow-up experiments (paper Section 4.11) that probe the boundaries of the random baseline phenomenon:

- **Pythia-70M validation**: The overparameterization pattern replicates on a real language model. TopK SAEs on Pythia-70M show 1.94× random stability at d_sae=256 (underparameterized), decaying to 1.01× at d_sae=4096. ReLU SAEs show even less stability (1.01-1.34×). The effective rank predictor (d_sae/eff_rank > 5 = near random) generalizes across model scales.
- **1-layer vs 2-layer comparison**: Both architectures show PWMCC = random at d_sae=1024, regardless of effective rank (33.5 vs 80.5). Stability peaks near d_sae ≈ 2× eff_rank for both.
- **Subspace stability**: The top-8 decoder subspace is 2.98× more stable than random, even when individual features match random baseline. Subspace-level analyses may be reliable when feature-level ones are not.
- **Contrastive alignment loss**: Adding a differentiable alignment penalty yields only +0.5% PWMCC improvement. Simple contrastive methods are insufficient.
- **Intervention stability**: Activation steering shows consistent behavioral effects across seeds despite low feature-level match, consistent with subspace stability.
- **Dictionary pinning**: Freezing 75% of decoder columns achieves 0.83 PWMCC but at 11× reconstruction cost. The tradeoff is steep.
- **Effective rank predictor**: d_sae/eff_rank > 5 reliably predicts near-random PWMCC across both model architectures.

All experiments, tests, and results are in the repository. Run with: `make run-followups`.

## What Comes Next

The highest-leverage next steps are:

1. **Scale to larger LLMs**: Pythia-70M validated the pattern. Pythia-1B+ and Llama would test whether richer semantic structure changes the story.
2. **Exploit subspace stability**: Develop interpretability methods that operate on the stable subspace rather than individual features.
3. **Stronger stability methods**: Beyond contrastive losses -- alternating optimization, curriculum-based alignment, or architectural constraints.
4. **External-aware training objectives**: coupling SAEBench/CE-Bench signals into the training loss, not just using them as post-hoc gates.

The code, data, and full experimental log are available in the HUSAI repository.

---

**References:**

- Paulo, G., & Belrose, N. (2025). Sparse Autoencoders Trained on the Same Data Learn Different Features. arXiv:2501.16615.
- Song, X., et al. (2025). Feature Consistency in Sparse Autoencoders. arXiv:2505.20254.
- Cui, Y., et al. (2025). On the Theoretical Understanding of Identifiable Sparse Autoencoders and Beyond. arXiv:2506.15963.
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.
