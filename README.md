# Compass Circuits

**Research question:** How does a tiny transformer compute displacement from a sequence of cardinal directions? Are horizontal and vertical coordinates represented in causally separable components?

We train a 2-layer, 4-head, 64-dimensional **causal decoder-only transformer from scratch** on 12-token movement sequences. The only vocabulary is `N`, `E`, `S`, `W`. The starting position is always (0, 0), and each action moves one unit. The model sees only the tokens and must classify the final x and y coordinates. For example, `N N E S E` has target `(2, 1)`.

## Run on an Apple M3 Pro

```bash
# macOS with Python 3.10+; Homebrew Python or an existing interpreter is fine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Check that Apple's MPS GPU backend is available
python -c 'import torch; print("torch:", torch.__version__, "MPS:", torch.backends.mps.is_available())'

# Verify the project
python -m pytest -q

# Train the model; device=auto chooses MPS on a supported Mac
python -m compass_circuits.train --steps 1200 --batch-size 256

# Run initial mechanistic interpretability experiments
python -m compass_circuits.analyze
```

If MPS is unavailable, the scripts automatically use CPU. To force CPU for debugging, add `--device cpu`. Training speed and final accuracy depend on hardware, PyTorch version, and training randomness; inspect the printed validation metrics before interpreting the results. If accuracy is poor, increase `--steps 5000` or try a smaller sequence length (`--seq-len 8`) for a simpler first experiment.

The model has 71,346 parameters for a sequence length of 12 (exact count printed when you run it), so it fits comfortably on an M3 Pro. We deliberately use explicit attention matrices rather than optimized fused attention to keep the attention computation inspectable.

## What the files do

| File | Purpose |
|---|---|
| `compass_circuits/data.py` | Generate unlimited movement sequences with exact x/y labels and counterfactual pairs. |
| `compass_circuits/model.py` | Minimal transformer with named activation hook sites, no pretrained weights. |
| `compass_circuits/train.py` | Online training, held-out random validation batches, checkpoint/metrics. |
| `compass_circuits/analyze.py` | Ridge linear probes and attention/MLP-output activation patching. |
| `tests/test_project.py` | Tests for the task, gradients, hook behavior and intervention pair validity. |

## Initial scientific experiments

### 1. Representation (correlational)

Capture the **last-token residual stream** before and after each layer. Fit a held-out ridge regression probe to predict x/y coordinates. Report test R² separately for both axes. A high R² shows *decodability*, not that the network causally uses those features.

### 2. Component-level causal intervention

Create a matched sequence pair in which **only the first token differs**, from `W` (base) to `E` (donor). The donor has **x + 2** and **exactly the same y**. For each layer separately:

1. Run the donor and cache the attention or MLP output at the last sequence position.
2. Run the base and replace that component's last-token output with the donor's.
3. Record whether its predicted x switches to the donor's x and its predicted y remains the base's y.

We measure effects only for pairs where the unmodified model predicts both base and donor correctly; the report includes the number of qualifying pairs. Partial, failed, or mixed interventions are findings to investigate, not bugs by default. A full-component patch is **not** proof of a dedicated x direction. More precise follow-up experiments would patch specific head outputs, use projection-based interventions, and test multiple seeds and pair-generating schemes.

## Design decisions and limitations

- This task can be solved by **counting token frequencies**, rather than learning a sequential world model. That's intentional: our objective is to reverse-engineer which algorithm is implemented, not to assume one.
- We train on fixed 12-token sequences and evaluate on fresh sequences of the same length. High IID accuracy **doesn't establish** length generalization or compositional generalization.
- Absolute positional embeddings, separate supervised x/y classification heads, and the specific seed can influence whether axes appear separable. We should later vary those conditions.
- Counterfactual pairs differ in one input token, so patching can transfer information about that token as well as aggregate state. A causal patch effect alone does **not** establish a location variable or a dedicated representation.
- First results are a **pilot experiment**, not a publishable discovery. To publish a research claim, replicate seeds, test controls and alternative explanations, and compare with previous counting-mechanism research.

## Artifacts

After training: `runs/compass.pt`, `runs/compass.json`. After analysis: `runs/analysis.json`.

## Next questions

- Where is coordinate information first decodable? Is it linearly represented?
- Do any individual attention heads causally contribute differently to the x and y outputs?
- Does swapping a hypothesized x-subspace move x without perturbing y across many different counterfactuals?
- Is this stable across random seeds, architecture choices and longer-than-trained sequences?
