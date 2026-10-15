# Additional analyses

These analyses use the trainer, flags and batch sizes of the paper (`src/`) and the checkpoints
trained with `scripts/`. They were run on one NVIDIA RTX PRO 6000 Blackwell with the versions
in `requirements.txt` (see `results/multi_seed/environment.txt`).

| Analysis | Code | Run script | Results |
|---|---|---|---|
| Cost of the bridge | `cost.py` | `run_cost.sh` | `results/cost/` |
| Performance by KG coverage | `coverage.py` | `run_coverage.sh` | `results/coverage/` |
| Vulnerable/patched pairs | `pair_analysis.py` | `run_coverage.sh` | `results/pair_analysis/` |
| KG only vs features only on StarCoderBase-3B | `src/primevul_phase1_auc.py` | `run_starcoder_source_ablation.sh` | `results/starcoder_source_ablation/` |
| Multi-seed encoders with and without the bridge | `src/primevul_phase1_auc.py`, `summarize_seeds.py` | `run_multi_seed.sh` | `results/multi_seed/` |

`common.py` holds the shared paths and the 16 model configurations (`CONFIGS`). Each
configuration points to its checkpoint under `checkpoints/`, in the folder written by the
matching script in `scripts/`.

## Cost of the bridge

All timings were taken on one otherwise idle GPU, with the batch size each model was trained
with. Train is the mean seconds per optimizer step over 200 steps after 20 warm-up steps, with
the GPU-only part in brackets. Infer is milliseconds per function over the 24,788 test
functions. Memory is the peak training memory.

| Backbone | Train s/step, base -> +BASK | Infer ms/function, base -> +BASK | Memory |
|---|---|---|---|
| CodeBERT (full, bs 64) | 0.48 -> 0.80 (0.41 -> 0.42) | 3.0 -> 7.7 | 21.5 GB |
| UniXcoder (full, bs 64) | 0.46 -> 0.76 (0.41 -> 0.41) | 2.9 -> 6.6 | 21.5 GB |
| StarCoderBase-3B (LoRA, bs 64) | 1.79 -> 2.07 (1.72 -> 1.73) | 11.7 -> 15.2 | 80.7 GB |
| StarCoder2-3B (LoRA, bs 32) | 1.04 -> 1.19 (1.00 -> 1.00) | 12.9 -> 17.0 | 52.2 GB |
| Qwen2.5-Coder-3B (LoRA, bs 32) | 1.15 -> 1.32 (1.10 -> 1.11) | 14.3 -> 17.9 | 73.2 GB |
| DeepSeek-Coder-6.7B (LoRA, bs 16) | 1.15 -> 1.21 (1.10 -> 1.10) | 29.7 -> 33.1 | 55.5 GB |

The bridge does not change peak memory (difference at most 0.1 GB) and adds at most 1.2% GPU
time per step. The remaining overhead is CPU-side KG retrieval and feature extraction in the
data loader: 3.6-3.8 ms per function (`retrieval_*.json`). It could be precomputed and cached.

One-off preprocessing:

| Step | Measured | File |
|---|---|---|
| Build the 660 project graphs (702k entities, 4.24M triples) | 43 s on one CPU core, 1.0 GB peak RAM | `kg_build.json` |
| Train the TransH embeddings (128-d, 50 epochs) | 8.5 h wall clock, mostly CPU-bound | `transh.json` |
| Storage of graphs and embeddings | 619 MB in total, 394 MB needed at inference | `storage_*.json` |
| Fine-tune CodeBERT+BASK for 10 epochs | 6.1 h (step time x 2,746 steps x 10 epochs) | `train.json` |

`*_paper_kg` files measure the KG used for the paper's runs; `*_rebuilt_kg` files measure
a fresh rebuild with the same scripts. `train.json` and `infer.json` also cover the encoder
LoRA and frozen configurations.

`infer.json` re-scores every checkpoint and compares the result with its `results.json`
(`recomputed` vs `published`). The fully fine-tuned encoders match exactly. The other
checkpoints are within one vulnerable test function (0.18 VD-S points), except
DeepSeek-Coder-6.7B with the bridge, which re-scores at VD-S 91.44 against the reported 88.89.

## Performance by KG coverage

Each test function is placed in a bucket by how many of its 32 KG slots hold a retrieved
entity rather than `[PAD]` (0, 1-7, 8-15, 16-31, 32), plus a bucket for functions whose project
has no graph. A vulnerable/patched pair goes to the bucket of its vulnerable function.

`results/coverage/buckets.json` gives, per bucket and per configuration:

- AUC;
- VD-S within the bucket;
- FNR at the threshold where the whole test set has 0.5% FPR;
- F1 and P-C.

It covers six backbones with and without the bridge. `coverage_per_function.csv` holds the
coverage of every function. `predictions/` holds the per-function test probabilities of all
12 configurations.

## Vulnerable/patched pairs

`results/pair_analysis/summary.json` compares, for each backbone, the outcome of every test
pair with and without the bridge, at each model's validation-calibrated threshold.

Of the 469 pairs, 91 consist of the same function before and after the fix; in the others, the
two functions only share a commit. `examples.md` and `examples_*.json` list same-function pairs
that the bridge classifies correctly and the baseline does not, the reverse, and pairs that
both get wrong. Each entry shows the code diff, the scores, the retrieved KG entities and the
structural features that change.

## KG only vs features only on StarCoderBase-3B

The StarCoderBase runs of the LLM source ablation (`scripts/run_ablations.sh`) use batch 32,
while its main runs use batch 64. These runs repeat the ablation at batch 64:

| StarCoderBase-3B, batch 64 | VD-S FNR | Source |
|---|---|---|
| no bridge | 92.35 | `scripts/run_llms.sh` (`G0_starcoder_base`) |
| KG only | 92.17 | `experiments/results/starcoder_source_ablation/kgonly` |
| features only | 88.71 | `experiments/results/starcoder_source_ablation/featonly` |
| both (BASK) | 89.07 | `scripts/run_llms.sh` (`G_starcoder_bridge`) |

## Multi-seed encoders with and without the bridge

CodeBERT and UniXcoder are trained with and without the bridge using the same recipe:

- the same loss and sampler;
- epoch selection on validation VD-S;
- a threshold calibrated on validation data.

The seeds are 42, 123, 456 and 123456; the runs are in `results/multi_seed/runs/`.

Mean and sample standard deviation over the four seeds (`results/multi_seed/summary.json`):

| Configuration | VD-S FNR | AUC | F1 | P-C (%) |
|---|---|---|---|---|
| CodeBERT | 89.48 +/- 1.37 | 0.826 +/- 0.016 | 0.212 +/- 0.009 | 13.86 +/- 3.47 |
| CodeBERT + BASK | 89.57 +/- 1.65 | 0.807 +/- 0.020 | 0.219 +/- 0.014 | 13.49 +/- 1.34 |
| UniXcoder | 88.93 +/- 1.00 | 0.811 +/- 0.031 | 0.221 +/- 0.018 | 14.13 +/- 1.57 |
| UniXcoder + BASK | 88.98 +/- 1.33 | 0.799 +/- 0.005 | 0.211 +/- 0.010 | 12.90 +/- 0.93 |

Runs with the same seed are not bit-identical on this GPU, because some CUDA kernels are
non-deterministic.
