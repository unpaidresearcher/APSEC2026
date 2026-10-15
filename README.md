# BASK: Bridge Attention with Structural and Knowledge information

Code and figure scripts for the APSEC 2026 paper

> **Structure-Aware Vulnerability Detection Without the Cost: A Modular Bridge for Any Code Model**
> Hrishikesh Ethari and Kisub Kim, DGIST, South Korea.
> 33rd Asia-Pacific Software Engineering Conference (APSEC 2026).

BASK injects a per-project software knowledge graph (KG), embedded with TransH, together with
24 security-relevant structural features into an existing code model. It does so through a
single gated bridge-attention layer that adds about 0.16% parameters and leaves the backbone
unchanged. The same module is attached to six backbones, from 125M encoders (CodeBERT,
UniXcoder) to 3B-6.7B code LLMs (StarCoderBase, StarCoder2, Qwen2.5-Coder, DeepSeek-Coder).

## Repository layout

```
src/            model, training and evaluation, KG construction, TransH training
scripts/        commands for every run reported in the paper
experiments/    cost, coverage, pair-level and multi-seed analyses, with their results
figures/        make_figures.py and fonts; regenerates Figs. 2-4 of the paper
data/           how to obtain PrimeVul and build the knowledge graphs
```

## Setup

```bash
conda create -n bask python=3.12 -y && conda activate bask
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130   # match your CUDA
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the experiments in `experiments/` (Python 3.12,
PyTorch 2.13, Transformers 5.15, PEFT 0.18). The hardware was one NVIDIA RTX PRO 6000 Blackwell
(96 GB).

The paper's checkpoints were trained with earlier library versions. When they are re-scored
with the pinned versions (`experiments/results/cost/infer.json`):

- the fully fine-tuned encoders reproduce exactly;
- the other checkpoints reproduce to within one vulnerable test function (0.18 VD-S points),
  except DeepSeek-Coder-6.7B with the bridge (VD-S 91.44 re-scored vs 88.89 reported).

The encoder runs fit on much smaller GPUs. The LLM runs peak at 52-81 GB with the batch sizes
below.

## Data, knowledge graphs and backbones

See [`data/README.md`](data/README.md). In short:

1. Download the PrimeVul function-level split (train/valid/test JSONL) into `data/primevul/`.
2. Build one KG per project with `src/build_primevul_graphs.py`. The 660 graphs take about 43 s
   on one CPU core.
3. Train 128-d TransH embeddings with `src/train_primevul_transh.py`. This takes about 8.5 h,
   mostly on the CPU, and produces 619 MB of KG data (394 MB is needed at inference).
4. Backbones are downloaded from the Hugging Face Hub (`BACKBONE_CONFIGS` in
   `src/primevul_phase1.py`):

| Backbone | Hugging Face ID | Adaptation | Batch size |
|---|---|---|---|
| CodeBERT | `microsoft/codebert-base` | full fine-tuning, lr 2e-5 | 64 |
| UniXcoder | `microsoft/unixcoder-base` | full fine-tuning, lr 2e-5 | 64 |
| StarCoderBase-3B | `bigcode/starcoderbase-3b` | LoRA r=16, lr 2e-4 | 64 |
| StarCoder2-3B | `bigcode/starcoder2-3b` | LoRA r=16, lr 2e-4 | 32 |
| Qwen2.5-Coder-3B | `Qwen/Qwen2.5-Coder-3B` | LoRA r=16, lr 2e-4 | 32 |
| DeepSeek-Coder-6.7B | `deepseek-ai/deepseek-coder-6.7b-base` | LoRA r=16, lr 2e-4 | 16 |

Each backbone uses the same batch size with and without the bridge. Every batch contains two
vulnerable functions, each ranked against its eight hardest benign negatives, so every run sees
16 ranking pairs per step.

## Training and evaluation

Everything runs through `src/primevul_phase1_auc.py`, from the `src/` directory. The seed is set
with the `PRIMEVUL_SEED` environment variable; the default is 123456, the seed used by PrimeVul.

```bash
cd src
python primevul_phase1_auc.py \
  --train  ../data/primevul/primevul_train.jsonl \
  --valid  ../data/primevul/primevul_valid.jsonl \
  --test   ../data/primevul/primevul_test.jsonl \
  --graphs ../data/primevul_graphs \
  --out    ../checkpoints/main \
  --systems D_codebert_bridge,E_unixcoder_bridge \
  --fixed_len_pad --use_structural --device cuda
```

| Flag | Meaning |
|---|---|
| `--systems` | comma-separated systems from `SYSTEMS_CONFIG`, e.g. `A_codebert` (no bridge), `D_codebert_bridge`, `G0_starcoder_base`, `G_starcoder_bridge` |
| `--fixed_len_pad` | fixed retrieval of K=32 KG slots with a learnable `[PAD]` embedding |
| `--use_structural` | add the 24 structural-feature tokens |
| `--structural_only` | ablation: the bridge sees the structural features only |
| `--drop_feature_group {complexity,security,detail}` | ablation: remove one feature group |
| `--backbone_mode {full,lora,frozen}` | how much of the backbone is trained |
| `--lr`, `--batch_size` | see the backbone table above |

`scripts/` contains the exact commands for every run in the paper.

## Additional analyses

`experiments/` contains five further analyses, each with its code, run script and results. See
[`experiments/README.md`](experiments/README.md).

- Measured GPU, CPU and storage cost on all six backbones.
- Performance by KG coverage.
- Vulnerable/patched pairs with and without the bridge.
- KG-only and features-only bridges on StarCoderBase-3B at batch size 64.
- Multi-seed comparison of the encoders with and without the bridge.

## Figures

```bash
python figures/make_figures.py   # writes PDFs to figures/figs/
```

## Citation

```bibtex
@inproceedings{ethari2026bask,
  title     = {Structure-Aware Vulnerability Detection Without the Cost: A Modular Bridge for Any Code Model},
  author    = {Ethari, Hrishikesh and Kim, Kisub},
  booktitle = {Proceedings of the 33rd Asia-Pacific Software Engineering Conference (APSEC)},
  year      = {2026}
}
```

## Contact

Hrishikesh Ethari (hrishikesh.ethari@dgist.ac.kr) and Kisub Kim (corresponding author), DGIST.

## License

MIT, see `LICENSE`. PrimeVul and the backbone models are not redistributed and remain under
their own licenses.
