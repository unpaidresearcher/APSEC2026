# Data and Knowledge-Graph Construction

## 1. PrimeVul
BASK uses the function-level PrimeVul benchmark (Ding et al., ICSE 2025). Get the
deduplicated, chronologically split JSONL files from the official PrimeVul release and place
them as:

    data/primevul/primevul_train.jsonl     # 175,797 functions
    data/primevul/primevul_valid.jsonl     #  23,948 functions
    data/primevul/primevul_test.jsonl      #  24,788 functions

Each line is a JSON object with at least `func` (source), `target` (0/1), `project` and
`commit_id`. The 469 test pairs used for P-C are formed from `commit_id` (the first vulnerable
and the first benign function of each commit); see `PrimeVulDataset` in `src/primevul_phase1.py`.

PrimeVul is not redistributed here. Please obtain it from its authors under its license.

## 2. Per-project knowledge graphs
Run from `src/`:

```bash
cd src
# (a) one graph per project: typed entities and the 9-relation taxonomy (Table I); ~43 s on one CPU core
python build_primevul_graphs.py \
  --train ../data/primevul/primevul_train.jsonl \
  --valid ../data/primevul/primevul_valid.jsonl \
  --test  ../data/primevul/primevul_test.jsonl \
  --out   ../data/primevul_graphs --min_funcs 10

# (b) 128-d TransH embeddings for every project; ~8.5 h, mostly CPU-bound
python train_primevul_transh.py \
  --graphs ../data/primevul_graphs --out ../data/primevul_graphs \
  --epochs 50 --dim 128 --device cuda
```

The result is 660 project graphs with 619 MB of graphs and embeddings. Of these, 394 MB (the
embedding tables and entity indices) are needed at inference. Each project folder holds
`graph.gpickle`, `entity_to_idx.json` and `entity_embeddings.pt`. Pass the folder to the
trainer with `--graphs ../data/primevul_graphs`.

## 3. Backbone models
Backbones are downloaded from the Hugging Face Hub by the IDs in `BACKBONE_CONFIGS`
(`src/primevul_phase1.py`):
`microsoft/codebert-base`, `microsoft/unixcoder-base`, `bigcode/starcoderbase-3b`,
`bigcode/starcoder2-3b`, `Qwen/Qwen2.5-Coder-3B`, `deepseek-ai/deepseek-coder-6.7b-base`.
StarCoderBase is a gated model, so accept its license on the Hub and log in with
`huggingface-cli login` first.
