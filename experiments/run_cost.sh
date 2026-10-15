#!/bin/bash
# Cost of BASK on one otherwise idle GPU. Run from the repository root.
set -e
OUT=experiments/results/cost
NEW=data/primevul_graphs_rebuild
python experiments/cost.py storage   --out $OUT --tag paper_kg
python experiments/cost.py retrieval --out $OUT --tag paper_kg
python experiments/cost.py kg_build  --out $OUT --graphs $NEW
python experiments/cost.py transh    --out $OUT --graphs $NEW
python experiments/cost.py storage   --out $OUT --graphs $NEW --tag rebuilt_kg
python experiments/cost.py retrieval --out $OUT --graphs $NEW --tag rebuilt_kg
python experiments/cost.py train     --out $OUT
python experiments/cost.py infer     --out $OUT
