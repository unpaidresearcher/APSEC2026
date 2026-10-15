#!/bin/bash
# Performance by KG coverage, then the pair-level analysis. Run from the repository root.
set -e
OUT=experiments/results/coverage
python experiments/coverage.py coverage --out $OUT
python experiments/coverage.py predict  --out $OUT
python experiments/coverage.py buckets  --out $OUT
python experiments/pair_analysis.py --coverage $OUT --out experiments/results/pair_analysis
