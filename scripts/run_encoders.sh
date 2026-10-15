#!/bin/bash
# CodeBERT and UniXcoder with and without BASK, seed 123456 (Table III).
# Run from src/. Output folders are under checkpoints/.
set -e
TRAIN="--train ../data/primevul/primevul_train.jsonl --valid ../data/primevul/primevul_valid.jsonl --test ../data/primevul/primevul_test.jsonl --graphs ../data/primevul_graphs --device cuda"
python primevul_phase1_auc.py $TRAIN --out ../checkpoints/encoders_base \
  --systems A_codebert,B_unixcoder
python primevul_phase1_auc.py $TRAIN --out ../checkpoints/encoders_bask \
  --systems D_codebert_bridge,E_unixcoder_bridge --fixed_len_pad --use_structural
