#!/bin/bash
# Encoders + BASK with LoRA and with a frozen backbone (Tables II and IV). Run from src/.
set -e
TRAIN="--train ../data/primevul/primevul_train.jsonl --valid ../data/primevul/primevul_valid.jsonl --test ../data/primevul/primevul_test.jsonl --graphs ../data/primevul_graphs --device cuda"
for mode in lora frozen; do
  python primevul_phase1_auc.py $TRAIN --out ../checkpoints/encoders_bask_${mode} \
    --systems D_codebert_bridge,E_unixcoder_bridge --fixed_len_pad --use_structural \
    --backbone_mode $mode --lr 2e-4
done
