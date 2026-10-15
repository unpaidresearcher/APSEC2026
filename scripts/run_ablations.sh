#!/bin/bash
# Knowledge-source and feature-group ablations (Figs. 3 and 4), and the LLM source ablation.
# Run from src/.
set -e
TRAIN="--train ../data/primevul/primevul_train.jsonl --valid ../data/primevul/primevul_valid.jsonl --test ../data/primevul/primevul_test.jsonl --graphs ../data/primevul_graphs --device cuda"
ENC="D_codebert_bridge,E_unixcoder_bridge"
# Fig. 3: KG only, features only (both = run_encoders.sh)
python primevul_phase1_auc.py $TRAIN --out ../checkpoints/ablation_kg_only  --systems $ENC --fixed_len_pad
python primevul_phase1_auc.py $TRAIN --out ../checkpoints/ablation_features_only       --systems $ENC \
  --fixed_len_pad --use_structural --structural_only
# Fig. 4: remove one feature group at a time
for g in complexity security detail; do
  python primevul_phase1_auc.py $TRAIN --out ../checkpoints/ablation_drop_$g --systems $ENC \
    --fixed_len_pad --use_structural --drop_feature_group $g
done
# CodeLLMs: KG only and features only (batch 32 for the 3B models, 16 for DeepSeek)
for sb in "G_starcoder_bridge starcoder 32" "J_starcoder2_bridge starcoder2 32" \
          "I_qwen_bridge qwen 32" "H_deepseek_bridge deepseek 16"; do
  set -- $sb
  python primevul_phase1_auc.py $TRAIN --out ../checkpoints/llm_ablation_$2_kg_only --systems $1 \
    --fixed_len_pad --backbone_mode lora --lr 2e-4 --batch_size $3
  python primevul_phase1_auc.py $TRAIN --out ../checkpoints/llm_ablation_$2_features_only --systems $1 \
    --fixed_len_pad --use_structural --structural_only --backbone_mode lora --lr 2e-4 --batch_size $3
done
