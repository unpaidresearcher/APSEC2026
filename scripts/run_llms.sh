#!/bin/bash
# The four CodeLLMs with and without BASK, LoRA r=16 (Fig. 2). Run from src/.
set -e
TRAIN="--train ../data/primevul/primevul_train.jsonl --valid ../data/primevul/primevul_valid.jsonl --test ../data/primevul/primevul_test.jsonl --graphs ../data/primevul_graphs --device cuda"
run () {  # $1=output folder  $2=batch size  $3=base system  $4=bridge system
  python primevul_phase1_auc.py $TRAIN --out ../checkpoints/$1 --systems $3 \
    --backbone_mode lora --lr 2e-4 --batch_size $2
  python primevul_phase1_auc.py $TRAIN --out ../checkpoints/$1 --systems $4 \
    --backbone_mode lora --lr 2e-4 --batch_size $2 --fixed_len_pad --use_structural
}
run llm_starcoder  64 G0_starcoder_base  G_starcoder_bridge
run llm_starcoder2 32 J0_starcoder2_base J_starcoder2_bridge
run llm_qwen       32 I0_qwen_base       I_qwen_bridge
run llm_deepseek   16 H0_deepseek_base   H_deepseek_bridge
