#!/usr/bin/env bash
set -euo pipefail

# Train matched ID and frozen-LLM+adapter SASRec models, then draw the two-panel
# Top-20 neighbor-similarity curve for each Amazon dataset.
gpu_id="${GPU_ID:-0}"
seed="${SEED:-42}"
topk="${TOPK:-20}"

datasets=(beauty2014 fashion games musical appliances)
ts_items=(6 2 13 9 3)
max_lens=(200 50 200 200 200)

for index in "${!datasets[@]}"; do
    dataset="${datasets[$index]}"
    ts_item="${ts_items[$index]}"
    max_len="${max_lens[$index]}"

    common_args=(
        --dataset "${dataset}"
        --hidden_size 64
        --train_batch_size 128
        --max_len "${max_len}"
        --gpu_id "${gpu_id}"
        --num_workers 8
        --num_train_epochs 200
        --seed "${seed}"
        --patience 20
        --lr 0.001
        --l2 0.0001
        --trm_num 2
        --num_heads 1
        --dropout_rate 0.5
        --ts_item "${ts_item}"
        --log
    )

    python3 train_baseline.py \
        "${common_args[@]}" \
        --model_name sasrec \
        --check_path neighbor_similarity_id

    python3 train_baseline.py \
        "${common_args[@]}" \
        --model_name llm_adapter_sasrec \
        --check_path neighbor_similarity_llm_adapter

    python3 scripts/item_neighbor_similarity.py \
        --dataset "${dataset}" \
        --id_checkpoint "saved/${dataset}/sasrec/neighbor_similarity_id/pytorch_model.bin" \
        --llm_checkpoint "saved/${dataset}/llm_adapter_sasrec/neighbor_similarity_llm_adapter/pytorch_model.bin" \
        --ts_item "${ts_item}" \
        --topk "${topk}"
done
