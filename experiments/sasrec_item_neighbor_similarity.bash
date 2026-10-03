#!/usr/bin/env bash
set -euo pipefail

# Train matched ID and frozen-LLM+adapter SASRec models, then draw the two-panel
# current-item-to-history similarity curve for each Amazon dataset.
gpu_id="${GPU_ID:-0}"
seed="${SEED:-42}"

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
        --track_sequence_similarity
        --sequence_similarity_interval 100
        --log
    )

    python3 train_baseline.py \
        "${common_args[@]}" \
        --model_name sasrec \
        --check_path sequence_similarity_id

    python3 train_baseline.py \
        "${common_args[@]}" \
        --model_name llm_adapter_sasrec \
        --check_path sequence_similarity_llm_adapter

    python3 scripts/item_sequence_similarity.py \
        --dataset "${dataset}" \
        --id_trace "outputs/item_sequence_similarity/${dataset}_sasrec_sequence_similarity_training_steps.csv" \
        --llm_trace "outputs/item_sequence_similarity/${dataset}_llm_adapter_sasrec_sequence_similarity_training_steps.csv"
done
