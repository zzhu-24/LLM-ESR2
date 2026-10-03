#!/usr/bin/env bash
set -euo pipefail

# Retrain only the shorter curve in each dataset from step 0. The existing
# longer-model CSVs are left untouched. Each run stops at the longer curve's
# current final optimizer step and records similarity every 100 steps.
gpu_id="${GPU_ID:-0}"
seed="${SEED:-42}"

datasets=(games musical appliances beauty2014 fashion)
models=(llm_adapter_sasrec sasrec llm_adapter_sasrec sasrec sasrec)
target_steps=(50400 29600 1500 65100 9700)
ts_items=(13 9 3 6 2)
max_lens=(200 200 200 200 50)
check_paths=(
    sequence_similarity_llm_adapter
    sequence_similarity_id
    sequence_similarity_llm_adapter
    sequence_similarity_id
    sequence_similarity_id
)

for index in "${!datasets[@]}"; do
    dataset="${datasets[$index]}"
    model="${models[$index]}"
    target_step="${target_steps[$index]}"
    ts_item="${ts_items[$index]}"
    max_len="${max_lens[$index]}"
    check_path="${check_paths[$index]}"

    echo "Retraining ${dataset}/${model} from step 0 to step ${target_step}"
    python3 train_baseline.py \
        --dataset "${dataset}" \
        --model_name "${model}" \
        --hidden_size 64 \
        --train_batch_size 128 \
        --max_len "${max_len}" \
        --gpu_id "${gpu_id}" \
        --num_workers 8 \
        --num_train_epochs 1000 \
        --max_train_steps "${target_step}" \
        --seed "${seed}" \
        --patience 20 \
        --lr 0.001 \
        --l2 0.0001 \
        --trm_num 2 \
        --num_heads 1 \
        --dropout_rate 0.5 \
        --ts_item "${ts_item}" \
        --track_sequence_similarity \
        --sequence_similarity_interval 100 \
        --check_path "${check_path}" \
        --log
done

DATASETS="games musical appliances beauty2014 fashion" \
    bash experiments/plot_sasrec_item_sequence_similarity.bash
