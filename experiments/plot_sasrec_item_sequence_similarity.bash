#!/usr/bin/env bash
set -euo pipefail

datasets=${DATASETS:-"beauty2014 fashion games musical appliances"}
output_dir=${OUTPUT_DIR:-"outputs/item_sequence_similarity"}

for dataset in ${datasets}; do
    python3 scripts/item_sequence_similarity.py \
        --dataset "${dataset}" \
        --id_trace "${output_dir}/${dataset}_sasrec_sequence_similarity_training_steps.csv" \
        --llm_trace "${output_dir}/${dataset}_llm_adapter_sasrec_sequence_similarity_training_steps.csv" \
        --output_dir "${output_dir}"
done
