#!/usr/bin/env bash
set -euo pipefail

gpu_id="${GPU_ID:-0}"
dataset="${1:-fashion}"
seed_list=(${SEEDS:-42})

case "${dataset}" in
  fashion)
    ts_user=3
    ts_item=2
    max_len=50
    batch_size=512
    epochs=40
    workers=8
    ;;
  beauty2014)
    ts_user=7
    ts_item=6
    max_len=20
    batch_size=512
    epochs=40
    workers=2
    ;;
  musical)
    ts_user=8
    ts_item=9
    max_len=20
    batch_size=128
    epochs=100
    workers=2
    ;;
  appliances)
    ts_user=5
    ts_item=3
    max_len=50
    batch_size=512
    epochs=500
    workers=2
    ;;
  games)
    ts_user=12
    ts_item=13
    max_len=20
    batch_size=512
    epochs=10
    workers=8
    ;;
  *)
    echo "Unsupported dataset: ${dataset}" >&2
    exit 1
    ;;
esac

model_name="llmesr_intent_colmod"

for seed in "${seed_list[@]}"
do
  python3 -u main.py --dataset "${dataset}" \
    --model_name "${model_name}" \
    --hidden_size 64 \
    --train_batch_size "${batch_size}" \
    --max_len "${max_len}" \
    --gpu_id "${gpu_id}" \
    --num_workers "${workers}" \
    --num_train_epochs "${epochs}" \
    --seed "${seed}" \
    --check_path "efficient_compress" \
    --patience 20 \
    --ts_user "${ts_user}" \
    --ts_item "${ts_item}" \
    --freeze \
    --log \
    --user_sim_func cl \
    --alpha 0.1 \
    --pair_loss_weight 0.01 \
    --collab_llm_ratio 1.0 \
    --semantic_filter_weight 0.5 \
    --semantic_graph_threshold 0.1 \
    --intent_gate_dropout 0.1 \
    --hgc_layers 2 \
    --enable_id \
    --use_adapter \
    --adapter_type mlp
done
