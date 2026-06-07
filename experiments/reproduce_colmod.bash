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
    check_path="orig_colmod"
    ;;
  beauty2014)
    ts_user=7
    ts_item=6
    max_len=20
    batch_size=512
    epochs=40
    workers=2
    check_path="orig_colmod"
    ;;
  musical)
    ts_user=8
    ts_item=9
    max_len=20
    batch_size=128
    epochs=100
    workers=2
    check_path="orig_colmod"
    ;;
  appliances)
    ts_user=5
    ts_item=3
    max_len=50
    batch_size=512
    epochs=500
    workers=2
    check_path="orig_colmod"
    ;;
  games)
    ts_user=12
    ts_item=13
    max_len=20
    batch_size=512
    epochs=10
    workers=8
    check_path="orig_colmod"
    ;;
  *)
    echo "Unsupported dataset: ${dataset}" >&2
    exit 1
    ;;
esac

model_name="llmesr_colmod"

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
    --check_path "${check_path}" \
    --patience 20 \
    --ts_user "${ts_user}" \
    --ts_item "${ts_item}" \
    --freeze \
    --log \
    --user_sim_func cl \
    --alpha 0.1 \
    --pair_loss_weight 0.01 \
    --collab_llm_ratio 1.0 \
    --enable_id
done
