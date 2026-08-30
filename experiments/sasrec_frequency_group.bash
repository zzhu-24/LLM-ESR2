## SASRec frequency-group performance: histogram + performance lines
##
## Prerequisite: a trained SASRec checkpoint at
## saved/<dataset>/sasrec/<CHECK_PATH>/pytorch_model.bin.
##
## Examples:
##   bash experiments/sasrec_frequency_group.bash
##   DATASETS="beauty2014 games" FREQ_BIN_SIZE=2 bash experiments/sasrec_frequency_group.bash
##   GROUP_BY=item FREQ_THRESHOLDS="5,10,20,50" bash experiments/sasrec_frequency_group.bash

set -euo pipefail

gpu_id=${GPU_ID:-0}
seed=${SEED:-42}
group_by=${GROUP_BY:-user}
topk=${TOPK:-10}
bin_size=${FREQ_BIN_SIZE:-1}
thresholds=${FREQ_THRESHOLDS:-}
check_path=${CHECK_PATH:-}
output_dir=${OUTPUT_DIR:-./outputs/sasrec_frequency_group}
datasets=${DATASETS:-"fashion beauty2014 games appliances musical"}

max_len_for_dataset() {
        case "$1" in
                fashion)
                        echo 50
                        ;;
                *)
                        echo 200
                        ;;
        esac
}

for dataset in ${datasets}
do
        max_len=$(max_len_for_dataset "${dataset}")
        extra_args=()
        if [ -n "${thresholds}" ]; then
                extra_args+=(--freq_thresholds "${thresholds}")
        else
                extra_args+=(--freq_bin_size "${bin_size}")
        fi
        if [ "${NO_CUDA:-0}" = "1" ]; then
                extra_args+=(--no_cuda)
        fi

        python3 train_baseline.py --dataset "${dataset}" \
                --model_name sasrec \
                --hidden_size 64 \
                --train_batch_size 128 \
                --max_len "${max_len}" \
                --gpu_id "${gpu_id}" \
                --num_workers 8 \
                --seed "${seed}" \
                --check_path "${check_path}" \
                --lr 0.001 \
                --l2 0.0001 \
                --trm_num 2 \
                --num_heads 1 \
                --dropout_rate 0.5 \
                --do_freq_group \
                --freq_group_by "${group_by}" \
                --freq_topk "${topk}" \
                --freq_output_dir "${output_dir}" \
                "${extra_args[@]}"
done
