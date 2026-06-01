gpu_id=0
dataset=${1:-${DATASET:-beauty2014}}
seed_list=(42 43 44)

for seed in "${seed_list[@]}"
do
  python main.py \
    --model_name llmesr_clean \
    --dataset ${dataset} \
    --fusion concat \
    --colmod_compat \
    --use_align_loss \
    --enable_id \
    --alpha 0.1 \
    --user_sim_func cl \
    --collab_llm_ratio 1.0 \
    --hidden_size 64 \
    --trm_num 2 \
    --num_heads 2 \
    --dropout_rate 0.5 \
    --max_len 20 \
    --freeze \
    --train_batch_size 128 \
    --test_neg 100 \
    --lr 0.001 \
    --num_train_epochs 40 \
    --patience 20 \
    --watch_metric NDCG@10 \
    --ts_user 7 \
    --ts_item 6 \
    --num_workers 2 \
    --seed ${seed} \
    --check_path clean_colmod_compat_seed${seed} \
    --gpu_id ${gpu_id} \
    --log
done
