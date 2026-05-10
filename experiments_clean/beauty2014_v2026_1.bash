gpu_id=0
dataset="beauty2014"
seed_list=(42 43 44)

for seed in "${seed_list[@]}"
do
  python main.py \
    --model_name llmesr_clean \
    --dataset ${dataset} \
    --fusion concat \
    --use_align_loss \
    --enable_id \
    --use_intent_gap \
    --dynamic_align \
    --dynamic_align_scale 1.0 \
    --graph_filter positive_residual \
    --adapter_type mlp \
    --alpha 0.1 \
    --collab_llm_ratio 1.0 \
    --hidden_size 64 \
    --trm_num 2 \
    --num_heads 1 \
    --dropout_rate 0.5 \
    --max_len 20 \
    --train_batch_size 2048 \
    --test_neg 100 \
    --lr 0.001 \
    --num_train_epochs 200 \
    --patience 20 \
    --watch_metric NDCG@10 \
    --seed ${seed} \
    --check_path clean_v2026_1_seed${seed} \
    --gpu_id ${gpu_id} \
    --log
done
