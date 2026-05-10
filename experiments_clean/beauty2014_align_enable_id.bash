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
    --alpha 0.1 \
    --collab_llm_ratio 1.0 \
    --hidden_size 64 \
    --trm_num 2 \
    --num_heads 1 \
    --dropout_rate 0.5 \
    --max_len 200 \
    --train_batch_size 256 \
    --test_neg 100 \
    --lr 0.001 \
    --num_train_epochs 200 \
    --patience 20 \
    --watch_metric NDCG@10 \
    --seed ${seed} \
    --check_path clean_colmod_align_enable_id_seed${seed} \
    --gpu_id ${gpu_id} \
    --log
done
