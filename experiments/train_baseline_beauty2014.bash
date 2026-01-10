## Baseline Models Training Script -- SASRec, Bert4Rec, GRU4Rec

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

gpu_id=0
dataset="beauty2014"        
seed_list=(42)
ts_user=7
ts_item=6

# Train SASRec
model_name="sasrec"
for seed in ${seed_list[@]}
do
        python3 train_baseline.py --dataset ${dataset} \
                --model_name ${model_name} \
                --hidden_size 64 \
                --train_batch_size 128 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --num_train_epochs 200 \
                --seed ${seed} \
                --check_path "" \
                --patience 20 \
                --lr 0.001 \
                --l2 0.0001 \
                --trm_num 2 \
                --num_heads 1 \
                --dropout_rate 0.5 \
                --log
                # --use_seq2seq  # uncomment to use seq2seq loss
done

# Train Bert4Rec
model_name="bert4rec"
mask_prob=0.6
for seed in ${seed_list[@]}
do
        python3 train_baseline.py --dataset ${dataset} \
                --model_name ${model_name} \
                --hidden_size 64 \
                --train_batch_size 128 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --mask_prob ${mask_prob} \
                --num_train_epochs 200 \
                --seed ${seed} \
                --check_path "" \
                --patience 20 \
                --lr 0.001 \
                --l2 0.0001 \
                --trm_num 2 \
                --num_heads 1 \
                --dropout_rate 0.5 \
                --log
done

# Train GRU4Rec
model_name="gru4rec"
for seed in ${seed_list[@]}
do
        python3 train_baseline.py --dataset ${dataset} \
                --model_name ${model_name} \
                --hidden_size 64 \
                --train_batch_size 128 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --num_train_epochs 200 \
                --seed ${seed} \
                --check_path "" \
                --patience 20 \
                --lr 0.001 \
                --l2 0.0001 \
                --num_layers 1 \
                --dropout_rate 0.5 \
                --log
                # --use_seq2seq  # uncomment to use seq2seq loss
done
