## LLM-ESR -- SASRec, Bert4Rec, GRU4Rec

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

gpu_id=0
dataset="beauty2014"        
seed_list=(42)
ts_user=7
ts_item=6

# model_name="llmesr_colmod"
# for seed in ${seed_list[@]}
# do
#         python3 main.py --dataset ${dataset} \
#                 --model_name ${model_name} \
#                 --hidden_size 64 \
#                 --train_batch_size 2048 \
#                 --max_len 20 \
#                 --gpu_id ${gpu_id} \
#                 --num_workers 2 \
#                 --num_train_epochs 40 \
#                 --seed ${seed} \
#                 --check_path "DoublePeft" \
#                 --keepon \
#                 --keepon_path "/home/zhuzhehua/2025/LLM-ESR2/saved/beauty2014/llmesr_colmod/DoublePeft" \
#                 --patience 20 \
#                 --ts_user ${ts_user} \
#                 --ts_item ${ts_item} \
#                 --freeze \
#                 --log \
#                 --user_sim_func cl \
#                 --alpha 0.1 \
#                 --pair_loss_weight 0.01 \
#                 --collab_llm_ratio 1.0 \
#                 --enable_id
#                 # --no_cuda
#                 # --use_cross_att
# done

model_name="llmesr_sasrec"
for seed in ${seed_list[@]}
do
        python3 main.py --dataset ${dataset} \
                --model_name ${model_name} \
                --hidden_size 64 \
                --train_batch_size 2048 \
                --max_len 20 \
                --gpu_id ${gpu_id} \
                --num_workers 4 \
                --num_train_epochs 40 \
                --seed ${seed} \
                --check_path "" \
                --patience 20 \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --freeze \
                --log \
                --user_sim_func kd \
                --alpha 0.1 \
                --use_cross_att
done


# model_name="llmesr_bert4rec"
# mask_prob=0.6
# for seed in ${seed_list[@]}
# do
#         python3 main.py --dataset ${dataset} \
#                 --model_name ${model_name} \
#                 --hidden_size 64 \
#                 --train_batch_size 128 \
#                 --max_len 200 \
#                 --gpu_id ${gpu_id} \
#                 --num_workers 8 \
#                 --mask_prob ${mask_prob} \
#                 --num_train_epochs 200 \
#                 --seed ${seed} \
#                 --check_path "" \
#                 --patience 20 \
#                 --ts_user ${ts_user} \
#                 --ts_item ${ts_item} \
#                 --freeze \
#                 --log \
#                 --user_sim_func kd \
#                 --alpha 0.1 \
#                 # --no_cuda
#                 # --use_cross_att
# done


# model_name="llmesr_gru4rec"
# for seed in ${seed_list[@]}
# do
#         python3 main.py --dataset ${dataset} \
#                 --model_name ${model_name} \
#                 --hidden_size 64 \
#                 --train_batch_size 128 \
#                 --max_len 200 \
#                 --gpu_id ${gpu_id} \
#                 --num_workers 8 \
#                 --num_train_epochs 200 \
#                 --seed ${seed} \
#                 --check_path "" \
#                 --patience 20 \
#                 --ts_user ${ts_user} \
#                 --ts_item ${ts_item} \
#                 --freeze \
#                 --log \
#                 --user_sim_func kd \
#                 --alpha 0.1 \
#                 # --no_cuda
#                 # --use_cross_att
# done