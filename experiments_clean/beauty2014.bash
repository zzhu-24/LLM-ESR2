export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python main.py \
  --model_name llmesr_clean \
  --dataset beauty2014 \
  --fusion sum \
  --hidden_size 64 \
  --train_batch_size 256 \
  --num_train_epochs 100 \
  --patience 3 \
  --check_path smoke \
  --gpu_id 0