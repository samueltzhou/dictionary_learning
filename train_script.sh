CUDA_VISIBLE_DEVICES=1;
# python demo.py \
#   --save_dir ./my_first_sae \
#   --model_name EleutherAI/pythia-14m \
#   --layers 1 \
#   --architectures standard \
#   --use_wandb;
python demo.py \
  --save_dir ./my_first_sae \
  --model_name EleutherAI/pythia-14m \
  --layers 1 \
  --architectures seqdropout_batch_top_k \
  --use_wandb