CUDA_VISIBLE_DEVICES=0,1;
python demo.py \
  --save_dir ./my_first_sae \
  --model_name EleutherAI/pythia-70m-deduped \
  --layers 1 \
  --architectures standard \
  --use_wandb;