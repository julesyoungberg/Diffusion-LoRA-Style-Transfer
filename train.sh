#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

python train/trainer.py \
--pretrained_path "/home/trapoom555/data/hf_models/stable-diffusion-2-1" \
--monet_path "/home/trapoom555/data/monet_data/all_monet_wiki_art" \
--center_crop false \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--train_strength 0.3 \
--learning_rate 0.00001 \
--batch_size 8 \
--grad_accumulation 4 \
--max_epochs 50 \
--save_name "monet_all_bs_32"