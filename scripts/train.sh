#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

python train/trainer.py \
--pretrained_path "/home/trapoom555/data/hf_models/stable-diffusion-v1-5" \
--monet_path "/home/trapoom555/data/monet_data/all_monet_wiki_art" \
--center_crop false \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.0 \
--train_strength 1.0 \
--learning_rate 0.00001 \
--batch_size 2 \
--grad_accumulation 1 \
--max_epochs 1 \
--save_name "monet_all"