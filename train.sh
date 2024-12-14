#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

python train/trainer.py \
--pretrained_path "/home/trapoom555/data/hf_models/stable-diffusion-2-1" \
--monet_path "/home/trapoom555/data/monet_data/monet_jpg" \
--center_crop false \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--train_strength 0.3 \
--learning_rate 0.00001 \
--batch_size 8 \
--max_epochs 50 \
--save_name "monet_300"