#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

python inference/transfer_style.py \
--pretrained_path "/home/trapoom555/data/hf_models/stable-diffusion-2-1" \
--lora_path "/home/trapoom555/code/Diffusion-LoRA-Style-Transfer/lora_ckpt/300_steps" \
--image_path "/home/trapoom555/code/Diffusion-LoRA-Style-Transfer/ting.JPG" \
--strength 0.3 \
--save_path "styled_image.jpg"