#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

OUT_FOLDER="out_monet_all"

if [ ! -d "$OUT_FOLDER" ]; then
    mkdir -p "$OUT_FOLDER"
fi

python inference/inference_folder.py \
--pretrained_path "/home/trapoom555/data/hf_models/stable-diffusion-2-1" \
--lora_path "/home/trapoom555/code/Diffusion-LoRA-Style-Transfer/lora_ckpt/monet_all" \
--folder_path "/home/trapoom555/data/monet_data/photo_jpg/" \
--strength 0.2 \
--save_path "$OUT_FOLDER"