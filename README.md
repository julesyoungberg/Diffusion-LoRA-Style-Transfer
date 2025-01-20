# Diffusion LoRA Style Transfer

This project aims to tackle the Neural Style Transfer problem using a fine-tuned diffusion model with the LoRA technique. It is demonstrated that the style transfer can be achieved with **under 1 minute of fine-tuning on a single RTX3090** using this method.

<img src="https://github.com/trapoom555/Diffusion-LoRA-Style-Transfer/blob/main/assets/example_results.png?raw=true" />

## Motivation

Neural style transfer methods such as [Cycle-GAN](https://arxiv.org/pdf/1703.10593) has achieved high performance. However, it requires extensive training time and memory usage due to the need to optimize 4 neural networks from scratch. This project explores an alternative way to tackle this problem using diffusion models by leveraging their large-scale pretraining advantage to achieve faster convergence time with easier objective and low memory requirement for achieving efficient style transfer.

## Methodology

We first fine-tune the diffusion model using Monet paintings and their corresponding painting names as captions. We prepend "A Monet painting," as an identifier to associate this phrase with the Monet style inspired by [DreamBooth](https://arxiv.org/pdf/2208.12242). We fine-tune the model using [LoRA](https://arxiv.org/pdf/2106.09685) parameter efficient fine-tuning method.

<img src="https://github.com/trapoom555/Diffusion-LoRA-Style-Transfer/blob/main/assets/method_train.png?raw=true" />

Once the model has learned the target style distribution, we use the model to denoise the diffused latent vector from N-th steps. We designed this pipeline based on the insight from [SDEdit](https://arxiv.org/pdf/2108.01073) that we can solve SDE from any intermediate timestep to modify the original image. To retain the original image details of the original image, we further add the [IP-Adapter](https://arxiv.org/pdf/2308.06721) as an image condition to the denoiser.

<img src="https://github.com/trapoom555/Diffusion-LoRA-Style-Transfer/blob/main/assets/method_inference.png?raw=true" />

## Data

We use the Monet painting dataset from WikiArt as our experimental dataset. It can be downloaded [here](https://www.kaggle.com/datasets/steubk/wikiart).

## How to use this repository

### Monet dataset caption generation

```bash
python ./data/caption.py
```

### LoRA fine-tuning

```bash
export PYTHONPATH="$PYTHONPATH:$pwd"

python train/trainer.py \
    --pretrained_path "/path/to/stable-diffusion-v1-5" \
    --images_path "<path_to_images>" \
    --center_crop false \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --train_strength 1.0 \
    --learning_rate 0.00001 \
    --batch_size 2 \
    --grad_accumulation 1 \
    --max_epochs 1 \
    --save_name "<style_name>"
```

### Transfer style to content image

```bash
./scripts/infer_img.sh
```

### Apply style to all content images in a folder

```bash
./scripts/infer_img.sh
```

## Hyperparameter Tips

The outcomes largely depend on two hyperparameters including `--image_cond_scale` and `--strength`. The first hyperparameter determines how strong we condition the original image on the output. If we want the output to be closer to the original image, set this value high (close to 1.0). The second hyperparameter indicates how many steps that we diffuse the latent vector, the higher this value is, the closer the output to the Monet distribution is. But if the strength is too high, the outcome will be far from the original image.

## Footnote

This work is one of the experiments in the final project of Big Data Intelligence Fall 2024 course at Tsinghua University 🟣. We would like to express our sincere gratitude to this course !
