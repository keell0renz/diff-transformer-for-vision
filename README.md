# Differential Transformer for Vision Classification

_UPDATE_: Research did not achieve intended objectives, maybe because inductive bias of diff transformer is not designed for visual data and / or baseline classic transformers are implemented improperly.

_UPDATE 2_: After talking to Tianzhu Ye, co-author of original paper, I will try to adjust diff vit code to original more, add bias in attention, change RMSNorm to LayerNorm, resize image into 96x96 and use 4x4 patch. Thank you, Mr. Ye!

_UPDATE 3_: Due to being GPU poor student, it is quite problematic to produce meaningful research in this topic. I will move on to other research projects.

Models can be found [here](https://huggingface.co/keell0renz/diff-transformer-for-vision/tree/main/checkpoints), only 10M versions.

At least I have learned a lot about methodology and pipeline building and will be better prepared for next projects.

This research explores the novel [Differential Transformer](https://arxiv.org/abs/2410.05258) architecture for supervised image classification tasks and measures its performance compared to traditional ViT architecture.

## Methodology

We train 6 models in total, of sizes 10, 20, 30 million parameters. These models are trained on classic [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet). The 6 models consist of 2 types of models: classic ViTs and ViT's based on [Differential Transformer](https://arxiv.org/abs/2410.05258) architecture, of different sizes.

Later we compare their training and validation loss curves, and compare them by size and architecture and draw conclusions about performance of _Differential Vision Transformer_.

## Models

| Model Type                  | Dataset                | Size |
|-----------------------------|------------------------|------|
| Classic ViT                 | Tiny ImageNet          | 10M  |
| Classic ViT                 | Tiny ImageNet          | 20M  |
| Classic ViT                 | Tiny ImageNet          | 30M  |
| Differential Transformer ViT| Tiny ImageNet          | 10M  |
| Differential Transformer ViT| Tiny ImageNet          | 20M  |
| Differential Transformer ViT| Tiny ImageNet          | 30M  |
<!-- | Classic ViT                 | Noised Tiny ImageNet   | 10M  |
| Classic ViT                 | Noised Tiny ImageNet   | 20M  |
| Classic ViT                 | Noised Tiny ImageNet   | 30M  |
| Differential Transformer ViT| Noised Tiny ImageNet   | 10M  |
| Differential Transformer ViT| Noised Tiny ImageNet   | 20M  |
| Differential Transformer ViT| Noised Tiny ImageNet   | 30M  | -->

## Training Hyperparameters

Hyperparameters are equal among the models.

|||
|-------------------|-----------------------|
| Hardware          | 4x NVIDIA RTX4090     |
| Training Images   | 100k                  |
| Validation Images | 10k                   |
| Epochs            | 50                    |
| Batch Size        | 512 (10M), 256 (30M)  |
| Warmup Steps      | 500                   |
| Optimiser         | AdamW                 |
| Learning Rate     | 1e-4, cosine annealing|
| Weight Decay      | 1e-2                  |
