# Differential Transformer for Vision Classification

This research explores the novel [Differential Transformer](https://arxiv.org/abs/2410.05258) architecture for supervised image classification tasks and measures its performance compared to traditional ViT architecture.

## Methodology

We train 12 models in total, of sizes 10, 20, 30 million parameters. 6 models are trained on classic [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet), while other 6 are trained on noised version of it. 6 models consist of 2 types of models: classic ViTs and ViT's based on [Differential Transformer](https://arxiv.org/abs/2410.05258) architecture, of different sizes.

Later we compare their training and validation loss curves, along with test accuracy, and compare them by size / arhitecture / dataset and draw conclusions about performance of _Differential Vision Transformer_.

## Models

| Model Type                  | Dataset                | Size |
|-----------------------------|------------------------|------|
| Classic ViT                 | Tiny ImageNet          | 10M  |
| Classic ViT                 | Tiny ImageNet          | 20M  |
| Classic ViT                 | Tiny ImageNet          | 30M  |
| Differential Transformer ViT| Tiny ImageNet          | 10M  |
| Differential Transformer ViT| Tiny ImageNet          | 20M  |
| Differential Transformer ViT| Tiny ImageNet          | 30M  |
| Classic ViT                 | Noised Tiny ImageNet   | 10M  |
| Classic ViT                 | Noised Tiny ImageNet   | 20M  |
| Classic ViT                 | Noised Tiny ImageNet   | 30M  |
| Differential Transformer ViT| Noised Tiny ImageNet   | 10M  |
| Differential Transformer ViT| Noised Tiny ImageNet   | 20M  |
| Differential Transformer ViT| Noised Tiny ImageNet   | 30M  |

## Training Hyperparameters

Hyperparameters are equal among the models.

|||
|-------------------|-----------------------|
| Hardware          | 4x NVIDIA RTX4090     |
| Training Images   | 100k                  |
| Validation Images | 10k                   |
| Epochs            | 50                    |
| Batch Size        | 1024                  |
| Warmup Steps      | 500                   |
| Optimiser         | AdamW                 |
| Learning Rate     | 1e-4, cosine annealing|
| Weight Decay      | 1e-2                  |
