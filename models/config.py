from .differential import DifferentialViT
from .classic import ClassicViT
from typing import Any

models: dict[str, Any] = {
    "classic": {
        "10M": ClassicViT(
            img_size=64,
            patch_size=8,
            num_classes=200,
            embed_dim=384,
            depth=12,
            num_heads=8,
            mlp_dim=1024,
            in_channels=3,
            dropout=0.1,
        ),
        "20M": ClassicViT(
            img_size=64,
            patch_size=8,
            num_classes=200,
            embed_dim=512,
            depth=14,
            num_heads=8,
            mlp_dim=1344,
            in_channels=3,
            dropout=0.1,
        ),
        "30M": ClassicViT(
            img_size=64,
            patch_size=8,
            num_classes=200,
            embed_dim=640,
            depth=16,
            num_heads=8,
            mlp_dim=1536,
            in_channels=3,
            dropout=0.1,
        ),
    },
    "differential": {
        "10M": DifferentialViT(
            img_size=64,
            patch_size=8,
            num_classes=200,
            embed_dim=288,
            depth=10,
            num_heads=8,
            mlp_dim=1088,
            in_channels=3,
            dropout=0.1,
        ),
        "20M": DifferentialViT(
            img_size=64,
            patch_size=8,
            num_classes=200,
            embed_dim=384,
            depth=12,
            num_heads=8,
            mlp_dim=1344,
            in_channels=3,
            dropout=0.1,
        ),
        "30M": DifferentialViT(
            img_size=64,
            patch_size=8,
            num_classes=200,
            embed_dim=464,
            depth=14,
            num_heads=8,
            mlp_dim=1472,
            in_channels=3,
            dropout=0.1,
        ),
    },
}