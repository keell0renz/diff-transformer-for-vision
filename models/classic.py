import torch.nn as nn
import torch


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x (batch_size, channels, height, width)

        batch_size, channels, height, width = x.shape

        assert height % self.patch_size == 0, "Height must be divisible by patch_size."
        assert width % self.patch_size == 0, "Width must be divisible by patch_size."

        x = self.proj(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x (sequence_len, batch_size, embed_dim)

        x2 = self.norm1(x)
        attn_output, _ = self.attn(x2, x2, x2)
        x = x + attn_output

        x2 = self.norm2(x)
        x = x + self.mlp(x2)

        return x


class ClassicViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        num_classes,
        embed_dim,
        depth,
        num_heads,
        mlp_dim,
        in_channels,
        dropout,
    ):
        super(ClassicViT, self).__init__()

        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Calculate number of patches for a 64x64 image with patch_size 8x8
        num_patches = (img_size // patch_size) ** 2  # Assuming patch_size divides image size

        # Fixed positional embedding for 64x64 image
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)  # Initialize positional embedding
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.normal_(self.head.bias, std=1e-6)

    def forward(self, x):
        # x (batch_size, channels, img_height, img_width)

        x = self.patch_embed(x)

        batch_size, num_patches, embed_dim = x.shape

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add fixed positional embeddings without interpolation
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = x.transpose(0, 1)  # (num_patches + 1, batch_size, embed_dim)

        for layer in self.transformer:
            x = layer(x)

        x = x.transpose(0, 1)  # (batch_size, num_patches + 1, embed_dim)
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)

        return logits