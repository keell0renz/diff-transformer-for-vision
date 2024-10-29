import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming RoPE is implemented in a separate module
from ._rotary import apply_rotary_emb  # RoPE function



def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class DifferentialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, depth, lambda_init):
        super(DifferentialAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_rep = num_heads // 2  # Splitting heads for differential attention

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Lambda parameters for differential attention
        self.lambda_init = lambda_init
        self.lambda_q1 = nn.Parameter(torch.randn(head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(head_dim) * 0.1)

        # RMSNorm as sublayer normalization
        self.subnorm = nn.RMSNorm(2 * head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, cos, sin):
        batch_size, seq_len, embed_dim = x.shape

        # Compute Q, K, V projections
        q = self.q_proj(x).view(batch_size, seq_len, 2 * self.num_heads, self.head_dim)
        k = self.k_proj(x).view(
            batch_size, seq_len, 2 * (self.num_heads // self.n_rep), self.head_dim
        )
        v = self.v_proj(x).view(
            batch_size, seq_len, self.num_heads // self.n_rep, 2 * self.head_dim
        )

        # Apply RoPE inside attention on q and k using provided cos and sin
        q = apply_rotary_emb(q, cos, sin, interleaved=True)
        k = apply_rotary_emb(k, cos, sin, interleaved=True)

        # Differential attention maps
        q = q.transpose(1, 2) # type: ignore
        k = k.transpose(1, 2).repeat(1, self.n_rep, 1, 1)  # type: ignore
        v = v.transpose(1, 2).repeat(1, self.n_rep, 1, 1)

        attn_weights = torch.matmul(q * self.head_dim**-0.5, k.transpose(-1, -2))

        # Skipping mask because it is ViT lmao

        # Compute differential attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn_weights = attn_weights.view(
            batch_size, self.num_heads, 2, seq_len, seq_len
        )
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            self.subnorm(attn_output).transpose(1, 2).reshape(batch_size, seq_len, -1)
        )
        return self.out_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.dropout(x)
        return self.fc2(x)


class DifferentialEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, depth, dropout=0.1):
        super(DifferentialEncoderBlock, self).__init__()
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = DifferentialAttention(
            embed_dim,
            num_heads,
            embed_dim // num_heads,
            depth,
            lambda_init_fn(depth)
        )
        self.norm2 = nn.RMSNorm(embed_dim)
        self.mlp = SwiGLU(embed_dim, mlp_dim, dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


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
        x = self.proj(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x


class DifferentialViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        num_classes,
        embed_dim,
        depth,
        num_heads,
        mlp_dim,
        in_channels=3,
        dropout=0.1,
    ):
        super(DifferentialViT, self).__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DifferentialEncoderBlock(
                embed_dim, 
                num_heads, 
                mlp_dim, 
                d, 
                dropout
            ) for d in range(depth)
        ])

        self.norm = nn.RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize parameters
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.normal_(self.head.bias, mean=0.0, std=1e-6)

    def forward(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, embed_dim = x.shape

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        seq_len += 1

        x = self.dropout(x)

        # Generate cos and sin tensors for RoPE
        rotary_dim = embed_dim // 2
        cos, sin = self.get_rotary_embedding(seq_len, rotary_dim)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        return self.head(x[:, 0])

    def get_rotary_embedding(self, seq_len, rotary_dim):
        # Define frequencies (logarithmically spaced)
        # Frequencies are inversely proportional to the dimensions to encode longer distances
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))

        # Calculate angles based on position and frequencies
        positions = torch.arange(seq_len).unsqueeze(1)
        angles = positions * inv_freq.unsqueeze(0)  # Shape: (seq_len, rotary_dim // 2)

        # Compute cosine and sine embeddings
        cos = torch.cos(angles)  # Shape: (seq_len, rotary_dim // 2)
        sin = torch.sin(angles)  # Shape: (seq_len, rotary_dim // 2)

        # Interleave cos and sin to match the rotary dimensions
        cos = cos.repeat_interleave(2, dim=1)  # Shape: (seq_len, rotary_dim)
        sin = sin.repeat_interleave(2, dim=1)  # Shape: (seq_len, rotary_dim)

        # Move cos and sin to the same device as the model parameters
        device = next(self.parameters()).device
        cos = cos.to(device)
        sin = sin.to(device)

        return cos, sin