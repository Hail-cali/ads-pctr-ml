"""DeepFM: Factorization Machine + Deep Neural Network (Guo et al., 2017).

Architecture:
  Input (dense 13 + sparse 26)
    -> Shared Embedding (dim=16)
    -> FM Layer (2nd order, O(kN))  +  DNN [256,128,64] (BN + Dropout)
    -> sigmoid -> P(click)
"""

from typing import List

import torch
import torch.nn as nn


class FeaturesLinear(nn.Module):
    """First-order feature interactions."""

    def __init__(self, num_dense: int, num_sparse_fields: int, hash_bucket_size: int):
        super().__init__()
        self.dense_linear = nn.Linear(num_dense, 1)
        total_sparse_size = num_sparse_fields * hash_bucket_size
        self.sparse_embedding = nn.Embedding(total_sparse_size, 1, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.sparse_embedding.weight)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        return self.dense_linear(dense) + self.sparse_embedding(sparse).sum(dim=1) + self.bias


class FeaturesEmbedding(nn.Module):
    """Shared embedding layer for FM and DNN.

    Dense features use per-field independent embeddings (scaled by feature value)
    to preserve FM's field-independence assumption.
    """

    def __init__(self, num_dense: int, num_sparse_fields: int, hash_bucket_size: int, embed_dim: int):
        super().__init__()
        self.dense_embedding = nn.Parameter(torch.empty(num_dense, embed_dim))
        total_sparse_size = num_sparse_fields * hash_bucket_size
        self.sparse_embedding = nn.Embedding(total_sparse_size, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.dense_embedding)
        nn.init.xavier_uniform_(self.sparse_embedding.weight)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        # Per-field: scale embedding by feature value → (batch, num_dense, embed_dim)
        dense_emb = dense.unsqueeze(2) * self.dense_embedding
        sparse_emb = self.sparse_embedding(sparse)
        return torch.cat([dense_emb, sparse_emb], dim=1)


class FactorizationMachine(nn.Module):
    """Second-order interactions via sum-of-square trick: O(kN)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        square_of_sum = x.sum(dim=1).pow(2)
        sum_of_square = x.pow(2).sum(dim=1)
        return 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)


class MultiLayerPerceptron(nn.Module):
    """DNN with BatchNorm and Dropout."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2, final_projection: bool = True):
        super().__init__()
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        if final_projection:
            layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DeepFM(nn.Module):
    """DeepFM = Linear + FM + DNN with shared embeddings."""

    def __init__(
        self,
        num_dense: int = 13,
        num_sparse_fields: int = 26,
        hash_bucket_size: int = 100_000,
        embed_dim: int = 16,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        num_fields = num_dense + num_sparse_fields
        self.linear = FeaturesLinear(
            num_dense = num_dense,
            num_sparse_fields = num_sparse_fields,
            hash_bucket_size = hash_bucket_size
        )
        self.embedding = FeaturesEmbedding(
            num_dense,
            num_sparse_fields,
            hash_bucket_size,
            embed_dim
        )
        self.fm = FactorizationMachine()
        self.dnn = MultiLayerPerceptron(
            num_fields * embed_dim,
            hidden_dims,
            dropout
        )

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear(dense, sparse)
        emb = self.embedding(dense, sparse)
        fm_out = self.fm(emb)
        dnn_out = self.dnn(emb.view(emb.size(0), -1))
        return (linear_out + fm_out + dnn_out).squeeze(1)
