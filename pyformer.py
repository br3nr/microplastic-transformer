import torch
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_channel: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 dropout: float = 0.3,
                 pe: bool = False):
        super().__init__()

        self._d_input = d_input
        self._d_channel = d_channel
        self._d_model = d_model
        self._pe = pe

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      dropout=dropout) for _ in range(N)])

        self._embedding = nn.Linear(self._d_channel, d_model)
        self._linear = nn.Linear(d_model * d_input, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        encoding = self._embedding(x)

        if self._pe:
            pe = torch.ones_like(encoding[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000)/self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            # shape:[input, d_model/2]
            temp = torch.matmul(position.float(), temp)
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding = encoding + pe

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        encoding = encoding.reshape(encoding.shape[0], -1)

        output = self._linear(encoding)

        return output


class PositionwiseFeedForward(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_ff: Optional[int] = 2048):
        """Initialize the PFF block."""
        super().__init__()

        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self._linear2(F.relu(self._linear1(x)))


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int):
        """Initialize the Multi Head Block."""
        super().__init__()

        self._q = q
        self._h = h

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q * h)
        self._W_k = nn.Linear(d_model, q * h)
        self._W_v = nn.Linear(d_model, v * h)

        # Output linear function
        self._W_o = nn.Linear(v * h, d_model)

        # Score placeholder
        self._scores = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:

        Q = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)
        # Scaled Dot Product
        self._scores = torch.matmul(
            Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        # Apply softmax
        # shape [batchsize * head_num, input, input]
        self._scores = F.softmax(self._scores, dim=-1)
        # scores * values
        attention = torch.matmul(self._scores, V)
        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)
        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)
        return self_attention


class Encoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 dropout: float = 0.3):
        """Initialize the Encoder block"""
        super().__init__()

        MHA = MultiHeadAttention
        self._selfAttention = MHA(d_model, q, v, h)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        # Dropout
        self._dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x = self._dropout(x)
        x = self._layerNorm1(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dropout(x)
        x = self._layerNorm2(x + residual)

        return x