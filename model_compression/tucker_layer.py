import torch
import torch.nn as nn

from typing import Sequence

from compress import TuckerMatrix

from .rieman_model import RiemannLayer



class TuckerLinear(RiemannLayer):
    def __init__(self, d_in: int, d_out: int, layer: nn.Linear, rank: Sequence[int],
                 dims: Sequence[Sequence[int]], bias=True, eps=1e-8):
        """
        :param d_in: input dimention
        :param d_out: output dimention
        :param layer: nn.Linear, layer which will be replaced
        :param rank: multilinear rank
        :param dims: sequence of dimentions of Tucker representation
        :param bias: whether to use bias
        :param eps: for HOSVD
        """
        self.d_in = d_in
        self.d_out = d_out
        self.dims = dims
        self.eps = eps
        matrix = self._decompose(layer, rank)
        super(TuckerLinear, self).__init__(matrix.core, matrix.factors, rank)
        if layer.bias is not None and bias:
            self.bias = nn.Parameter(layer.bias)
        else:
            self.bias = torch.zeros((d_out,), dtype=torch.float32)
            self.register_buffer("bias", self.bias)

    def to(self, device, dtype=None, non_blocking=True):
        super().to(device, dtype, non_blocking)
        self.bias.to(device, non_blocking=non_blocking)
        return self

    def _decompose(self, layer: nn.Linear, rank) -> TuckerMatrix:
        data = layer.weight.data
        # data = torch.randn_like(data)
        n = len(self.dims[0]) // 2
        data = data.reshape(self.dims[0])
        data = data.permute([i // 2 if i % 2 == 0 else i // 2 + n for i in range(2 * n)])
        data = data.reshape(self.dims[1])
        matrix = TuckerMatrix.from_dense(data, self.dims[0][:3], self.dims[0][3:], self.eps)
        matrix = matrix.round(max_rank=rank)
        return matrix

    def forward(self, x):
        # X: B x seq_len x h
        x_wav = x.reshape((x.shape[0], x.shape[1], *self.dims[0][:3]))
        matrix = TuckerMatrix(self.core, self.factors, self.dims[0][:3], self.dims[0][3:])
        res = matrix @ x_wav
        return res.reshape(*x.shape[:-1], -1) + self.bias
