import torch
import torch.nn as nn

from compress.tucker_decomposition.tucker_manifold import TangentVector


class RiemannParameter(nn.Parameter):
    def __init__(self, data: torch.Tensor):
        super().__init__()
        super().__new__(RiemannParameter, data)

    @property
    def riemann(self):
        return True


class RiemannModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def regular_parameters(self):
        """
        :return: regulary trainable parameters (conv, batchnorm, etc).
        Does not return riemann parameters
        """
        params = nn.ParameterList()
        for param in self.parameters():
            if not isinstance(param, RiemannParameter):
                params.append(param)
        return params

    # def to(self, device, dtype=None, non_blocking=True):
    #     for param in self.riemann_parameters():
    #         param.to(device, non_blocking=non_blocking)
    #     return self

    def riemann_parameters(self):
        params = nn.ParameterList()

        def traverse_model(entry):
            for child in entry.children():
                traverse_model(child)
                if isinstance(child, RiemannLayer):
                    params.append(child.riemann_parameters())

        traverse_model(self)
        return params

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class RiemannLayer(nn.Module):
    def __init__(self, core, factors, rank):
        super().__init__()
        self.core = RiemannParameter(core)
        self.factors = nn.ParameterList(RiemannParameter(factor) for factor in factors)
        self.rank = rank

    def riemann_parameters(self):
        params = nn.ParameterList([self.core, *self.factors])
        return params

    def to(self, device, dtype=None, non_blocking=True):
        self.core.to(device, non_blocking=non_blocking)
        for factor in self.factors:
            factor.to(device, non_blocking=non_blocking)
        return self

    def train(self, mode: bool = True):
        if mode:
            if self.core.shape[0] == self.rank[0]:
                self.core.data = TangentVector.group_cores(
                    self.core.data, self.core.data
                )
                for i in range(len(self.factors)):
                    self.factors[i].data = torch.hstack(
                        [self.factors[i].data, torch.zeros_like(self.factors[i].data)]
                    )
        else:
            if self.core.shape[0] > self.rank[0]:
                rank_slices = [
                    slice(0, self.rank[i], None) for i in range(len(self.rank))
                ]
                self.core.data = self.core[tuple(rank_slices)]
                for i in range(len(self.factors)):
                    self.factors[i].data = self.factors[i][:, : self.rank[i]]
        super().train(mode)
