import torch
from torch.optim import Optimizer

from compress import Tucker
from compress.tucker_decomposition.tucker_manifold import TangentVector, project


class RiemannTuckerOptimizer(Optimizer):
    """
    RiemannOptimizer([
        {"params": ..., "rank: ..., "lr": ...},
    ])
    """

    def __init__(self, params_groups):
        for group in params_groups:
            self._gauge_params(group["params"], group["rank"])

        # defaults = dict(base_lr=base_lr, momentum_beta=momentum_beta)
        super().__init__(params_groups, {})

        # self.base_lr = base_lr
        # self.momentum_beta = momentum_beta

        self.momentums = [None] * len(self.param_groups)
        self.directions = [None] * len(self.param_groups)

    @staticmethod
    def _gauge_params(params, rank):
        # pad params with zeros
        for i in range(len(params)):
            param = params[i]
            if i == 0:
                param.data = TangentVector.group_cores(param.data, param.data)
            else:
                param.data = torch.hstack([param.data, torch.zeros_like(param.data)])


class SGDmomentumTucker(RiemannTuckerOptimizer):
    def __init__(self, riemann_param_groups):
        """
        each layer's params goes in separate param group
        """
        super().__init__(riemann_param_groups)

    def _riemann_grad(self):
        for group_idx, group in enumerate(self.param_groups):
            rk = group["rank"]
            xk_factors = []
            d_factors = []
            rank_slices = None
            for idx, param in enumerate(group["params"]):
                if idx == 0:
                    S, dS = param.data, param.grad
                    rank_slices = tuple([slice(0, rk[i]) for i in range(dS.ndim)])
                    dS = dS[rank_slices]
                    S = S[rank_slices]
                    modes = list(torch.arange(0, S.ndim))
                else:
                    U = param.data[:, : rk[idx - 1]]
                    xk_factors.append(U)
                    dU = param.grad[:, rk[idx - 1] :]
                    dU = dU - U @ (U.T @ dU)
                    unfolding = torch.permute(
                        S, [modes[idx - 1], *(modes[: idx - 1] + modes[idx - 1 + 1 :])]
                    )
                    unfolding = torch.flatten(unfolding, 1)
                    gram = unfolding @ unfolding.T
                    lu, pivot, _ = torch.linalg.lu_factor_ex(gram)
                    dU = torch.linalg.lu_solve(lu, pivot, dU.T, left=True).T
                    d_factors.append(dU)
                    # factors.append(torch.hstack([U, dU]))
            x_k = Tucker(S, xk_factors)
            r_grad = TangentVector(x_k, dS, d_factors)
            if self.directions[group_idx] is not None:
                self.momentums[group_idx] = project(
                    x_k, self.directions[group_idx].construct()
                )
                self.directions[group_idx] = r_grad.linear_comb(
                    1, group["momentum_beta"], self.momentums[group_idx]
                )
            else:
                self.directions[group_idx] = r_grad

    # @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        self._riemann_grad()
        for group_idx, group in enumerate(self.param_groups):
            rank = group["rank"]
            x_k = self.directions[group_idx].linear_comb(-group["lr"]).construct()
            x_k = x_k.round(rank)

            group["params"][0].data = x_k.core
            for factor_idx in range(len(group["params"]) - 1):
                group["params"][factor_idx + 1].data = x_k.factors[factor_idx]
            self._gauge_params(group["params"], group["rank"])


class TuckerRiemannAdam(RiemannTuckerOptimizer):
    def __init__(
        self, riemann_param_groups, betas=(0.9, 0.999), eps=1e-8, step_velocity=1
    ):
        """
        each layer's params goes in separate param group
        """
        super().__init__(riemann_param_groups)
        self.second_momentums = torch.zeros(len(riemann_param_groups))
        self.betas = betas
        self.eps = eps
        self.step_velocity = step_velocity

        self.step_t = 1

    def _riemann_grad(self):
        for group_idx, group in enumerate(self.param_groups):
            rk = group["rank"]
            xk_factors = []
            d_factors = []
            rank_slices = None
            for idx, param in enumerate(group["params"]):
                if idx == 0:
                    S, dS = param.data, param.grad
                    if dS is None:
                        raise ValueError(
                            f"{group_idx} param group have not grad", group
                        )
                    rank_slices = tuple([slice(0, rk[i]) for i in range(dS.ndim)])
                    dS = dS[rank_slices]
                    S = S[rank_slices]
                    modes = list(torch.arange(0, S.ndim))
                else:
                    U = param.data[:, : rk[idx - 1]]
                    xk_factors.append(U)
                    dU = param.grad[:, rk[idx - 1] :]
                    dU = dU - U @ (U.T @ dU)
                    unfolding = torch.permute(
                        S, [modes[idx - 1], *(modes[: idx - 1] + modes[idx - 1 + 1 :])]
                    )
                    unfolding = torch.flatten(unfolding, 1)
                    gram = unfolding @ unfolding.T
                    lu, pivot, _ = torch.linalg.lu_factor_ex(gram)
                    dU = torch.linalg.lu_solve(lu, pivot, dU.T, left=True).T
                    d_factors.append(dU)
            x_k = Tucker(S, xk_factors)
            r_grad = TangentVector(x_k, dS, d_factors)
            r_grad_norm = r_grad.construct().norm(qr_based=True).detach()
            if self.momentums[group_idx] is not None:
                self.momentums[group_idx] = project(
                    x_k, self.momentums[group_idx].construct()
                ).linear_comb(self.betas[0], 1 - self.betas[0], r_grad)
            else:
                self.momentums[group_idx] = (1 - self.betas[0]) * r_grad
            self.second_momentums[group_idx] = (
                self.betas[1] * self.second_momentums[group_idx]
                + (1 - self.betas[1]) * r_grad_norm**2
            )
            second_momentum_corrected = self.second_momentums[group_idx] / (
                1 - self.betas[1] ** (self.step_t // self.step_velocity + 1)
            )
            bias_correction_ratio = (
                1 - self.betas[0] ** (self.step_t // self.step_velocity + 1)
            ) * torch.sqrt(second_momentum_corrected) + self.eps
            self.directions[group_idx] = (1 / bias_correction_ratio) * self.momentums[
                group_idx
            ]

    # @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        self._riemann_grad()
        for group_idx, group in enumerate(self.param_groups):
            rank = group["rank"]

            x_k = self.directions[group_idx].linear_comb(-group["lr"]).construct()
            x_k = x_k.round(rank)

            group["params"][0].data = x_k.core
            for factor_idx in range(len(group["params"]) - 1):
                group["params"][factor_idx + 1].data = x_k.factors[factor_idx]
            # group["params"][-1].data = x_k.shared_factor
            self._gauge_params(group["params"], group["rank"])
        self.step_t += 1
