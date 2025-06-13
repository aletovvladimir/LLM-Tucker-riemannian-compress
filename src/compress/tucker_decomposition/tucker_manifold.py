from string import ascii_letters
from typing import Callable, List, Tuple, Union

from .tucker import Tucker
from compress import backend as back
from compress.tucker_decomposition.matrix import TuckerMatrix


class TangentVector:
    """Representation of a tangent vector at a point on the Tucker manifold."""

    point: Tucker
    delta_core: back.type()
    delta_factors: List[back.type()]

    def __init__(
        self,
        manifold_point: Tucker,
        delta_core: Union[None, back.type()] = None,
        delta_factors: Union[None, List[back.type()]] = None,
    ):
        self.point = manifold_point
        self.delta_core = delta_core if delta_core is not None else self.point.core
        self.delta_factors = (
            delta_factors
            if delta_factors is not None
            else [back.zeros_like(f) for f in self.point.factors]
        )

    @staticmethod
    def group_cores(corner_core, padding_core):
        """Combine corner and padding cores into grouped core."""
        d = corner_core.ndim
        new_core = back.copy(corner_core)
        to_concat = back.copy(padding_core)

        for i in range(d):
            pad_shape = [
                (0, corner_core.shape[j]) if j == i - 1 else (0, 0) for j in range(d)
            ]
            to_concat = back.pad(
                to_concat, pad_shape, mode="constant", constant_values=0
            )
            new_core = back.concatenate([new_core, to_concat], axis=i)

        return new_core

    def construct(self) -> Tucker:
        """Construct Tucker tensor from this tangent vector."""
        grouped_core = self.group_cores(self.delta_core, self.point.core)
        factors = [
            back.concatenate([self.point.factors[i], self.delta_factors[i]], axis=1)
            for i in range(self.point.ndim)
        ]
        if isinstance(self.point, TuckerMatrix):
            return TuckerMatrix(grouped_core, factors, self.point.n, self.point.m)
        return Tucker(grouped_core, factors)

    def __rmul__(self, a: float) -> "TangentVector":
        return TangentVector(
            self.point,
            a * self.delta_core,
            [a * f for f in self.delta_factors],
        )

    def __neg__(self) -> "TangentVector":
        return -1 * self

    def __add__(self, other: "TangentVector") -> "TangentVector":
        new_delta_core = self.delta_core + other.delta_core
        new_delta_factors = [
            sf + of for sf, of in zip(self.delta_factors, other.delta_factors)
        ]
        return TangentVector(self.point, new_delta_core, new_delta_factors)

    def norm(self) -> float:
        """Frobenius norm of the tangent vector."""
        norms = back.norm(self.delta_core) ** 2
        core_letters = ascii_letters[: self.point.ndim]

        for i, factor in enumerate(self.delta_factors):
            R = back.qr(factor)[1]
            einsum_str = (
                f"{core_letters},y{core_letters[i]}->"
                f"{core_letters[:i]}y{core_letters[i + 1:]}"
            )

            projected = back.einsum(einsum_str, self.delta_core, R)
            norms += back.norm(projected) ** 2

        return back.sqrt(norms)


def grad(
    f: Callable[[Tucker], float],
    X: Tucker,
    retain_graph: bool = False,
) -> Tuple[TangentVector, float]:
    """Compute the Riemannian gradient of function `f` at point `X`."""
    fx = None

    def h(delta_core, delta_factors):
        nonlocal fx
        tangent_X = TangentVector(X, delta_core, delta_factors).construct()
        fx = f(tangent_X)
        return fx

    dh = back.grad(h, [0, 1], retain_graph=retain_graph)
    dS, dV = dh(X.core, [back.zeros_like(f) for f in X.factors])

    core_letters = ascii_letters[: X.ndim]

    for i in range(X.ndim):
        dV[i] = dV[i] - X.factors[i] @ (X.factors[i].T @ dV[i])
        einsum_str = (
            f"{core_letters[:i]}X{core_letters[i+1:]},"
            f"{core_letters[:i]}Y{core_letters[i+1:]}->XY"
        )
        gram_core = back.einsum(einsum_str, X.core, X.core)

        dV[i] = back.cho_solve(dV[i].T, back.cho_factor(gram_core)[0]).T

    return TangentVector(X, dS, dV), fx


def project(X: Tucker, xi: Tucker, retain_graph: bool = False) -> TangentVector:
    """Project `xi` onto the tangent space of `X`."""

    def f(x):
        return x.flat_inner(xi)

    return grad(f, X, retain_graph)[0]
