import unittest
from unittest import TestCase

import numpy as np

from compress import Tucker, TuckerRiemannian
from compress import backend as back
from compress import set_backend
from compress.tucker_decomposition.matrix import TuckerMatrix


class RiemannianTest(TestCase):
    def createTestTensor(self, n=4):
        """
        A = [G; U, V, V], V = ones(n x n)
        """
        common_factor = np.random.randn(n, n)
        common_factor = back.tensor(common_factor)
        common_factor = back.qr(common_factor)[0]
        symmetric_factor = back.tensor(np.random.randn(n, n))
        symmetric_factor = back.qr(symmetric_factor)[0]
        core = back.tensor(np.random.randn(n, n, n))
        return Tucker(core, [common_factor, symmetric_factor, symmetric_factor])

    @staticmethod
    def f(T: Tucker):
        A = T.to_dense()
        return (A**2 - A).sum()

    def testGradProjection(self):
        set_backend("pytorch")
        np.random.seed(229)

        def f_full(A):
            return (A**2 - A).sum()

        full_grad = back.grad(f_full, argnums=0)

        T = self.createTestTensor(4)

        eucl_grad = full_grad(T.to_dense())
        riem_grad, _ = TuckerRiemannian.grad(self.f, T)
        riem_grad = riem_grad.construct()

        assert np.allclose(
            back.to_numpy(eucl_grad), back.to_numpy(riem_grad.to_dense()), atol=1e-5
        )

    def testProject(self):
        np.random.seed(229)

        T = self.createTestTensor(4)
        tg_vector, _ = TuckerRiemannian.grad(self.f, T)
        tg_vector_proj = TuckerRiemannian.project(T, tg_vector.construct())
        assert np.allclose(
            back.to_numpy(tg_vector.construct().to_dense()),
            back.to_numpy(tg_vector_proj.construct().to_dense()),
            atol=1e-5,
        )

    def testAdd(self):
        np.random.seed(229)

        T = self.createTestTensor(4)
        tg_vector1, _ = TuckerRiemannian.grad(self.f, T)
        tg_vector2 = TuckerRiemannian.TangentVector(
            T,
            back.randn(T.core.shape),
            [back.randn(T.factors[0].shape) for _ in range(T.ndim)],
        )
        addition = tg_vector1 + tg_vector2
        dumb_addition = tg_vector1.construct() + tg_vector2.construct()
        assert (addition.construct() - dumb_addition).norm(
            qr_based=True
        ) / dumb_addition.norm(qr_based=True) <= 1e-6

    def testScalarMultiplication(self):
        np.random.seed(229)

        T = self.createTestTensor(4)
        tg_vector, _ = TuckerRiemannian.grad(self.f, T)
        rmul = 420 * tg_vector
        dumb_rmul = 420 * tg_vector.construct()
        assert (rmul.construct() - dumb_rmul).norm(qr_based=True) / dumb_rmul.norm(
            qr_based=True
        ) <= 1e-6

    def testMatrixGrad(self):
        eye = back.ones((8, 8))
        matrix = back.copy(eye)
        eye = back.reshape(eye, (2, 2, 2, 2, 2, 2))
        eye = back.transpose(eye, (0, 3, 1, 4, 2, 5))
        eye = back.reshape(eye, (4, 4, 4))
        eye = TuckerMatrix.from_dense(eye, (2, 2, 2), (2, 2, 2), eps=1e-7)
        x = back.ones(8)
        x = back.reshape(x, (2, 2, 2))
        x_dense = back.reshape(x, (8,))

        def loss(A):
            return back.norm(A @ x) ** 2

        def loss_dense(A):
            return back.norm(A @ x_dense) ** 2

        eucl_grad = back.grad(loss_dense, argnums=0)(matrix)
        riem_grad, _ = TuckerRiemannian.grad(loss, eye)
        riem_grad = riem_grad.construct()

        riem_grad = riem_grad.to_dense()
        riem_grad = back.reshape(riem_grad, (2, 2, 2, 2, 2, 2))
        riem_grad = back.transpose(riem_grad, (0, 2, 4, 1, 3, 5))
        riem_grad = back.reshape(riem_grad, (8, 8))

        assert np.allclose(
            back.to_numpy(eucl_grad), back.to_numpy(riem_grad.to_dense()), atol=1e-5
        )

    def testNorm(self):
        T = self.createTestTensor(4)
        tg_vector1, _ = TuckerRiemannian.grad(self.f, T)
        true_norm = tg_vector1.construct().norm(qr_based=True)
        computed_norm = tg_vector1.norm()

        assert (true_norm - computed_norm) < 1e-5


if __name__ == "__main__":
    unittest.main()
