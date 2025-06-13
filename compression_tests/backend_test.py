import unittest
from unittest import TestCase

from compress import set_backend
from compression_tests.tucker_tests import tucker_test


class BackendTest(TestCase):
    def testPytorchBackend(self):
        set_backend("pytorch")
        instance = tucker_test.TuckerTensorTest()
        instance.testFull2Tuck()
        instance.testAdd()
        instance.testMul()
        instance.testNorm()
        instance.testModeProd()

    def testBackend(self):
        self.testPytorchBackend()


if __name__ == "__main__":
    unittest.main()
