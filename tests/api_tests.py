import sys
import unittest
import tensorflow as tf

import tfdiffeq

sys.path.insert(0, '..')
from tests.problems import construct_problem, DTYPES, DEVICES, ADAPTIVE_METHODS
from tests.check_grad import gradcheck

# torch.set_default_dtype(torch.float64)
EPS = {tf.float32: 1e-5, tf.float64: 1e-12}
TEST_DEVICE = "gpu:0" if tf.test.is_gpu_available() else "cpu:0"


def max_abs(tensor):
    return tf.reduce_max(tf.abs(tensor))


class TestCollectionState(unittest.TestCase):

    def test_forward(self):
        for dtype in DTYPES:
            eps = EPS[dtype]
            for device in DEVICES:
                f, y0, t_points, sol = construct_problem(dtype=dtype, device=device)
                tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
                tuple_y0 = (y0, y0)
                for method in ADAPTIVE_METHODS:

                    with self.subTest(dtype=dtype, device=device, method=method):
                        tuple_y = tfdiffeq.odeint(tuple_f, tuple_y0, t_points, method=method, options={'dtype': dtype})
                        print(sol, tuple_y[0])
                        max_error0 = tf.reduce_max(sol - tuple_y[0])
                        max_error1 = tf.reduce_max(sol - tuple_y[1])
                        self.assertLess(max_error0, eps)
                        self.assertLess(max_error1, eps)

    def test_gradient(self):
        for device in DEVICES:
            f, y0, t_points, sol = construct_problem(device=device)
            tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
            for method in ADAPTIVE_METHODS:

                with self.subTest(device=device, method=method):
                    for i in range(2):
                        func = lambda y0, t_points: tfdiffeq.odeint(tuple_f, (y0, y0), t_points, method=method)[i]
                        self.assertTrue(gradcheck(func, (y0, t_points)))


if __name__ == '__main__':
    unittest.main(__file__)
