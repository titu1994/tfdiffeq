import sys
import unittest
import tensorflow as tf

import tfdiffeq

sys.path.insert(0, '..')
from tests.problems import construct_problem
from tests.check_grad import gradcheck

if not tf.executing_eagerly():
    tf.enable_v2_behavior()

eps = 1e-5

# torch.set_default_dtype(torch.float64)
TEST_DEVICE = "gpu:0" if tf.test.is_gpu_available() else "cpu:0"


def max_abs(tensor):
    return tf.reduce_max(tf.abs(tensor))


class TestCollectionState(unittest.TestCase):

    def test_dopri5(self):
        f, y0, t_points, sol = construct_problem(TEST_DEVICE)

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
        tuple_y0 = (y0, y0)

        tuple_y = tfdiffeq.odeint(tuple_f, tuple_y0, t_points, method='dopri5')
        max_error0 = tf.reduce_max(sol - tuple_y[0])
        max_error1 = tf.reduce_max(sol - tuple_y[1])
        self.assertLess(max_error0, eps)
        self.assertLess(max_error1, eps)

    def test_dopri5_gradient(self):
        f, y0, t_points, sol = construct_problem(TEST_DEVICE)

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))

        for i in range(2):
            func = lambda y0, t_points: tfdiffeq.odeint(tuple_f, (y0, y0), t_points, method='dopri5')[i]
            self.assertTrue(gradcheck(func, (y0, t_points)))

    def test_dopri8(self):
        f, y0, t_points, sol = construct_problem(TEST_DEVICE)

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
        tuple_y0 = (y0, y0)

        tuple_y = tfdiffeq.odeint(tuple_f, tuple_y0, t_points, method='dopri8')
        max_error0 = tf.reduce_max(sol - tuple_y[0])
        max_error1 = tf.reduce_max(sol - tuple_y[1])
        self.assertLess(max_error0, eps)
        self.assertLess(max_error1, eps)

    def test_dopri8_gradient(self):
        f, y0, t_points, sol = construct_problem(TEST_DEVICE)

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))

        for i in range(2):
            func = lambda y0, t_points: tfdiffeq.odeint(tuple_f, (y0, y0), t_points, method='dopri8')[i]
            self.assertTrue(gradcheck(func, (y0, t_points)))

    def test_adams(self):
        f, y0, t_points, sol = construct_problem(TEST_DEVICE)

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
        tuple_y0 = (y0, y0)

        tuple_y = tfdiffeq.odeint(tuple_f, tuple_y0, t_points, method='adams')
        max_error0 = tf.reduce_max(sol - tuple_y[0])
        max_error1 = tf.reduce_max(sol - tuple_y[1])
        self.assertLess(max_error0, eps)
        self.assertLess(max_error1, eps)

    def test_adams_gradient(self):
        f, y0, t_points, sol = construct_problem(TEST_DEVICE)

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))

        for i in range(2):
            func = lambda y0, t_points: tfdiffeq.odeint(tuple_f, (y0, y0), t_points, method='adams')[i]
            self.assertTrue(gradcheck(func, (y0, t_points)))

    def test_adaptive_heun(self):
        f, y0, t_points, sol = construct_problem(TEST_DEVICE)

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
        tuple_y0 = (y0, y0)

        tuple_y = tfdiffeq.odeint(tuple_f, tuple_y0, t_points, method='adaptive_heun')
        max_error0 = tf.reduce_max(sol - tuple_y[0])
        max_error1 = tf.reduce_max(sol - tuple_y[1])
        self.assertLess(max_error0, eps)
        self.assertLess(max_error1, eps)

    def test_adaptive_heun_gradient(self):
        f, y0, t_points, sol = construct_problem(TEST_DEVICE)

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))

        for i in range(2):
            func = lambda y0, t_points: tfdiffeq.odeint(tuple_f, (y0, y0), t_points, method='adaptive_heun')[i]
            self.assertTrue(gradcheck(func, (y0, t_points)))


if __name__ == '__main__':
    unittest.main(__file__)
