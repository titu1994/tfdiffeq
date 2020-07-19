import unittest

import tensorflow as tf
import tfdiffeq

from tests import problems
from tests.check_grad import gradcheck

eps = 1e-12

# torch.set_default_dtype(torch.float64)
TEST_DEVICE = "gpu:0" if tf.test.is_gpu_available() else "cpu:0"


def max_abs(tensor):
    return tf.reduce_max(tf.abs(tensor))


class TestGradient(unittest.TestCase):

    def test_huen(self):
        f, y0, t_points, _ = problems.construct_problem(TEST_DEVICE)

        func = lambda y0, t_points: tfdiffeq.odeint(f, y0, t_points, method='huen')
        self.assertTrue(gradcheck(func, (y0, t_points)))

    def test_adaptive_heun(self):
        f, y0, t_points, _ = problems.construct_problem(TEST_DEVICE)

        func = lambda y0, t_points: tfdiffeq.odeint(f, y0, t_points, method='adaptive_heun')
        self.assertTrue(gradcheck(func, (y0, t_points)))

    def test_bosh3(self):
        f, y0, t_points, _ = problems.construct_problem(TEST_DEVICE)

        func = lambda y0, t_points: tfdiffeq.odeint(f, y0, t_points, method='bosh3')
        self.assertTrue(gradcheck(func, (y0, t_points)))

    def test_midpoint(self):
        f, y0, t_points, _ = problems.construct_problem(TEST_DEVICE)

        func = lambda y0, t_points: tfdiffeq.odeint(f, y0, t_points, method='midpoint')
        self.assertTrue(gradcheck(func, (y0, t_points)))

    def test_rk4(self):
        f, y0, t_points, _ = problems.construct_problem(TEST_DEVICE)

        func = lambda y0, t_points: tfdiffeq.odeint(f, y0, t_points, method='rk4')
        self.assertTrue(gradcheck(func, (y0, t_points)))

    def test_dopri5(self):
        f, y0, t_points, _ = problems.construct_problem(TEST_DEVICE)

        func = lambda y0, t_points: tfdiffeq.odeint(f, y0, t_points, method='dopri5')
        self.assertTrue(gradcheck(func, (y0, t_points)))

    def test_dopri8(self):
        f, y0, t_points, _ = problems.construct_problem(TEST_DEVICE)

        func = lambda y0, t_points: tfdiffeq.odeint(f, y0, t_points, method='dopri8')
        self.assertTrue(gradcheck(func, (y0, t_points)))

    def test_adams(self):
        f, y0, t_points, _ = problems.construct_problem(TEST_DEVICE)

        func = lambda y0, t_points: tfdiffeq.odeint(f, y0, t_points, method='adams')
        self.assertTrue(gradcheck(func, (y0, t_points)))

    def test_adjoint(self):
        """
        Test against dopri5
        """
        tf.compat.v1.set_random_seed(0)
        f, y0, t_points, _ = problems.construct_problem(TEST_DEVICE)
        y0 = tf.cast(y0, tf.float64)
        t_points = tf.cast(t_points, tf.float64)

        func = lambda y0, t_points: tfdiffeq.odeint(f, y0, t_points, method='dopri5')

        with tf.GradientTape() as tape:
            tape.watch(t_points)
            ys = func(y0, t_points)

        reg_t_grad, reg_a_grad, reg_b_grad = tape.gradient(ys, [t_points, f.a, f.b])

        f, y0, t_points, _ = problems.construct_problem(TEST_DEVICE)
        y0 = tf.cast(y0, tf.float64)
        t_points = tf.cast(t_points, tf.float64)

        y0 = (y0,)

        func = lambda y0, t_points: tfdiffeq.odeint_adjoint(f, y0, t_points, method='dopri5')

        with tf.GradientTape() as tape:
            tape.watch(t_points)
            ys = func(y0, t_points)

        grads = tape.gradient(ys, [t_points, f.a, f.b])
        adj_t_grad, adj_a_grad, adj_b_grad = grads

        self.assertLess(max_abs(reg_t_grad - adj_t_grad), 1.2e-7)
        self.assertLess(max_abs(reg_a_grad - adj_a_grad), 1.2e-7)
        self.assertLess(max_abs(reg_b_grad - adj_b_grad), 1.2e-7)


class TestCompareAdjointGradient(unittest.TestCase):

    def problem(self):
        tf.keras.backend.set_floatx('float64')

        class Odefunc(tf.keras.Model):

            def __init__(self):
                super(Odefunc, self).__init__()
                self.A = tf.Variable([[-0.1, -2.0], [2.0, -0.1]], dtype=tf.float64)
                self.unused_module = tf.keras.layers.Dense(5, dtype=tf.float64)
                self.unused_module.build((5,))

            def call(self, t, y):
                y = tfdiffeq.cast_double(y)
                return tf.linalg.matvec(self.A, y ** 3)

        y0 = tf.convert_to_tensor([2., 0.], dtype=tf.float64)
        t_points = tf.linspace(
            tf.constant(0., dtype=tf.float64),
            tf.constant(25., dtype=tf.float64),
            10
        )
        func = Odefunc()
        return func, y0, t_points

    def test_dopri5_adjoint_against_dopri5(self):
        tf.keras.backend.set_floatx('float64')
        tf.compat.v1.set_random_seed(0)
        with tf.GradientTape(persistent=True) as tape:
            func, y0, t_points = self.problem()
            tape.watch(t_points)
            tape.watch(y0)
            ys = tfdiffeq.odeint_adjoint(func, y0, t_points, method='dopri5')

        gradys = 0.1 * tf.random.uniform(shape=ys.shape, dtype=tf.float64)
        adj_y0_grad, adj_t_grad, adj_A_grad = tape.gradient(
            ys,
            [y0, t_points, func.A],
            output_gradients=gradys
        )

        w_grad, b_grad = tape.gradient(ys, func.unused_module.variables)
        self.assertIsNone(w_grad)
        self.assertIsNone(b_grad)

        with tf.GradientTape() as tape:
            func, y0, t_points = self.problem()
            tape.watch(y0)
            tape.watch(t_points)
            ys = tfdiffeq.odeint(func, y0, t_points, method='dopri5')

        y_grad, t_grad, a_grad = tape.gradient(
            ys,
            [y0, t_points, func.A],
            output_gradients=gradys
        )

        self.assertLess(max_abs(y_grad - adj_y0_grad), 3e-4)
        self.assertLess(max_abs(t_grad - adj_t_grad), 1e-4)
        self.assertLess(max_abs(a_grad - adj_A_grad), 2e-3)

    # def test_adams_adjoint_against_dopri5(self):
    #    tf.keras.backend.set_floatx('float64')
    #    tf.compat.v1.set_random_seed(0)
    #    with tf.GradientTape(persistent=True) as tape:
    #        func, y0, t_points = self.problem()
    #        tape.watch(t_points)
    #        tape.watch(y0)
    #        ys = tfdiffeq.odeint_adjoint(func, y0, t_points, method='adams')

    #    gradys = 0.1 * tf.random.uniform(shape=ys.shape, dtype=tf.float64)
    #    adj_y0_grad, adj_t_grad, adj_A_grad = tape.gradient(
    #          ys,
    #          [y0, t_points, func.A],
    #          output_gradients=gradys
    #    )

    #    with tf.GradientTape() as tape:
    #        func, y0, t_points = self.problem()
    #        tape.watch(y0)
    #        tape.watch(t_points)
    #        ys = tfdiffeq.odeint(func, y0, t_points, method='dopri5')

    #    y_grad, t_grad, a_grad = tape.gradient(
    #        ys,
    #        [y0, t_points, func.A],
    #        output_gradients=gradys
    #    )

    #    self.assertLess(max_abs(y_grad - adj_y0_grad), 3e-4)
    #    self.assertLess(max_abs(t_grad - adj_t_grad), 1e-4)
    #    self.assertLess(max_abs(a_grad - adj_A_grad), 2e-3)


if __name__ == '__main__':
    unittest.main()
