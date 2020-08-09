import unittest

import tensorflow as tf
import tfdiffeq

from tests import problems
from tests.check_grad import gradcheck

eps = 1e-12

tf.keras.backend.set_floatx('float64')
TEST_DEVICE = "gpu:0" if tf.test.is_gpu_available() else "cpu:0"


def max_abs(tensor):
    return tf.reduce_max(tf.abs(tensor))


class TestGradient(unittest.TestCase):

    def test_odeint(self):
        for device in problems.DEVICES:
            for method in problems.METHODS:
                with self.subTest(device=device, method=method):
                    f, y0, t_points, _ = problems.construct_problem(device=device)
                    func = lambda y0, t_points: tfdiffeq.odeint(f, y0, t_points, method=method)
                    self.assertTrue(gradcheck(func, (y0, t_points)))

    def test_adjoint(self):
        for device in problems.DEVICES:
            for ode in problems.PROBLEMS:
                if ode == 'constant':
                    eps = 1e-12
                elif ode == 'linear':
                    eps = 1e-5
                elif ode == 'sine':
                    eps = 5e-3
                else:
                    raise RuntimeError

                with self.subTest(device=device, ode=ode):
                    f, y0, t_points, _ = problems.construct_problem(device=device, ode=ode)

                    with tf.GradientTape() as tape:
                        tape.watch(y0)
                        tape.watch(t_points)
                        ys = tfdiffeq.odeint(f, y0, t_points, rtol=1e-9, atol=1e-12, method='dopri5')

                    reg_t_grad, reg_y0_grad, reg_param_grad = tape.gradient(ys, [t_points, y0, *f.trainable_variables])

                    with tf.GradientTape() as tape:
                        tape.watch(y0)
                        tape.watch(t_points)
                        ys = tfdiffeq.odeint_adjoint(f, y0, t_points, rtol=1e-9, atol=1e-12, method='dopri5')

                    adj_t_grad, adj_y0_grad, adj_param_grad = tape.gradient(ys, [t_points, y0, *f.trainable_variables])

                    self.assertLess(max_abs(reg_t_grad - adj_t_grad), eps)
                    self.assertLess(max_abs(reg_y0_grad - adj_y0_grad), eps)

                    for reg_grad, adj_grad in zip(reg_param_grad, adj_param_grad):
                        self.assertLess(max_abs(reg_grad - adj_grad), eps)


class TestCompareAdjointGradient(unittest.TestCase):

    def problem(self, device):
        tf.keras.backend.set_floatx('float64')

        class Odefunc(tf.keras.Model):

            def __init__(self):
                super(Odefunc, self).__init__()
                self.A = tf.Variable([[-0.1, -2.0], [2.0, -0.1]], dtype=tf.float64)
                self.unused_module = tf.keras.layers.Dense(5, dtype=tf.float64)
                self.unused_module.build((5,))

            def call(self, t, y):
                return tf.linalg.matvec(self.A, y ** 3)

        with tf.device(device):
            y0 = tf.convert_to_tensor([2., 0.], dtype=tf.float64)
            t_points = tf.linspace(
                tf.constant(0., dtype=tf.float64),
                tf.constant(25., dtype=tf.float64),
                10
            )
            func = Odefunc()
        return func, y0, t_points

    def test_against_dopri5(self):
        method_eps = {'dopri5': (3e-4, 1e-4, 2e-3)}  # TODO: add in adaptive adams if/when it's fixed.
        for device in problems.DEVICES:
            for method, eps in method_eps.items():
                for t_grad in (True, False):
                    with self.subTest(device=device, method=method):
                        func, y0, t_points = self.problem(device=device)

                        with tf.GradientTape(persistent=True) as tape:
                            tape.watch(y0)
                            tape.watch(t_points)
                            ys = tfdiffeq.odeint_adjoint(func, y0, t_points, method=method)

                        tf.random.set_seed(0)
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
                            tape.watch(y0)
                            tape.watch(t_points)
                            ys = tfdiffeq.odeint(func, y0, t_points, method='dopri5')

                        y_grad, t_grad, a_grad = tape.gradient(
                            ys,
                            [y0, t_points, func.A],
                            output_gradients=gradys
                        )

                        self.assertLess(max_abs(y_grad - adj_y0_grad), eps[0])
                        self.assertLess(max_abs(t_grad - adj_t_grad), eps[1])
                        self.assertLess(max_abs(a_grad - adj_A_grad), eps[2])

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
