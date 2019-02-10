import unittest

import tensorflow as tf

import tfdiffeq
from tests import problems

if not tf.executing_eagerly():
    tf.enable_eager_execution()

error_tol = 1e-4

# torch.set_default_dtype(torch.float64)
TEST_DEVICE = "gpu:0" if tf.test.is_gpu_available() else "cpu"


def max_abs(tensor):
    return tf.reduce_max(tf.abs(tensor))


def rel_error(true, estimate):
    return max_abs((true - estimate) / true)


class TestSolverError(unittest.TestCase):

    def test_euler(self):
        f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE)

        y = tfdiffeq.odeint(f, y0, t_points, method='euler')
        self.assertLess(rel_error(sol, y), error_tol)

    def test_midpoint(self):
        f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE)

        y = tfdiffeq.odeint(f, y0, t_points, method='midpoint')
        self.assertLess(rel_error(sol, y), error_tol)

    def test_rk4(self):
        f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE)

        y = tfdiffeq.odeint(f, y0, t_points, method='rk4')
        self.assertLess(rel_error(sol, y), error_tol)

    def test_explicit_adams(self):
        f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE)

        y = tfdiffeq.odeint(f, y0, t_points, method='explicit_adams')
        self.assertLess(rel_error(sol, y), error_tol)

    def test_adams(self):
        for ode in problems.PROBLEMS.keys():
            f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, ode=ode)
            y = tfdiffeq.odeint(f, y0, t_points, method='adams')
            with self.subTest(ode=ode):
                self.assertLess(rel_error(sol, y), error_tol)

    def test_dopri5(self):
        for ode in problems.PROBLEMS.keys():
            f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, ode=ode)
            y = tfdiffeq.odeint(f, y0, t_points, method='dopri5')
            with self.subTest(ode=ode):
                self.assertLess(rel_error(sol, y), error_tol)

    # def test_adjoint(self):
    #     for ode in problems.PROBLEMS.keys():
    #         f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True)
    #
    #         y = tfdiffeq.odeint_adjoint(f, y0, t_points, method='dopri5')
    #         with self.subTest(ode=ode):
    #             self.assertLess(rel_error(sol, y), error_tol)


class TestSolverBackwardsInTimeError(unittest.TestCase):

    def test_euler(self):
        f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True)

        y = tfdiffeq.odeint(f, y0, t_points, method='euler')
        self.assertLess(rel_error(sol, y), error_tol)

    def test_midpoint(self):
        f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True)

        y = tfdiffeq.odeint(f, y0, t_points, method='midpoint')
        self.assertLess(rel_error(sol, y), error_tol)

    def test_rk4(self):
        f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True)

        y = tfdiffeq.odeint(f, y0, t_points, method='rk4')
        self.assertLess(rel_error(sol, y), error_tol)

    def test_explicit_adams(self):
        f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True)

        y = tfdiffeq.odeint(f, y0, t_points, method='explicit_adams')
        self.assertLess(rel_error(sol, y), error_tol)

    def test_adams(self):
        for ode in problems.PROBLEMS.keys():
            f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True)

            y = tfdiffeq.odeint(f, y0, t_points, method='adams')
            with self.subTest(ode=ode):
                self.assertLess(rel_error(sol, y), error_tol)

    def test_dopri5(self):
        for ode in problems.PROBLEMS.keys():
            f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True)

            y = tfdiffeq.odeint(f, y0, t_points, method='dopri5')
            with self.subTest(ode=ode):
                self.assertLess(rel_error(sol, y), error_tol)

    # def test_adjoint(self):
    #     for ode in problems.PROBLEMS.keys():
    #         f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True)
    #
    #         y = tfdiffeq.odeint_adjoint(f, y0, t_points, method='dopri5')
    #         with self.subTest(ode=ode):
    #             self.assertLess(rel_error(sol, y), error_tol)


class TestNoIntegration(unittest.TestCase):

    def test_midpoint(self):
        f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True)

        y = tfdiffeq.odeint(f, y0, t_points[0:1], method='midpoint')
        self.assertLess(max_abs(sol[0] - y), error_tol)

    def test_rk4(self):
        f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True)

        y = tfdiffeq.odeint(f, y0, t_points[0:1], method='rk4')
        self.assertLess(max_abs(sol[0] - y), error_tol)

    def test_explicit_adams(self):
        f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True)

        y = tfdiffeq.odeint(f, y0, t_points[0:1], method='explicit_adams')
        self.assertLess(max_abs(sol[0] - y), error_tol)

    def test_adams(self):
        f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True)

        y = tfdiffeq.odeint(f, y0, t_points[0:1], method='adams')
        self.assertLess(max_abs(sol[0] - y), error_tol)

    def test_dopri5(self):
        f, y0, t_points, sol = problems.construct_problem(TEST_DEVICE, reverse=True)

        y = tfdiffeq.odeint(f, y0, t_points[0:1], method='dopri5')
        self.assertLess(max_abs(sol[0] - y), error_tol)


if __name__ == '__main__':
    tf.enable_eager_execution()
    unittest.main()
