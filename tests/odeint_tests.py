import unittest

import tensorflow as tf

import tfdiffeq
from tests import problems


def max_abs(tensor):
    return tf.reduce_max(tf.abs(tensor))


def rel_error(true, estimate):
    return max_abs((true - estimate) / true)


class TestSolverError(unittest.TestCase):

    def test_odeint(self):
        for reverse in (False, True):
            for dtype in problems.DTYPES:
                for device in problems.DEVICES:
                    for method in problems.METHODS:
                        if dtype == tf.float32 and method == 'dopri8':
                            continue

                        kwargs = dict(rtol=1e-12, atol=1e-14) if method == 'dopri8' else dict()
                        ode_problems = problems.PROBLEMS if method in problems.ADAPTIVE_METHODS else ('constant',)

                        for ode in ode_problems:
                            if method == 'adaptive_heun':
                                eps = 4e-3
                            elif method == 'bosh3':
                                eps = 3e-3
                            elif ode == 'linear':
                                eps = 2e-3
                            else:
                                eps = 1e-4

                            with self.subTest(reverse=reverse, dtype=dtype, device=device, ode=ode, method=method):
                                f, y0, t_points, sol = problems.construct_problem(dtype=dtype, device=device, ode=ode,
                                                                                  reverse=reverse)
                                y = tfdiffeq.odeint(f, y0, t_points, method=method, **kwargs)
                                self.assertLess(rel_error(sol, y), eps)

    def test_adjoint(self):
        for reverse in (False, True):
            for dtype in problems.DTYPES:
                for device in problems.DEVICES:
                    for ode in problems.PROBLEMS:
                        if ode == 'linear':
                            eps = 2e-3
                        else:
                            eps = 1e-4

                        with self.subTest(reverse=reverse, dtype=dtype, device=device, ode=ode):
                            f, y0, t_points, sol = problems.construct_problem(dtype=dtype, device=device, ode=ode,
                                                                              reverse=reverse)
                            y = tfdiffeq.odeint_adjoint(f, y0, t_points)
                            self.assertLess(rel_error(sol, y), eps)


class TestSolverBackwardsInTimeError(unittest.TestCase):

    def test_odeint(self):
        for reverse in (False, True):
            for dtype in problems.DTYPES:
                for device in problems.DEVICES:
                    for method in problems.METHODS:
                        for ode in problems.PROBLEMS:

                            with self.subTest(reverse=reverse, dtype=dtype, device=device, ode=ode, method=method):
                                f, y0, t_points, sol = problems.construct_problem(dtype=dtype, device=device, ode=ode,
                                                                                  reverse=reverse)
                                y = tfdiffeq.odeint(f, y0, t_points[0:1], method=method)
                                self.assertLess((sol[0] - y).abs().max(), 1e-12)



if __name__ == '__main__':
    tf.enable_eager_execution()
    unittest.main()
