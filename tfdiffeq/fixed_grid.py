import tensorflow as tf
from tfdiffeq.rk_common import rk4_alt_step_func
from tfdiffeq.solvers import FixedGridODESolver


class Euler(FixedGridODESolver):
    order = 1

    def __init__(self, eps=0., **kwargs):
        super(Euler, self).__init__(**kwargs)

        with tf.device(self.device):
            self.eps = tf.convert_to_tensor(eps, dtype=self.dtype)

    def _step_func(self, func, t, dt, y):
        return dt * func(t + self.eps, y)


class Midpoint(FixedGridODESolver):
    order = 2

    def __init__(self, eps=0., **kwargs):
        super(Midpoint, self).__init__(**kwargs)

        with tf.device(self.device):
            self.eps = tf.convert_to_tensor(eps, dtype=self.dtype)

    def _step_func(self, func, t, dt, y):
        half_dt = 0.5 * dt
        y_mid = y + func(t + self.eps, y) * half_dt
        return dt * func(t + half_dt, y_mid)

        # y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t + self.eps, y)))
        # return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid))


class Heun(FixedGridODESolver):
    order = 2

    def __init__(self, eps=0., **kwargs):
        super(Heun, self).__init__(**kwargs)

        with tf.device(self.device):
            self.eps = tf.convert_to_tensor(eps, dtype=self.dtype)

    def _step_func(self, func, t, dt, y):
        f_outs = func(t + self.eps, y)
        ft_1_hat = y + dt * f_outs
        ft_1_outs = func(t + dt, ft_1_hat)
        return dt / 2.0 * (f_outs + ft_1_outs)


class RK4(FixedGridODESolver):
    order = 4

    def __init__(self, eps=0., **kwargs):
        super(RK4, self).__init__(**kwargs)

        with tf.device(self.device):
            self.eps = tf.convert_to_tensor(eps, dtype=self.dtype)

    def _step_func(self, func, t, dt, y):
        return rk4_alt_step_func(func, t + self.eps, dt - 2 * self.eps, y)
