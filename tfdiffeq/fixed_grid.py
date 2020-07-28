from tfdiffeq import rk_common
from tfdiffeq.solvers import FixedGridODESolver

class Euler(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return tuple(dt * f_ for f_ in func(t + self.eps, y))

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t + self.eps, y)))
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid))


    @property
    def order(self):
        return 2


class Heun(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        f_outs = func(t + self.eps, y)
        ft_1_hat = tuple(y_ + dt * f_ for y_, f_ in zip(y, f_outs))
        ft_1_outs = func(t + dt, ft_1_hat)
        return tuple(dt / 2. * (ft_ + ft_1_hat_) for ft_, ft_1_hat_ in zip(f_outs, ft_1_outs))

    @property
    def order(self):
        return 2


class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return rk_common.rk4_alt_step_func(func, t + self.eps, dt, y)

    @property
    def order(self):
        return 4
