from tfdiffeq import rk_common
from tfdiffeq.misc import func_cast_double, cast_double
from tfdiffeq.solvers import FixedGridODESolver


class Euler(FixedGridODESolver):

    @func_cast_double
    def step_func(self, func, t, dt, y):
        return tuple(dt * f_ for f_ in cast_double(func(t, y)))

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):

    @func_cast_double
    def step_func(self, func, t, dt, y):
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, cast_double(func(t, y))))
        return tuple(dt * f_ for f_ in cast_double(func(t + dt / 2, y_mid)))

    @property
    def order(self):
        return 2


class Huen(FixedGridODESolver):

    @func_cast_double
    def step_func(self, func, t, dt, y):
        ft_1_hat = tuple(y_ + dt * f_ for y_, f_ in zip(y, cast_double(func(t, y))))
        return tuple(dt / 2. * (ft_ + ft_1_hat_) for ft_, ft_1_hat_ in zip(cast_double(func(t, y)),
                                                                           cast_double(func(t + dt,
                                                                                            ft_1_hat))))

    @property
    def order(self):
        return 2


class RK4(FixedGridODESolver):

    @func_cast_double
    def step_func(self, func, t, dt, y):
        return rk_common.rk4_alt_step_func(func, t, dt, y)

    @property
    def order(self):
        return 4
