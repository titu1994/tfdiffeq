import abc

import tensorflow as tf

from tfdiffeq.misc import (_assert_increasing, _handle_unused_kwargs,
                           move_to_device, cast_double, func_cast_double)


class AdaptiveStepsizeODESolver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, atol, rtol, **unused_kwargs):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.atol = atol
        self.rtol = rtol

    def before_integrate(self, t):
        pass

    @abc.abstractmethod
    def advance(self, next_t):
        raise NotImplementedError

    def integrate(self, t):
        _assert_increasing(t)
        solution = [cast_double(self.y0)]
        t = move_to_device(tf.cast(t, tf.float64), self.y0[0].device)
        self.before_integrate(t)
        for i in range(1, t.shape[0]):
            y = self.advance(t[i])
            y = cast_double(y)
            solution.append(y)
        return tuple(map(tf.stack, tuple(zip(*solution))))


class FixedGridODESolver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, step_size=None, grid_constructor=None, eps=0.0, **unused_kwargs):
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('atol', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.eps = eps

        if step_size is not None and grid_constructor is None:
            self.grid_constructor = self._grid_constructor_from_step_size(step_size)
        elif grid_constructor is None:
            self.grid_constructor = lambda f, y0, t: t
        else:
            raise ValueError("step_size and grid_constructor are exclusive arguments.")

    def _grid_constructor_from_step_size(self, step_size):

        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = tf.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = move_to_device(tf.range(0, niters), t) * step_size + start_time
            if t_infer[-1] > t[-1]:
                t_infer[-1] = t[-1]

            return t_infer

        return _grid_constructor

    @property
    @abc.abstractmethod
    def order(self):
        pass

    @abc.abstractmethod
    def step_func(self, func, t, dt, y):
        pass

    def integrate(self, t):
        _assert_increasing(t)
        t = tf.cast(t, self.y0[0].dtype)
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert tf.equal(time_grid[0], t[0]) and tf.equal(time_grid[-1], t[-1])
        time_grid = move_to_device(time_grid, self.y0[0].device)

        solution = [cast_double(self.y0)]

        j = 1
        y0 = cast_double(self.y0)
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dy = self.step_func(self.func, t0, t1 - t0, y0)
            y1 = tuple(y0_ + dy_ for y0_, dy_ in zip(y0, dy))

            while j < t.shape[0] and t1 >= t[j]:
                y = self._linear_interp(t0, t1, y0, y1, t[j])
                solution.append(y)
                j += 1

            y0 = y1

        return tuple(map(tf.stack, tuple(zip(*solution))))

    @func_cast_double
    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        t0 = move_to_device(t0, y0[0].device)
        t1 = move_to_device(t1, y0[0].device)
        t = move_to_device(t, y0[0].device)
        slope = tuple((y1_ - y0_) / (t1 - t0) for y0_, y1_, in zip(y0, y1))
        return tuple(y0_ + slope_ * (t - t0) for y0_, slope_ in zip(y0, slope))
