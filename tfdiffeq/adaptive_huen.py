# Based on https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/integrate
import tensorflow as tf
from tfdiffeq.misc import (
    _scaled_dot_product, _convert_to_tensor, _is_finite, _select_initial_step, _handle_unused_kwargs, _is_iterable,
    _optimal_step_size, _compute_error_ratio, move_to_device
)
from tfdiffeq.solvers import AdaptiveStepsizeODESolver
from tfdiffeq.interp import _interp_fit, _interp_evaluate
from tfdiffeq.rk_common import _RungeKuttaState, _ButcherTableau, _runge_kutta_step

_ADAPTIVE_HEUN_TABLEAU = _ButcherTableau(
    alpha=[1.],
    beta=[
        [1.],
    ],
    c_sol=[0.5, 0.5],
    c_error=[
        0.5,
        -0.5,
    ],
)

AH_C_MID = [
    0.5, 0.
]


def _interp_fit_adaptive_heun(y0, y1, k, dt, tableau=_ADAPTIVE_HEUN_TABLEAU):
    """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
    dt = tf.cast(dt, y0[0].dtype)
    y_mid = tuple(y0_ + _scaled_dot_product(dt, AH_C_MID, k_) for y0_, k_ in zip(y0, k))
    f0 = tuple(k_[0] for k_ in k)
    f1 = tuple(k_[-1] for k_ in k)
    return _interp_fit(y0, y1, y_mid, f0, f1, dt)


def _abs_square(x):
    return tf.multiply(x, x)


def _ta_append(list_of_tensors, value):
    """Append a value to the end of a list of PyTorch tensors."""
    list_of_tensors.append(value)
    return list_of_tensors


class AdaptiveHeunSolver(AdaptiveStepsizeODESolver):

    def __init__(
            self, func, y0, rtol, atol, first_step=None, safety=0.9, ifactor=10.0, dfactor=0.2,
            max_num_steps=2 ** 31 - 1,
            **unused_kwargs
    ):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
        self.atol = atol if _is_iterable(atol) else [atol] * len(y0)
        self.first_step = first_step
        self.safety = _convert_to_tensor(safety, dtype=tf.float64, device=y0[0].device)
        self.ifactor = _convert_to_tensor(ifactor, dtype=tf.float64, device=y0[0].device)
        self.dfactor = _convert_to_tensor(dfactor, dtype=tf.float64, device=y0[0].device)
        self.max_num_steps = _convert_to_tensor(max_num_steps, dtype=tf.int32, device=y0[0].device)

    def before_integrate(self, t):
        f0 = self.func(tf.cast(t[0], self.y0[0].dtype), self.y0)
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, 1, self.rtol[0], self.atol[0], f0=f0)
            first_step = move_to_device(tf.cast(first_step, t.dtype), t.device)
        else:
            first_step = _convert_to_tensor(self.first_step, dtype=t.dtype, device=t.device)
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, interp_coeff=[self.y0] * 5)

    def advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_heun_step(self.rk_state)
            n_steps += 1
        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)

    def _adaptive_heun_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state
        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        for y0_ in y0:
            assert _is_finite(tf.math.abs(y0_)), 'non-finite values in state `y`: {}'.format(y0_)
        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, tableau=_ADAPTIVE_HEUN_TABLEAU)

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        # print("y error", y1_error[0].numpy())
        mean_sq_error_ratio = _compute_error_ratio(y1_error, atol=self.atol, rtol=self.rtol, y0=y0, y1=y1)
        # print("mean sq error ratio", mean_sq_error_ratio[0].numpy())
        accept_step = tf.reduce_all(tf.convert_to_tensor(mean_sq_error_ratio, dtype=tf.float64) <= 1.)

        ########################################################
        #                   Update RK State                    #
        ########################################################
        y_next = y1 if accept_step else y0
        f_next = f1 if accept_step else f0
        t_next = t0 + dt if accept_step else t0
        interp_coeff = _interp_fit_adaptive_heun(y0, y1, k, dt) if accept_step else interp_coeff
        dt_next = _optimal_step_size(
            dt, mean_sq_error_ratio, safety=self.safety, ifactor=self.ifactor, dfactor=self.dfactor, order=5
        )
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state
