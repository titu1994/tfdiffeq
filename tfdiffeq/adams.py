import collections

import tensorflow as tf

from tfdiffeq.misc import (
    _handle_unused_kwargs, _select_initial_step, _convert_to_tensor, _scaled_dot_product, _is_iterable,
    _optimal_step_size, _compute_error_ratio, move_to_device
)
from tfdiffeq.solvers import AdaptiveStepsizeODESolver
from tfdiffeq import compat

_MIN_ORDER = 1
_MAX_ORDER = 12

gamma_star = [
    1, -1 / 2, -1 / 12, -1 / 24, -19 / 720, -3 / 160, -863 / 60480, -275 / 24192, -33953 / 3628800, -0.00789255,
    -0.00678585, -0.00592406, -0.00523669, -0.0046775, -0.00421495, -0.0038269
]


class _VCABMState(collections.namedtuple('_VCABMState', 'y_n, prev_f, prev_t, next_t, phi, order')):
    """Saved state of the variable step size Adams-Bashforth-Moulton solver as described in

        Solving Ordinary Differential Equations I - Nonstiff Problems III.5
        by Ernst Hairer, Gerhard Wanner, and Syvert P Norsett.
    """


def g_and_explicit_phi(prev_t, next_t, implicit_phi, k):
    curr_t = prev_t[0]
    dt = next_t - prev_t[0]

    with tf.device(prev_t[0].device):
        g = tf.Variable(tf.zeros([k + 1]), trainable=False)

    explicit_phi = collections.deque(maxlen=k)
    beta = move_to_device(tf.convert_to_tensor(1.), prev_t[0].device)

    # tf.assign(g[0], 1)
    compat.assign(g[0], 1)

    c = 1 / move_to_device(tf.range(1, k + 2), prev_t[0].device)
    explicit_phi.append(implicit_phi[0])

    beta = tf.cast(beta, next_t.dtype)
    for j in range(1, k):
        beta = (next_t - prev_t[j - 1]) / (curr_t - prev_t[j]) * beta
        beta_cast = move_to_device(beta, implicit_phi[j][0].device)
        beta_cast = tf.cast(beta_cast, implicit_phi[0][0].dtype)
        explicit_phi.append(tuple(iphi_ * beta_cast for iphi_ in implicit_phi[j]))

        c = c[:-1] - c[1:] if j == 1 else c[:-1] - c[1:] * dt / (next_t - prev_t[j - 1])
        # tf.assign(g[j], tf.cast(c[0], g[j].dtype))
        compat.assign(g[j], tf.cast(c[0], g[j].dtype))
        # g[j] = c[0]

    c = c[:-1] - c[1:] * dt / (next_t - prev_t[k - 1])
    # tf.assign(g[k], tf.cast(c[0], g[k].dtype))
    compat.assign(g[k], tf.cast(c[0], g[k].dtype))

    return g, explicit_phi


def compute_implicit_phi(explicit_phi, f_n, k):
    k = min(len(explicit_phi) + 1, k)
    implicit_phi = collections.deque(maxlen=k)
    implicit_phi.append(f_n)

    def _typesafe_sub(iphi, ephi):
        if ephi.dtype != iphi.dtype:
            ephi = tf.cast(ephi, iphi.dtype)

        return iphi - ephi

    for j in range(1, k):
        implicit_phi.append(
            tuple(_typesafe_sub(iphi_, ephi_) for iphi_, ephi_ in zip(implicit_phi[j - 1], explicit_phi[j - 1])))
    return implicit_phi


class VariableCoefficientAdamsBashforth(AdaptiveStepsizeODESolver):

    def __init__(
            self, func, y0, rtol, atol, implicit=True, first_step=None, max_order=_MAX_ORDER, safety=0.9,
            ifactor=10.0, dfactor=0.2, **unused_kwargs
    ):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
        self.atol = atol if _is_iterable(atol) else [atol] * len(y0)
        self.implicit = implicit
        self.first_step = first_step
        self.max_order = int(max(_MIN_ORDER, min(max_order, _MAX_ORDER)))
        self.safety = _convert_to_tensor(safety, dtype=tf.float64, device=y0[0].device)
        self.ifactor = _convert_to_tensor(ifactor, dtype=tf.float64, device=y0[0].device)
        self.dfactor = _convert_to_tensor(dfactor, dtype=tf.float64, device=y0[0].device)

    def before_integrate(self, t):
        prev_f = collections.deque(maxlen=self.max_order + 1)
        prev_t = collections.deque(maxlen=self.max_order + 1)
        phi = collections.deque(maxlen=self.max_order)

        t0 = t[0]
        f0 = self.func(tf.cast(t0, self.y0[0].dtype), self.y0)
        prev_t.appendleft(t0)
        prev_f.appendleft(f0)
        phi.appendleft(f0)

        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, 2, self.rtol[0], self.atol[0], f0=f0)
        else:
            first_step = _select_initial_step(self.func, t[0], self.y0, 2, self.rtol[0], self.atol[0], f0=f0)

        first_step = move_to_device(first_step, t.device)
        first_step = tf.cast(first_step, t[0].dtype)
        self.vcabm_state = _VCABMState(self.y0, prev_f, prev_t, next_t=t[0] + first_step, phi=phi, order=1)

    def advance(self, final_t):
        final_t = _convert_to_tensor(final_t, device=self.vcabm_state.prev_t[0].device)
        while final_t > self.vcabm_state.prev_t[0]:
            # print("VCABM State T = ", final_t.numpy(), self.vcabm_state.y_n)
            self.vcabm_state = self._adaptive_adams_step(self.vcabm_state, final_t)

        assert tf.equal(final_t, self.vcabm_state.prev_t[0])
        return self.vcabm_state.y_n

    def _adaptive_adams_step(self, vcabm_state, final_t):
        y0, prev_f, prev_t, next_t, prev_phi, order = vcabm_state
        if next_t > final_t:
            next_t = final_t
        dt = (next_t - prev_t[0])
        dt_cast = move_to_device(dt, y0[0].device)
        dt_cast = tf.cast(dt_cast, y0[0].dtype)

        # Explicit predictor step.
        g, phi = g_and_explicit_phi(prev_t, next_t, prev_phi, order)
        # g = move_to_device(g, y0[0].device)

        g = tf.cast(g, dt_cast.dtype)
        phi = [tf.cast(phi_, dt_cast.dtype) for phi_ in phi]

        p_next = tuple(
            y0_ + _scaled_dot_product(dt_cast, g[:max(1, order - 1)], phi_[:max(1, order - 1)])
            for y0_, phi_ in zip(y0, tuple(zip(*phi)))
        )

        # Update phi to implicit.
        next_t = move_to_device(next_t, p_next[0].device)
        next_f0 = self.func(tf.cast(next_t, self.y0[0].dtype), p_next)
        implicit_phi_p = compute_implicit_phi(phi, next_f0, order + 1)

        # Implicit corrector step.
        y_next = tuple(
            p_next_ + dt_cast * g[order - 1] * tf.cast(iphi_, dt_cast.dtype)
            for p_next_, iphi_ in zip(p_next, implicit_phi_p[order - 1])
        )

        # Error estimation.
        tolerance = tuple(
            atol_ + rtol_ * tf.reduce_max([tf.abs(y0_), tf.abs(y1_)])
            for atol_, rtol_, y0_, y1_ in zip(self.atol, self.rtol, y0, y_next)
        )
        local_error = tuple(dt_cast * (g[order] - g[order - 1]) * tf.cast(iphi_, dt_cast.dtype)
                            for iphi_ in implicit_phi_p[order])
        error_k = _compute_error_ratio(local_error, tolerance)
        accept_step = tf.reduce_all((tf.convert_to_tensor(error_k) <= 1))

        if not accept_step:
            # Retry with adjusted step size if step is rejected.
            dt_next = _optimal_step_size(dt, error_k, self.safety, self.ifactor, self.dfactor, order=order)
            return _VCABMState(y0, prev_f, prev_t, prev_t[0] + dt_next, prev_phi, order=order)

        # We accept the step. Evaluate f and update phi.
        next_t = move_to_device(next_t, p_next[0].device)
        next_f0 = self.func(tf.cast(next_t, self.y0[0].dtype), y_next)
        implicit_phi = compute_implicit_phi(phi, next_f0, order + 2)

        next_order = order

        if len(prev_t) <= 4 or order < 3:
            next_order = min(order + 1, 3, self.max_order)
        else:
            implicit_phi_p = [tf.cast(iphi_, dt_cast.dtype) for iphi_ in implicit_phi_p]

            error_km1 = _compute_error_ratio(
                tuple(dt_cast * (g[order - 1] - g[order - 2]) * iphi_
                      for iphi_ in implicit_phi_p[order - 1]),
                tolerance
            )
            error_km2 = _compute_error_ratio(
                tuple(dt_cast * (g[order - 2] - g[order - 3]) * iphi_
                      for iphi_ in implicit_phi_p[order - 2]),
                tolerance
            )
            if min(error_km1 + error_km2) < max(error_k):
                next_order = order - 1
            elif order < self.max_order:
                error_kp1 = _compute_error_ratio(
                    tuple(dt_cast * gamma_star[order] * iphi_ for iphi_ in implicit_phi_p[order]), tolerance
                )
                if max(error_kp1) < max(error_k):
                    next_order = order + 1

        # Keep step size constant if increasing order. Else use adaptive step size.
        dt_next = dt if next_order > order else _optimal_step_size(
            dt, error_k, self.safety, self.ifactor, self.dfactor, order=order + 1
        )

        prev_f.appendleft(next_f0)
        prev_t.appendleft(next_t)
        return _VCABMState(p_next, prev_f, prev_t, next_t + dt_next, implicit_phi, order=next_order)
