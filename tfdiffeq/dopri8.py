import tensorflow as tf
from tfdiffeq.misc import (
    _scaled_dot_product, _convert_to_tensor, _is_finite, _select_initial_step, _handle_unused_kwargs, _is_iterable,
    _optimal_step_size, _compute_error_ratio, cast_double, move_to_device
)
from tfdiffeq.solvers import AdaptiveStepsizeODESolver
from tfdiffeq.interp import _interp_fit, _interp_evaluate
from tfdiffeq.rk_common import _RungeKuttaState, _ButcherTableau, _runge_kutta_step

import numpy as np

A = [1 / 18, 1 / 12, 1 / 8, 5 / 16, 3 / 8, 59 / 400, 93 / 200, 5490023248 / 9719169821, 13 / 20,
     1201146811 / 1299019798, 1, 1, 1]

B = [
    [1 / 18],
    [1 / 48, 1 / 16],
    [1 / 32, 0, 3 / 32],
    [5 / 16, 0, -75 / 64, 75 / 64],
    [3 / 80, 0, 0, 3 / 16, 3 / 20],
    [29443841 / 614563906, 0, 0, 77736538 / 692538347, -28693883 / 1125000000, 23124283 / 1800000000],
    [16016141 / 946692911, 0, 0, 61564180 / 158732637, 22789713 / 633445777, 545815736 / 2771057229,
     -180193667 / 1043307555],
    [39632708 / 573591083, 0, 0, -433636366 / 683701615, -421739975 / 2616292301, 100302831 / 723423059,
     790204164 / 839813087, 800635310 / 3783071287],
    [246121993 / 1340847787, 0, 0, -37695042795 / 15268766246, -309121744 / 1061227803, -12992083 / 490766935,
     6005943493 / 2108947869, 393006217 / 1396673457, 123872331 / 1001029789],
    [-1028468189 / 846180014, 0, 0, 8478235783 / 508512852, 1311729495 / 1432422823, -10304129995 / 1701304382,
     -48777925059 / 3047939560, 15336726248 / 1032824649, -45442868181 / 3398467696, 3065993473 / 597172653],
    [185892177 / 718116043, 0, 0, -3185094517 / 667107341, -477755414 / 1098053517, -703635378 / 230739211,
     5731566787 / 1027545527, 5232866602 / 850066563, -4093664535 / 808688257, 3962137247 / 1805957418,
     65686358 / 487910083],
    [403863854 / 491063109, 0, 0, -5068492393 / 434740067, -411421997 / 543043805, 652783627 / 914296604,
     11173962825 / 925320556, -13158990841 / 6184727034, 3936647629 / 1978049680, -160528059 / 685178525,
     248638103 / 1413531060, 0],
    [14005451 / 335480064, 0, 0, 0, 0, -59238493 / 1068277825, 181606767 / 758867731, 561292985 / 797845732,
     -1041891430 / 1371343529, 760417239 / 1151165299, 118820643 / 751138087, -528747749 / 2220607170, 1 / 4]
]

C_sol = [14005451 / 335480064, 0, 0, 0, 0, -59238493 / 1068277825, 181606767 / 758867731, 561292985 / 797845732,
         -1041891430 / 1371343529, 760417239 / 1151165299, 118820643 / 751138087, -528747749 / 2220607170, 1 / 4, 0]

C_err = [14005451 / 335480064 - 13451932 / 455176623, 0, 0, 0, 0, -59238493 / 1068277825 - -808719846 / 976000145,
         181606767 / 758867731 - 1757004468 / 5645159321, 561292985 / 797845732 - 656045339 / 265891186,
         -1041891430 / 1371343529 - -3867574721 / 1518517206, 760417239 / 1151165299 - 465885868 / 322736535,
         118820643 / 751138087 - 53011238 / 667516719, -528747749 / 2220607170 - 2 / 45, 1 / 4, 0]

h = 1 / 2

C_mid = np.zeros(14)

C_mid[0] = (- 6.3448349392860401388 * (h ** 5) + 22.1396504998094068976 * (h ** 4) - 30.0610568289666450593 * (
            h ** 3) + 19.9990069333683970610 * (h ** 2) - 6.6910181737837595697 * h + 1.0) / (1 / h)
C_mid[5] = (- 39.6107919852202505218 * (h ** 5) + 116.4422149550342161651 * (h ** 4) - 121.4999627731334642623 * (
            h ** 3) + 52.2273532792945524050 * (h ** 2) - 7.6142658045872677172 * h) / (1 / h)
C_mid[6] = (20.3761213808791436958 * (h ** 5) - 67.1451318825957197185 * (h ** 4) + 83.1721004639847717481 * (
            h ** 3) - 46.8919164181093621583 * (h ** 2) + 10.7281392630428866124 * h) / (1 / h)
C_mid[7] = (7.3347098826795362023 * (h ** 5) - 16.5672243527496524646 * (h ** 4) + 9.5724507555993664382 * (
            h ** 3) - 0.1890893225010595467 * (h ** 2) + 0.5526637063753648783 * h) / (1 / h)
C_mid[8] = (32.8801774352459155182 * (h ** 5) - 89.9916014847245016028 * (h ** 4) + 87.8406057677205645007 * (
            h ** 3) - 35.7075975946222072821 * (h ** 2) + 4.2186562625665153803 * h) / (1 / h)
C_mid[9] = (- 10.1588990526426760954 * (h ** 5) + 22.6237489648532849093 * (h ** 4) - 17.4152107770762969005 * (
            h ** 3) + 6.2736448083240352160 * (h ** 2) - 0.6627209125361597559 * h) / (1 / h)
C_mid[10] = (- 12.5401268098782561200 * (h ** 5) + 32.2362340167355370113 * (h ** 4) - 28.5903289514790976966 * (
            h ** 3) + 10.3160881272450748458 * (h ** 2) - 1.2636789001135462218 * h) / (1 / h)
C_mid[11] = (29.5553001484516038033 * (h ** 5) - 82.1020315488359848644 * (h ** 4) + 81.6630950584341412934 * (
            h ** 3) - 34.7650769866611817349 * (h ** 2) + 5.4106037898590422230 * h) / (1 / h)
C_mid[12] = (- 41.7923486424390588923 * (h ** 5) + 116.2662185791119533462 * (h ** 4) - 114.9375291377009418170 * (
            h ** 3) + 47.7457971078225540396 * (h ** 2) - 7.0321379067945741781 * h) / (1 / h)
C_mid[13] = (20.3006925822100825485 * (h ** 5) - 53.9020777466385396792 * (h ** 4) + 50.2558364226176017553 * (
            h ** 3) - 19.0082099341608028453 * (h ** 2) + 2.3537586759714983486 * h) / (1 / h)
C_mid = C_mid.tolist()

_DOPRI8_TABLEAU = _ButcherTableau(alpha=A,
                                  beta=B,
                                  c_sol=C_sol,
                                  c_error=C_err)

c_mid = C_mid


def _interp_fit_dopri8(y0, y1, k, dt, tableau=_DOPRI8_TABLEAU):
    """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
    dt = cast_double(dt)
    y0 = cast_double(y0)

    y_mid = tuple(y0_ + _scaled_dot_product(dt, c_mid, k_) for y0_, k_ in zip(y0, k))
    f0 = tuple(k_[0] for k_ in k)
    f1 = tuple(k_[-1] for k_ in k)
    return _interp_fit(y0, y1, y_mid, f0, f1, dt)


def _abs_square(x):
    return tf.multiply(x, x)


def _ta_append(list_of_tensors, value):
    """Append a value to the end of a list of PyTorch tensors."""
    list_of_tensors.append(value)
    return list_of_tensors


class Dopri8Solver(AdaptiveStepsizeODESolver):

    def __init__(
            self, func, y0, rtol, atol, first_step=None, safety=0.9, ifactor=10.0, dfactor=0.2,
            max_num_steps=2 ** 31 - 1,
            **unused_kwargs
    ):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs
        super(Dopri8Solver, self).__init__(func, y0, atol, rtol)

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
            first_step = _select_initial_step(self.func, t[0], self.y0, 7, self.rtol[0], self.atol[0], f0=f0)
            first_step = move_to_device(first_step, t.device)
        else:
            first_step = _convert_to_tensor(self.first_step, dtype=t.dtype, device=t.device)
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, interp_coeff=[self.y0] * 5)

    def advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_dopri8_step(self.rk_state)
            n_steps += 1
        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)

    def _adaptive_dopri8_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state
        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        for y0_ in y0:
            assert _is_finite(tf.abs(y0_)), 'non-finite values in state `y`: {}'.format(y0_)
        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, tableau=_DOPRI8_TABLEAU)

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        mean_sq_error_ratio = _compute_error_ratio(y1_error, atol=self.atol, rtol=self.rtol, y0=y0, y1=y1)
        accept_step = tf.reduce_all(tf.convert_to_tensor(mean_sq_error_ratio) <= 1)

        ########################################################
        #                   Update RK State                    #
        ########################################################
        y_next = y1 if accept_step else y0
        f_next = f1 if accept_step else f0
        t_next = t0 + dt if accept_step else t0
        interp_coeff = _interp_fit_dopri8(y0, y1, k, dt) if accept_step else interp_coeff
        dt_next = _optimal_step_size(
            dt, mean_sq_error_ratio, safety=self.safety, ifactor=self.ifactor, dfactor=self.dfactor, order=8
        )
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state
