import tensorflow as tf
from tfdiffeq.rk_common import RKAdaptiveStepsizeODESolver, _ButcherTableau

_ADAPTIVE_HEUN_TABLEAU = _ButcherTableau(
    alpha=tf.convert_to_tensor([1.], dtype=tf.float64),
    beta=[
        tf.convert_to_tensor([1.], dtype=tf.float64),
    ],
    c_sol=tf.convert_to_tensor([0.5, 0.5], dtype=tf.float64),
    c_error=tf.convert_to_tensor([
        0.5,
        -0.5,
    ], dtype=tf.float64),
)

_AH_C_MID = tf.convert_to_tensor([
    0.5, 0.
], dtype=tf.float64)


class AdaptiveHeunSolver(RKAdaptiveStepsizeODESolver):
    order = 2
    tableau = _ADAPTIVE_HEUN_TABLEAU
    mid = _AH_C_MID
