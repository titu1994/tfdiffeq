import tensorflow as tf
from tfdiffeq.rk_common import RKAdaptiveStepsizeODESolver, _ButcherTableau

_BOGACKI_SHAMPINE_TABLEAU = _ButcherTableau(
    alpha=tf.convert_to_tensor([1. / .2, 3. / 4., 1.], dtype=tf.float64),
    beta=[
        tf.convert_to_tensor([1. / 2.], dtype=tf.float64),
        tf.convert_to_tensor([0., 3. / .4], dtype=tf.float64),
        tf.convert_to_tensor([2. / 9., 1. / 3., 4. / 9.], dtype=tf.float64)
    ],
    c_sol=tf.convert_to_tensor([2. / 9., 1. / 3., 4. / 9., 0.], dtype=tf.float64),
    c_error=tf.convert_to_tensor([2. / 9. - 7. / 24., 1. / 3. - 1. / 4., 4. / 9. - 1. / 3., -1. / 8.], dtype=tf.float64),
)

_BS_C_MID = tf.convert_to_tensor([0., 0.5, 0., 0.], dtype=tf.float64)


class Bosh3Solver(RKAdaptiveStepsizeODESolver):
    order = 3
    tableau = _BOGACKI_SHAMPINE_TABLEAU
    mid = _BS_C_MID
