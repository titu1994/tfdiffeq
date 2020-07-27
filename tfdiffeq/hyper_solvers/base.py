import tensorflow as tf
from abc import ABC, abstractmethod


class AbstractHyperSolver(tf.keras.Model, ABC):

    def __init__(self, func: tf.keras.Model, hyper_solver: tf.keras.Model, **kwargs):
        """
        A port of the HyperSolver form the paper [Hypersolvers: Toward Fast Continuous-Depth Models](https://arxiv.org/abs/2007.09601)

        Args:
            func: The ODE function that should be learned.
            hyper_solver: A HyperNetwork that approximates the higher order terms of a given solver.
            kwargs: Any additional kwarg to the solver Model
        """
        super(AbstractHyperSolver, self).__init__(**kwargs)

        self.f = func
        self.g = hyper_solver

        print(f"{self.__class__.__name__} is an experimental model, the API may change significantly !")

    @tf.function
    def call(self, t: tf.Tensor, y: tf.Tensor, dy: tf.Tensor):
        """

        Args:
            t: Tensor of some length T
            y: Input to the self.f = func(t, y0)
            dy: Output of calling self.f = func(t, y0)

        Returns:
            tf.Tensor of same shape as `y`
        """
        # Estimate higher order terms to compensate for truncation error at `x`
        t = t * tf.ones([*y.shape[:1], 1, *y.shape[2:]], dtype=t.dtype)
        y = tf.concat([y, dy, t], axis=1)
        y = self.g(y)
        return y

    @abstractmethod
    def trajectory(self, t_span: tf.Tensor, y: tf.Tensor):
        # Extrapolate a trajectory with span `s_span`
        pass

    @abstractmethod
    def residual_trajectory(self, t_span: tf.Tensor, base_traj: tf.Tensor):
        # Recover residuals from a base trajectory
        pass

    @tf.function
    def _hypersolver_residuals(self, t_span: tf.Tensor, base_traj: tf.Tensor):
        # Calculate residual hypersolver predictions starting from a trajectory
        traj = tf.TensorArray(dtype=base_traj.dtype, size=0, dynamic_size=True)
        dt = t_span[1] - t_span[0]
        for i in tf.range(t_span.shape[0]):
            t = t_span[i]
            y = base_traj[i]
            dy = self.f(t, y)
            res = self(dt, y, dy)

            traj = traj.write(i, res[None])

        traj = traj.concat()
        return traj
