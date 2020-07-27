import tensorflow as tf
from tfdiffeq.hyper_solvers.base import AbstractHyperSolver


class HyperEuler(AbstractHyperSolver):

    @tf.function
    def trajectory(self, t_span: tf.Tensor, y: tf.Tensor):
        # Extrapolate a trajectory with span `s_span`
        traj = tf.TensorArray(dtype=y.dtype, size=0, dynamic_size=True)
        dt = t_span[1] - t_span[0]
        for i in tf.range(t_span.shape[0]):
            t = t_span[i]
            dy = self.f(t, y)
            traj = traj.write(i, y[None])

            y = y + dy * dt + (dt ** 2) * self(dt, y, dy)

        traj = traj.concat()
        return traj

    @tf.function
    def residual_trajectory(self, t_span: tf.Tensor, base_traj: tf.Tensor):
        # Recover residuals from a base trajectory
        dt = t_span[1] - t_span[0]
        fi = tf.concat(
            [self.f(t_span[i], base_traj[i])[None, :, :]
             for i in tf.range(t_span.shape - 1)],
            axis=0
        )
        return (base_traj[1:] - base_traj[:-1] - dt * fi) / dt ** 2


class HyperMidpoint(AbstractHyperSolver):

    @tf.function
    def trajectory(self, t_span: tf.Tensor, y: tf.Tensor):
        # Extrapolate a trajectory with span `s_span`
        traj = tf.TensorArray(dtype=y.dtype, size=0, dynamic_size=True)
        dt = t_span[1] - t_span[0]
        for i in tf.range(t_span.shape[0]):
            t = t_span[i]
            traj = traj.write(i, y[None])

            dy = self.f(t, y)
            y_mid = y + dy * dt / 2. + (dt ** 2) * self(dt, y, dy)
            dy_2 = self.f(t + dt / 2., y_mid)
            y = y + dt * dy_2 + (dt ** 3) * self(dt, y_mid, dy_2)

        traj = traj.concat()
        return traj

    @tf.function
    def residual_trajectory(self, t_span: tf.Tensor, base_traj: tf.Tensor):
        raise NotImplementedError()


class HyperHeun(AbstractHyperSolver):

    @tf.function
    def trajectory(self, t_span: tf.Tensor, y: tf.Tensor):
        # Extrapolate a trajectory with span `s_span`
        traj = tf.TensorArray(dtype=y.dtype, size=0, dynamic_size=True)
        dt = t_span[1] - t_span[0]
        for i in tf.range(t_span.shape[0]):
            t = t_span[i]
            traj = traj.write(i, y[None])

            dy = self.f(t, y)
            y2 = y + dy * dt + (dt ** 2) * self(dt, y, dy)
            dy_2 = self.f(t + dt, y2)
            y = y + dt / 2. * (dy + dy_2) + (dt ** 3) * self(dt, y2, dy_2)

        traj = traj.concat()
        return traj

    @tf.function
    def residual_trajectory(self, t_span: tf.Tensor, base_traj: tf.Tensor):
        raise NotImplementedError()
