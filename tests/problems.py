import math

import numpy as np
import scipy.linalg
import tensorflow as tf


class ConstantODE(tf.keras.Model):

    def __init__(self, dtype=None):
        super(ConstantODE, self).__init__(dtype=dtype)
        self.a = tf.Variable(0.2, dtype=self.dtype)
        self.b = tf.Variable(3.0, dtype=self.dtype)

    def call(self, t, y):
        return self.a + (y - (self.a * t + self.b)) ** 5

    def y_exact(self, t):
        t = tf.cast(t, self.dtype)
        return self.a * t + self.b


class SineODE(tf.keras.Model):

    def __init__(self, dtype=None):
        super(SineODE, self).__init__(dtype=dtype)

    def call(self, t, y):
        return 2 * y / t + t ** 4 * tf.sin(2 * t) - t ** 2 + 4 * t ** 3

    def y_exact(self, t):
        t = tf.cast(t, self.dtype)
        return -0.5 * t ** 4 * tf.cos(2 * t) + 0.5 * t ** 3 * tf.sin(2 * t) + 0.25 * t ** 2 * tf.cos(
            2 * t
        ) - t ** 3 + 2 * t ** 4 + (math.pi - 0.25) * t ** 2


class LinearODE(tf.keras.Model):

    def __init__(self, dim=10, dtype=None):
        super(LinearODE, self).__init__(dtype=dtype)
        self.dim = dim
        U = np.random.randn(dim, dim) * 0.1
        A = 2 * U - (U + U.transpose(0, 1))
        self.A = tf.Variable(A, dtype=dtype)
        self.initial_val = np.ones((dim, 1))

    def call(self, t, y):
        y = tf.reshape(y, [self.dim, 1])
        out = tf.matmul(self.A, y)
        out = tf.reshape(out, [-1])
        return out

    def y_exact(self, t):
        t = tf.cast(t, self.dtype)
        t = t.numpy()
        A_np = self.A.numpy()
        ans = []
        for t_i in t:
            ans.append(np.matmul(scipy.linalg.expm(A_np * t_i), self.initial_val))
        v = tf.stack([tf.Variable(ans_) for ans_ in ans])
        v = tf.reshape(v, [t.shape[0], self.dim])
        return v


PROBLEMS = {'constant': ConstantODE, 'linear': LinearODE, 'sine': SineODE}
DTYPES = (tf.float32, tf.float64)
DEVICES = ['cpu']

if tf.test.is_gpu_available():
    DEVICES.append('gpu:0')

FIXED_METHODS = ('euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams')
ADAPTIVE_METHODS = ('dopri5', 'bosh3', 'adaptive_heun', 'dopri8')  # TODO: add in adaptive adams and tsit5 if/when they're fixed
METHODS = FIXED_METHODS + ADAPTIVE_METHODS


def construct_problem(device, npts=10, ode='constant', reverse=False, dtype=tf.float64):
    with tf.device(device):
        f = PROBLEMS[ode](dtype=dtype)

    t_points = tf.linspace(1., 8., npts)
    sol = f.y_exact(t_points)

    def _flip(x, dim):
        # indices = [slice(None)] * len(x.shape)
        # indices[dim] = tf.range(x.shape[dim] - 1, -1, -1, dtype=tf.int64)
        return x[::-1]  # x[list(indices)]

    if reverse:
        t_points = tf.identity(_flip(t_points, 0))
        sol = tf.identity(_flip(sol, 0))

    return f, sol[0], t_points, sol


if __name__ == '__main__':
    f = SineODE()
    t_points = tf.linspace(1., 8., 100)
    sol = f.y_exact(t_points)

    import matplotlib.pyplot as plt

    plt.plot(t_points.numpy(), sol.numpy())
    plt.show()
