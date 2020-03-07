import math

import numpy as np
import scipy.linalg
import tensorflow as tf

if tf.version.VERSION.startswith("1."):
    tf.enable_v2_behavior()

tf.keras.backend.set_floatx('float64')


class ConstantODE(tf.keras.Model):

    def __init__(self):
        super(ConstantODE, self).__init__()
        self.a = tf.Variable(0.2, dtype=tf.float64)
        self.b = tf.Variable(3.0, dtype=tf.float64)

    def call(self, t, y):
        return self.a + (y - (self.a * t + self.b)) ** 5

    def y_exact(self, t):
        t = tf.cast(t, tf.float64)
        return self.a * t + self.b


class SineODE(tf.keras.Model):

    def __init__(self):
        super(SineODE, self).__init__()

    def call(self, t, y):
        return 2 * y / t + t ** 4 * tf.sin(2 * t) - t ** 2 + 4 * t ** 3

    def y_exact(self, t):
        t = tf.cast(t, tf.float64)
        return -0.5 * t ** 4 * tf.cos(2 * t) + 0.5 * t ** 3 * tf.sin(2 * t) + 0.25 * t ** 2 * tf.cos(
            2 * t
        ) - t ** 3 + 2 * t ** 4 + (math.pi - 0.25) * t ** 2


class LinearODE(tf.keras.Model):

    def __init__(self, dim=10):
        super(LinearODE, self).__init__()
        self.dim = dim
        U = np.random.randn(dim, dim) * 0.1
        A = 2 * U - (U + U.transpose(0, 1))
        self.A = tf.Variable(A)
        self.initial_val = np.ones((dim, 1))

    def call(self, t, y):
        y = tf.reshape(y, [self.dim, 1])
        out = tf.matmul(self.A, y)
        out = tf.reshape(out, [-1])
        return out

    def y_exact(self, t):
        t = tf.cast(t, tf.float64)
        t = t.numpy()
        A_np = self.A.numpy()
        ans = []
        for t_i in t:
            ans.append(np.matmul(scipy.linalg.expm(A_np * t_i), self.initial_val))
        v = tf.stack([tf.Variable(ans_) for ans_ in ans])
        v = tf.reshape(v, [t.shape[0], self.dim])
        return v


PROBLEMS = {'constant': ConstantODE, 'linear': LinearODE, 'sine': SineODE}


def construct_problem(device, npts=10, ode='constant', reverse=False):
    with tf.device(device):
        f = PROBLEMS[ode]()

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
    tf.enable_eager_execution()

    f = SineODE()
    t_points = tf.linspace(1., 8., 100)
    sol = f.y_exact(t_points)

    import matplotlib.pyplot as plt

    plt.plot(t_points.numpy(), sol.numpy())
    plt.show()
