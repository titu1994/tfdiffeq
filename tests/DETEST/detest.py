import math
import numpy as np
import tensorflow as tf


####################################
# Problem Class A. Single equations.
####################################
def A1():
    diffeq = lambda t, y: -y
    init = lambda: (tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor(1., dtype=tf.float64))
    solution = lambda t: tf.exp(-t)
    return diffeq, init, solution


def A2():
    diffeq = lambda t, y: -y**3 / 2
    init = lambda: (tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor(1., dtype=tf.float64))
    solution = lambda t: 1 / tf.sqrt(t + 1)
    return diffeq, init, solution


def A3():
    diffeq = lambda t, y: y * tf.cos(t)
    init = lambda: (tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor(1., dtype=tf.float64))
    solution = lambda t: tf.exp(tf.sin(t))
    return diffeq, init, solution


def A4():
    diffeq = lambda t, y: y / 4 * (1 - y / 20)
    init = lambda: (tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor(1., dtype=tf.float64))
    solution = lambda t: 20 / (1 + 19 * tf.exp(-t / 4))
    return diffeq, init, solution


def A5():
    diffeq = lambda t, y: (y - t) / (y + t)
    init = lambda: (tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor(4., dtype=tf.float64))
    return diffeq, init, None


#################################
# Problem Class B. Small systems.
#################################
def B1():

    def diffeq(t, y):
        dy0 = 2 * (y[0] - y[0] * y[1])
        dy1 = -(y[1] - y[0] * y[1])
        return tf.stack([dy0, dy1])

    def init():
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor([1., 3.], dtype=tf.float64)

    return diffeq, init, None


def B2():

    A = tf.convert_to_tensor([[-1., 1., 0.], [1., -2., 1.], [0., 1., -1.]], dtype=tf.float64)

    def diffeq(t, y):
        dy = _matrix_vector(A, y)
        return dy

    def init():
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor([2., 0., 1.], dtype=tf.float64)

    return diffeq, init, None


def B3():

    def diffeq(t, y):
        dy0 = -y[0]
        dy1 = y[0] - y[1] * y[1]
        dy2 = y[1] * y[1]
        return tf.stack([dy0, dy1, dy2])

    def init():
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor([1., 0., 0.], dtype=tf.float64)

    return diffeq, init, None


def B4():

    def diffeq(t, y):
        a = tf.sqrt(y[0] * y[0] + y[1] * y[1])
        dy0 = -y[1] - y[0] * y[2] / a
        dy1 = y[0] - y[1] * y[2] / a
        dy2 = y[0] / a
        return tf.stack([dy0, dy1, dy2])

    def init():
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor([3., 0., 0.], dtype=tf.float64)

    return diffeq, init, None


def B5():

    def diffeq(t, y):
        dy0 = y[1] * y[2]
        dy1 = -y[0] * y[2]
        dy2 = -0.51 * y[0] * y[1]
        return tf.stack([dy0, dy1, dy2])

    def init():
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor([0., 1., 1.], dtype=tf.float64)

    return diffeq, init, None


####################################
# Problem Class C. Moderate systems.
####################################
def C1():

    A = np.zeros([10, 10])
    A = A.flatten()
    A[:-1:11] = -1.
    A[10::11] = 1.
    A = A.reshape([10, 10])

    A = tf.convert_to_tensor(A, dtype=tf.float64)

    def diffeq(t, y):
        return _matrix_vector(A, y)

    def init():
        y0 = np.zeros(10)
        y0[0] = 1
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor(y0, dtype=tf.float64)

    return diffeq, init, None


def C2():

    A = np.zeros([10, 10])
    A = A.flatten()
    A[:-1:11] = np.linspace(-1., -9., 9)
    A[10::11] = np.linspace(1., 9., 9)
    A = A.reshape([10, 10])

    A = tf.convert_to_tensor(A, dtype=tf.float64)

    def diffeq(t, y):
        return _matrix_vector(A, y)

    def init():
        y0 = np.zeros(10)
        y0[0] = 1
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor(y0, dtype=tf.float64)

    return diffeq, init, None


def C3():
    n = 10
    A = np.zeros([n, n])
    A = A.flatten()
    A[::n + 1] = -2.
    A[n::n + 1] = 1.
    A[1::n + 1] = 1.
    A = A.reshape([n, n])

    A = tf.convert_to_tensor(A, dtype=tf.float64)

    def diffeq(t, y):
        return _matrix_vector(A, y)

    def init():
        y0 = np.zeros(n)
        y0[0] = 1
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor(y0, dtype=tf.float64)

    return diffeq, init, None


def C4():
    n = 51
    A = np.zeros([n, n])
    A = A.flatten()
    A[::n + 1] = -2.
    A[n::n + 1] = 1.
    A[1::n + 1] = 1.
    A = A.reshape([n, n])

    A = tf.convert_to_tensor(A, dtype=tf.float64)

    def diffeq(t, y):
        return _matrix_vector(A, y)

    def init():
        y0 = np.zeros(n)
        y0[0] = 1
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor(y0, dtype=tf.float64)

    return diffeq, init, None


def C5():

    k2 = tf.convert_to_tensor(2.95912208286, dtype=tf.float64)
    m0 = tf.convert_to_tensor(1.00000597682, dtype=tf.float64)
    m = tf.convert_to_tensor([
        0.000954786104043,
        0.000285583733151,
        0.0000437273164546,
        0.0000517759138449,
        0.00000277777777778,
    ], dtype=tf.float64)
    m = tf.reshape(m, [1, 5])

    def diffeq(t, y):
        # y is 2 x 3 x 5
        # y[0] contains y, y[0] contains y'
        # second axis indexes space (x,y,z).
        # third axis indexes 5 bodies.
        dy = y[1, :, :]
        y = y[0]
        r = tf.reshape(tf.sqrt(tf.reduce_sum(y ** 2, axis=0)), (1, 5))
        d = tf.sqrt(tf.reduce_sum((y[:, :, None] - y[:, None, :]) ** 2, 0))
        F = tf.reshape(m, (1, 1, 5)) * ((y[:, None, :] - y[:, :, None]) / tf.reshape(d * d * d, (1, 5, 5)) + tf.reshape(y, (3, 1, 5)) /
                               tf.reshape(r * r * r, (1, 1, 5)))
        # F.view(3, 5 * 5)[:, ::6] = 0
        F = tf.reshape(F, (3, 5 * 5))
        F = F.numpy()
        F[:, ::6] = 0
        F = F.reshape((3, 5, 5))
        F = tf.convert_to_tensor(F, dtype=tf.float64)

        ddy = k2 * (-(m0 + m) * y / (r * r * r)) + tf.reduce_sum(F, 2)

        return tf.stack([dy, ddy], 0)

    def init():
        y0 = tf.convert_to_tensor([
            3.42947415189, 3.35386959711, 1.35494901715, 6.64145542550, 5.97156957878, 2.18231499728, 11.2630437207,
            14.6952576794, 6.27960525067, -30.1552268759, 165699966404, 1.43785752721, -21.1238353380, 28.4465098142,
            15.388265967
        ], dtype=tf.float64)
        y0 = tf.transpose(tf.reshape(y0, shape=[5, 3]), perm=[1, 0])

        dy0 = tf.convert_to_tensor([
            -.557160570446, .505696783289, .230578543901, -.415570776342, .365682722812, .169143213293, -.325325669158,
            .189706021964, .0877265322780, -.0240476254170, -.287659532608, -.117219543175, -.176860753121,
            -.216393453025, -.0148647893090
        ], dtype=tf.float64)
        dy0 = tf.transpose(tf.reshape(dy0, [5, 3]), perm=[1, 0])

        return tf.convert_to_tensor(0., dtype=tf.float64), tf.stack([y0, dy0], 0)

    return diffeq, init, None


###################################
# Problem Class D. Orbit equations.
###################################
def _DTemplate(eps):

    def diffeq(t, y):
        r = (y[0]**2 + y[1]**2)**(3 / 2)
        dy0 = y[2]
        dy1 = y[3]
        dy2 = -y[0] / r
        dy3 = -y[1] / r
        return tf.stack([dy0, dy1, dy2, dy3])

    def init():
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor([1 - eps, 0, 0, math.sqrt((1 + eps) / (1 - eps))], dtype=tf.float64)

    return diffeq, init, None


D1 = lambda: _DTemplate(0.1)
D2 = lambda: _DTemplate(0.3)
D3 = lambda: _DTemplate(0.5)
D4 = lambda: _DTemplate(0.7)
D5 = lambda: _DTemplate(0.9)


##########################################
# Problem Class E. Higher order equations.
##########################################
def E1():

    def diffeq(t, y):
        dy0 = y[1]
        dy1 = -(y[1] / (t + 1) + (1 - 0.25 / (t + 1)**2) * y[0])
        return tf.stack([dy0, dy1])

    def init():
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor([.671396707141803, .0954005144474744], dtype=tf.float64)

    return diffeq, init, None


def E2():

    def diffeq(t, y):
        dy0 = y[1]
        dy1 = (1 - y[0]**2) * y[1] - y[0]
        return tf.stack([dy0, dy1])

    def init():
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor([2., 0.], dtype=tf.float64)

    return diffeq, init, None


def E3():

    def diffeq(t, y):
        dy0 = y[1]
        dy1 = y[0]**3 / 6 - y[0] + 2 * tf.sin(2.78535 * t)
        return tf.stack([dy0, dy1])

    def init():
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor([0., 0.], dtype=tf.float64)

    return diffeq, init, None


def E4():

    def diffeq(t, y):
        dy0 = y[1]
        dy1 = .32 - .4 * y[1]**2
        return tf.stack([dy0, dy1])

    def init():
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor([30., 0.], dtype=tf.float64)

    return diffeq, init, None


def E5():

    def diffeq(t, y):
        dy0 = y[1]
        dy1 = tf.sqrt(1 + y[1]**2) / (25 - t)
        return tf.stack([dy0, dy1])

    def init():
        return tf.convert_to_tensor(0., dtype=tf.float64), tf.convert_to_tensor([0., 0.], dtype=tf.float64)

    return diffeq, init, None


###################
# Helper functions.
###################
def _to_tensor(x):
    if not tf.is_numeric_tensor(x):
        x = tf.convert_to_tensor(x, dtype=tf.float64)
    return x


def _matrix_vector(A, y):
    A = tf.cast(A, tf.float64)
    y = tf.cast(y, tf.float64)

    y = tf.expand_dims(y, -1)
    z = tf.matmul(A, y)

    z = tf.squeeze(z)
    return z
