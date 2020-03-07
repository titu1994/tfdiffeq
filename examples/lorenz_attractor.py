import time

import matplotlib.pyplot as plt
import tensorflow as tf

from tfdiffeq import odeint, cast_double

if tf.version.VERSION.startswith("1."):
    tf.enable_v2_behavior()


# tf.keras.backend.set_floatx('float64')


# Lorenz Attractor requires a large number of non-matrix computations
# Therefore, it is better to run it on the CPU only.
device = 'cpu:0'


class Lorenz(tf.Module):

    def __init__(self, sigma=10., beta=8 / 3., rho=28., **kwargs):
        super().__init__(**kwargs)

        self.sigma = float(sigma)
        self.beta = float(beta)
        self.rho = float(rho)

    @tf.function
    def __call__(self, t, y):
        """ y here is [x, y, z] """
        y = cast_double(y)

        dx_dt = self.sigma * (y[1] - y[0])
        dy_dt = y[0] * (self.rho - y[2]) - y[1]
        dz_dt = y[0] * y[1] - self.beta * y[2]

        dL_dt = tf.stack([dx_dt, dy_dt, dz_dt])
        return dL_dt


t = tf.range(0.0, 100.0, 0.01, dtype=tf.float64)
initial_state = tf.convert_to_tensor([1., 1., 1.], dtype=tf.float64)

sigma = 10.
beta = 8. / 3.
rho = 28.

with tf.device(device):
    t1 = time.time()
    solution = odeint(Lorenz(sigma, beta, rho), initial_state, t).numpy()
    t2 = time.time()

print("Finished integrating ! Result shape :", solution.shape)
print("Time required (s): ", t2 - t1)

from mpl_toolkits.mplot3d import Axes3D  # needed for plotting in 3d
_ = Axes3D

fig = plt.figure(figsize=(16, 16))
ax = fig.gca(projection='3d')
ax.set_title('Lorenz Attractor')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.plot(solution[:, 0], solution[:, 1], solution[:, 2])
plt.show()
