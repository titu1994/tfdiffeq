import os
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import tensorflow as tf
import tensorflow_probability as tfp

matplotlib.use('agg')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--train_dir', type=str, default="./cnf/")
parser.add_argument('--results_dir', type=str, default="./cnf/results")
args = parser.parse_args()

if args.adjoint:
    from tfdiffeq import odeint_adjoint as odeint
else:
    from tfdiffeq import odeint

# Cast everything to float 64
float_dtype = tf.float64
tf.keras.backend.set_floatx(float_dtype.base_dtype.name)


class CNF(tf.keras.Model):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """

    def __init__(self, in_out_dim, hidden_dim, width, **kwargs):
        super().__init__(**kwargs, dtype=float_dtype.base_dtype.name)
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width, dtype=float_dtype.base_dtype.name)

    @tf.function
    def call(self, t, states):
        logp_z = states[1]

        with tf.GradientTape(persistent=True) as inner_tape:
            z = states[0]
            inner_tape.watch(z)

            batchsize = z.shape[0]
            W, B, U = self.hyper_net(t)

            z = tf.cast(z, W.dtype)
            Z = tf.tile(tf.expand_dims(z, axis=0), (self.width, 1, 1))

            h = tf.tanh(tf.matmul(Z, W) + B)
            dz_dt = tf.reduce_mean(tf.matmul(h, U), axis=0)
            dz_dt_sum = tf.reduce_sum(dz_dt, axis=0)

            outputs = [dz_dt_sum[i] for i in range(dz_dt_sum.shape[0])]

        """ Inlining to satisfy gradient tape
        Calculates the trace of the Jacobian df/dz.
        Obtained from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
        """
        sum_diag = 0.
        for i in range(z.shape[1]):
            sum_diag += inner_tape.gradient(outputs[i], z)[:, i]

        dlogp_z_dt = -tf.reshape(sum_diag, (batchsize, 1))
        return (dz_dt, dlogp_z_dt)


class HyperNetwork(tf.keras.Model):
    """Hyper-network allowing f(z(t), t) to change with time.
    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """

    def __init__(self, in_out_dim, hidden_dim, width, **kwargs):
        super().__init__(**kwargs)

        blocksize = width * in_out_dim

        self.fc1 = tf.keras.layers.Dense(hidden_dim, bias_initializer='he_uniform', **kwargs)
        self.fc2 = tf.keras.layers.Dense(hidden_dim, bias_initializer='he_uniform', **kwargs)
        self.fc3 = tf.keras.layers.Dense(3 * blocksize + width, bias_initializer='he_uniform', **kwargs)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    @tf.function
    def call(self, t):
        # predict params
        params = tf.reshape(t, (1, 1))
        params = tf.tanh(self.fc1(params))
        params = tf.tanh(self.fc2(params))
        params = self.fc3(params)

        # restructure
        params = tf.reshape(params, [-1])
        W = tf.reshape(params[:self.blocksize], (self.width, self.in_out_dim, 1))

        U = tf.reshape(params[self.blocksize:2 * self.blocksize], (self.width, 1, self.in_out_dim))

        G = tf.reshape(params[2 * self.blocksize:3 * self.blocksize], (self.width, 1, self.in_out_dim))
        U = U * tf.sigmoid(G)

        B = tf.reshape(params[3 * self.blocksize:], (self.width, 1, 1))
        return [W, B, U]


def get_batch(num_samples):
    points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
    x = tf.convert_to_tensor(points, dtype=float_dtype)
    logp_diff_t1 = tf.zeros([num_samples, 1], dtype=float_dtype)

    return (x, logp_diff_t1)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':
    t0 = 0
    t1 = 10

    if tf.test.is_gpu_available():
        device = 'gpu'
    else:
        device = 'cpu'

    with tf.device(device):

        # model
        func = CNF(in_out_dim=2, hidden_dim=args.hidden_dim, width=args.width)
        optimizer = tf.keras.optimizers.Adam(lr=args.lr)

        covariance_matrix = tf.convert_to_tensor([[0.1, 0.0], [0.0, 0.1]])
        p_z0 = tfp.distributions.MultivariateNormalTriL(
            loc=tf.convert_to_tensor([0.0, 0.0]),
            scale_tril=tf.linalg.cholesky(covariance_matrix)
        )
        loss_meter = RunningAverageMeter()

        if args.train_dir is not None:
            if not os.path.exists(args.train_dir):
                os.makedirs(args.train_dir)

            ckpt_path = os.path.join(args.train_dir, 'ckpt')
            checkpoint = tf.train.Checkpoint(func=func, optimizer=optimizer)
            checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.train_dir, max_to_keep=1)
            latest_checkpoint = checkpoint_manager.latest_checkpoint

            checkpoint_manager.restore_or_initialize()

            if latest_checkpoint is not None:
                print('Loaded ckpt from {}'.format(ckpt_path))

        if args.mode == 'train':
            try:
                for itr in range(1, args.niters + 1):
                    with tf.GradientTape() as tape:
                        x, logp_diff_t1 = get_batch(args.num_samples)

                        z_t, logp_diff_t = odeint(
                            func,
                            (x, logp_diff_t1),
                            tf.convert_to_tensor([t1, t0], dtype=float_dtype),
                            atol=1e-5,
                            rtol=1e-5,
                            method='dopri5',
                        )

                        z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

                        # Float 32 required for log_prob()
                        z_t0 = tf.cast(z_t0, tf.float32)
                        logp_diff_t0 = tf.cast(logp_diff_t0, tf.float32)

                        logp_x = p_z0.log_prob(z_t0) - tf.reshape(logp_diff_t0, [-1])

                        # Recast
                        logp_x = tf.cast(logp_x, float_dtype)

                        loss = -tf.reduce_mean(logp_x, axis=0)

                    grads = tape.gradient(loss, func.trainable_variables)
                    grads = [tf.cast(grad, float_dtype) for grad in grads]
                    grads_vars = zip(grads, func.trainable_variables)

                    optimizer.apply_gradients(grads_vars)

                    loss_meter.update(loss.numpy())

                    print('Iter: {}, running avg loss: {:.4f}'.format(itr, loss_meter.avg))

                    if (itr % 100 == 0) and itr != 0:
                        print("Saving weights...")
                        checkpoint_manager.save()

            except KeyboardInterrupt:
                checkpoint_manager.save()
                print("Training stopped. Weights have been saved.")

            print('Training complete after {} iters.'.format(itr))

        if args.visualize:
            print("Visualizing samples...")

            viz_samples = 30000
            viz_timesteps = 41
            target_sample, _ = get_batch(viz_samples)

            if not os.path.exists(args.results_dir):
                os.makedirs(args.results_dir)

            checkpoint_manager.restore_or_initialize()
            print("Parameters restored !")

            # Generate evolution of samples
            z_t0 = p_z0.sample([viz_samples])
            logp_diff_t0 = tf.zeros([viz_samples, 1], dtype=tf.float32)

            z_t_samples, _ = odeint(
                func,
                (z_t0, logp_diff_t0),
                tf.convert_to_tensor(np.linspace(t0, t1, viz_timesteps)),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )

            # Generate evolution of density
            x = np.linspace(-1.5, 1.5, 100)
            y = np.linspace(-1.5, 1.5, 100)
            points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

            z_t1 = tf.convert_to_tensor(points, dtype=tf.float32)
            logp_diff_t1 = tf.zeros([z_t1.shape[0], 1], dtype=tf.float32)

            z_t_density, logp_diff_t = odeint(
                func,
                (z_t1, logp_diff_t1),
                tf.convert_to_tensor(np.linspace(t1, t0, viz_timesteps)),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )

            # Create plots for each timestep
            for (t, z_sample, z_density, logp_diff) in zip(
                    np.linspace(t0, t1, viz_timesteps),
                    z_t_samples, z_t_density, logp_diff_t
            ):
                fig = plt.figure(figsize=(12, 4), dpi=200)
                plt.tight_layout()
                plt.axis('off')
                plt.margins(0, 0)
                fig.suptitle(f'{t:.2f}s')

                ax1 = fig.add_subplot(1, 3, 1)
                ax1.set_title('Target')
                ax1.get_xaxis().set_ticks([])
                ax1.get_yaxis().set_ticks([])
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.set_title('Samples')
                ax2.get_xaxis().set_ticks([])
                ax2.get_yaxis().set_ticks([])
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_title('Log Probability')
                ax3.get_xaxis().set_ticks([])
                ax3.get_yaxis().set_ticks([])

                ax1.hist2d(*target_sample.numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                ax2.hist2d(*z_sample.numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                z_density = tf.cast(z_density, tf.float32)
                logp_diff = tf.cast(logp_diff, tf.float32)

                logp = p_z0.log_prob(z_density) - tf.reshape(logp_diff, [-1])
                # logp = logp - tf.reduce_max(logp)

                ax3.tricontourf(*z_t1.numpy().T,
                                np.exp(logp.numpy()), 200)

                plt.savefig(os.path.join(args.results_dir, f"cnf-viz-{int(t * 1000):05d}.jpg"),
                            pad_inches=0.2, bbox_inches='tight')
                plt.close()

            img, *imgs = [Image.open(f) for f in
                          sorted(glob.glob(os.path.join(args.results_dir, f"cnf-viz-*.jpg")))]
            img.save(fp=os.path.join(args.results_dir, "cnf-viz.gif"), format='GIF', append_images=imgs,
                     save_all=True, duration=250, loop=0)

            print('Saved visualization animation at {}'.format(os.path.join(args.results_dir, "cnf-viz.gif")))
