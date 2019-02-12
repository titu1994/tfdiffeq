import argparse
import os

import matplotlib
import numpy as np
import numpy.random as npr

matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default='latent')
args = parser.parse_args()


from tfdiffeq import odeint, move_to_device, cast_double, func_cast_double


def generate_spiral2d(nspiral=1000,
                      ntotal=500,
                      nsample=100,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      a=0.,
                      b=1.,
                      savefig=True):
    """Parametric formula for 2d spiral is `r = a + b * theta`.

    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check

    Returns: 
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)

    if savefig:
        plt.figure()
        plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        plt.savefig('./ground_truth.png', dpi=500)
        print('Saved ground truth spiral at {}'.format('./ground_truth.png'))

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    for _ in range(nspiral):
        # don't sample t0 very near the start or the end
        t0_idx = npr.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample

        cc = bool(npr.rand() > .5)  # uniformly select rotation
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)

    return orig_trajs, samp_trajs, orig_ts, samp_ts


class LatentODEfunc(tf.keras.Model):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.fc1 = tf.keras.layers.Dense(nhidden, activation='elu')
        self.fc2 = tf.keras.layers.Dense(nhidden, activation='elu')
        self.fc3 = tf.keras.layers.Dense(latent_dim)
        self.nfe = 0

    def call(self, t, x):
        self.nfe += 1
        x = cast_double(x)

        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(tf.keras.Model):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = tf.keras.layers.Dense(nhidden, activation='tanh')
        self.h2o = tf.keras.layers.Dense(latent_dim * 2)

    def call(self, x, h):
        x = cast_double(x)
        h = cast_double(h)

        combined = tf.concat((x, h), axis=1)
        h = self.i2h(combined)
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return tf.zeros([self.nbatch, self.nhidden], dtype=tf.float64)


class Decoder(tf.keras.Model):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.fc1 = tf.keras.layers.Dense(nhidden, activation='relu')
        self.fc2 = tf.keras.layers.Dense(obs_dim)

    def call(self, z):
        z = cast_double(z)

        out = self.fc1(z)
        out = self.fc2(out)
        return out


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


@func_cast_double
def log_normal_pdf(x, mean, logvar):
    const = tf.convert_to_tensor(np.array([2. * np.pi]), dtype=tf.float64)
    const = move_to_device(const, device)
    const = tf.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / tf.exp(logvar))


@func_cast_double
def normal_kl(mu1, lv1, mu2, lv2):
    v1 = tf.exp(lv1)
    v2 = tf.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def save_states(orig_ts, orig_trajs, samp_ts, samp_trajs):
    ots = orig_ts.numpy()
    otjs = orig_trajs.numpy()
    sts = samp_ts.numpy()
    stjs = samp_trajs.numpy()

    orig_ts_path = os.path.join(args.train_dir, 'orig_ts')
    orig_trajs_path = os.path.join(args.train_dir, 'orig_trajs')
    samp_ts_path = os.path.join(args.train_dir, 'samp_ts')
    samp_trajs_path = os.path.join(args.train_dir, 'samp_trajs')

    np.save(orig_ts_path, ots)
    np.save(orig_trajs_path, otjs)
    np.save(samp_ts_path, sts)
    np.save(samp_trajs_path, stjs)


def restore_states():
    orig_ts_path = os.path.join(args.train_dir, 'orig_ts.npy')
    orig_trajs_path = os.path.join(args.train_dir, 'orig_trajs.npy')
    samp_ts_path = os.path.join(args.train_dir, 'samp_ts.npy')
    samp_trajs_path = os.path.join(args.train_dir, 'samp_trajs.npy')

    ots = tf.convert_to_tensor(np.load(orig_ts_path), dtype=tf.float64)
    otjs = tf.convert_to_tensor(np.load(orig_trajs_path), dtype=tf.float32)
    sts = tf.convert_to_tensor(np.load(samp_ts_path), dtype=tf.float32)
    stjs = tf.convert_to_tensor(np.load(samp_trajs_path), dtype=tf.float32)

    states = dict(orig_ts=ots, orig_trajs=otjs,
                  samp_ts=sts, samp_trajs=stjs)

    return states


if __name__ == '__main__':
    latent_dim = 4
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 2
    nspiral = 1000
    start = 0.
    stop = 6 * np.pi
    noise_std = .3
    a = 0.
    b = .3
    ntotal = 1000
    nsample = 100
    device = 'gpu:' + str(args.gpu) if tf.test.is_gpu_available() else 'cpu'

    with tf.device(device):
        # generate toy spiral data
        orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
            nspiral=nspiral,
            start=start,
            stop=stop,
            noise_std=noise_std,
            a=a, b=b
        )

        orig_ts = tf.convert_to_tensor(orig_ts, dtype=tf.float64)
        orig_trajs = tf.convert_to_tensor(orig_trajs, dtype=tf.float32)
        samp_trajs = tf.convert_to_tensor(samp_trajs, dtype=tf.float32)
        samp_ts = tf.convert_to_tensor(samp_ts, dtype=tf.float32)

        # model
        func = LatentODEfunc(latent_dim, nhidden)
        rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, nspiral)
        dec = Decoder(latent_dim, obs_dim, nhidden)
        optimizer = tf.train.AdamOptimizer(args.lr)
        loss_meter = RunningAverageMeter()

        saver = tf.train.Checkpoint(func=func, rec=rec, dec=dec, optimizer=optimizer)

        if args.train_dir is not None:
            if not os.path.exists(args.train_dir):
                os.makedirs(args.train_dir)
            else:
                if tf.train.checkpoint_exists(args.train_dir):
                    path = tf.train.latest_checkpoint(args.train_dir)

                    if path is not None:
                        saver.restore(path)

                        states = restore_states()
                        orig_trajs = states['orig_trajs']
                        samp_trajs = states['samp_trajs']
                        orig_ts = states['orig_ts']
                        samp_ts = states['samp_ts']
                        print('Loaded ckpt from {}'.format(path))

        for itr in range(1, args.niters + 1):
            # backward in time to infer q(z_0)
            with tf.GradientTape() as tape:
                h = rec.initHidden()
                for t in reversed(range(samp_trajs.shape[1])):
                    obs = samp_trajs[:, t, :]
                    out, h = rec(obs, h)
                qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
                epsilon = tf.convert_to_tensor(np.random.randn(*qz0_mean.shape.as_list()), dtype=qz0_mean.dtype)
                z0 = epsilon * tf.exp(.5 * qz0_logvar) + qz0_mean

                # forward in time and solve ode for reconstructions
                pred_z = tf.transpose(odeint(func, z0, samp_ts), [1, 0, 2])
                pred_x = dec(pred_z)

                # compute loss
                noise_std_ = tf.zeros(pred_x.shape, dtype=tf.float64) + noise_std
                noise_logvar = 2. * tf.log(noise_std_)
                logpx = tf.reduce_sum(log_normal_pdf(
                    samp_trajs, pred_x, noise_logvar), axis=-1)
                logpx = tf.reduce_sum(logpx, axis=-1)
                pz0_mean = pz0_logvar = tf.zeros(z0.shape, dtype=tf.float64)
                analytic_kl = tf.reduce_sum(normal_kl(qz0_mean, qz0_logvar,
                                                      pz0_mean, pz0_logvar), axis=-1)
                loss = tf.reduce_mean(-logpx + analytic_kl, axis=0)

            params = (list(func.variables) + list(dec.variables) + list(rec.variables))
            grad = tape.gradient(loss, params)
            grad_vars = zip(grad, params)

            optimizer.apply_gradients(grad_vars)
            loss_meter.update(loss.numpy())

            print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))

            if itr != 0 and (itr + 1) % 100 == 0:
                if args.train_dir is not None:
                    ckpt_path = os.path.join(args.train_dir, 'ckpt')

                    saver.save(ckpt_path)
                    save_states(orig_ts, orig_trajs, samp_ts, samp_trajs)
                    print('Stored ckpt at {}'.format(ckpt_path))

        print('Training complete after {} iters.'.format(itr))

        if args.visualize:
            # sample from trajectorys' approx. posterior
            h = rec.initHidden()
            for t in reversed(range(samp_trajs.shape[1])):
                obs = samp_trajs[:, t, :]
                out, h = rec(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = tf.convert_to_tensor(np.random.randn(*qz0_mean.shape.as_list()), dtype=tf.float64)
            z0 = epsilon * tf.exp(.5 * qz0_logvar) + qz0_mean
            orig_ts = tf.convert_to_tensor(orig_ts, dtype=tf.float32)

            # take first trajectory for visualization
            z0 = z0[0:1]

            ts_pos = np.linspace(0., 2. * np.pi, num=2000)
            ts_neg = np.linspace(-np.pi, 0., num=2000)[::-1].copy()
            ts_pos = tf.convert_to_tensor(ts_pos, dtype=tf.float32)
            ts_neg = tf.convert_to_tensor(ts_neg, dtype=tf.float32)

            zs_pos = odeint(func, z0, ts_pos)
            zs_neg = odeint(func, z0, ts_neg)

            xs_pos = dec(zs_pos)
            xs_neg = tf.reverse(dec(zs_neg), axis=[0])

            xs_pos = xs_pos.numpy().squeeze(1)
            xs_neg = xs_neg.numpy().squeeze(1)
            orig_traj = orig_trajs[0].numpy()
            samp_traj = samp_trajs[0].numpy()

            # xs_neg = np.clip(xs_neg, xs_pos.min(), xs_pos.max())

            plt.figure()
            plt.plot(orig_traj[:, 0], orig_traj[:, 1],
                     'g', label='true trajectory')
            plt.plot(xs_pos[:, 0], xs_pos[:, 1], 'r',
                     label='learned trajectory (t>0)')
            plt.plot(xs_neg[:, 0], xs_neg[:, 1], 'c',
                     label='learned trajectory (t<0)')
            plt.scatter(samp_traj[:, 0], samp_traj[
                        :, 1], label='sampled data', s=3)
            plt.legend()
            plt.savefig('./vis.png', dpi=500)
            print('Saved visualization figure at {}'.format('./vis.png'))
