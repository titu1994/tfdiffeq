""" Ported from https://github.com/rtqichen/torchdiffeq/blob/master/tests/DETEST/run.py """
import time
import numpy as np
from scipy.stats.mstats import gmean
import tensorflow as tf
from tfdiffeq import odeint
import detest


if tf.version.VERSION.startswith("1."):
    tf.enable_eager_execution()


class NFEDiffEq:

    def __init__(self, diffeq):
        self.diffeq = diffeq
        self.nfe = 0

    def __call__(self, t, y):
        self.nfe += 1
        return self.diffeq(t, y)


def main():

    sol = dict()
    for method in ['dopri5', 'adams']:
        for tol in [1e-3, 1e-6, 1e-9]:
            print('======= {} | tol={:e} ======='.format(method, tol))
            nfes = []
            times = []
            errs = []
            for c in ['A', 'B', 'C', 'D', 'E']:
                for i in ['1', '2', '3', '4', '5']:
                    diffeq, init, _ = getattr(detest, c + i)()
                    t0, y0 = init()
                    diffeq = NFEDiffEq(diffeq)

                    if not c + i in sol:
                        sol[c + i] = odeint(
                            diffeq, y0, tf.stack([t0, tf.convert_to_tensor(20., dtype=tf.float64)]), atol=1e-12, rtol=1e-12,
                            method='dopri5'
                        )[1]
                        diffeq.nfe = 0

                    start_time = time.time()
                    est = odeint(diffeq, y0, tf.stack([t0, tf.convert_to_tensor(20., dtype=tf.float64)]), atol=tol, rtol=tol,
                                 method=method)
                    time_spent = time.time() - start_time

                    error = tf.sqrt(tf.reduce_mean((sol[c + i] - est[1])**2))

                    errs.append(error.numpy())
                    nfes.append(diffeq.nfe)
                    times.append(time_spent)

                    print('{}: NFE {} | Time {} | Err {:e}'.format(c + i, diffeq.nfe, time_spent, error.numpy()))

            print('Total NFE {} | Total Time {} | GeomAvg Error {:e}'.format(np.sum(nfes), np.sum(times), gmean(errs)))


if __name__ == '__main__':
    with tf.device('cpu:0'):
        main()
