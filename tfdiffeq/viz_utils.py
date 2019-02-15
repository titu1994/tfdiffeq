import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

plt.style.use('seaborn-paper')


def plot_phase_plot(func, xlims=None, ylims=None, num_points=20, xlabel='X', ylabel='Y', ip_rank=None):
    """
    Plots the phase portrait of a system of ODEs containing two dimensions.

    Args:
        func: Must be a callable function with the signature func(t, y)
            where t is a tf.float64 tensor representing the independent
            time dimension and y is a tensor of shape [2] if `ip_rank`
            if not specified, otherwise a tensor of rank = `ip_rank`.
            The function must emit exactly 2 outputs, in any shape as it
            will be flattened.

        xlims: A list of 2 floating point numbers. Declares the range of
            the `x` space that will be plotted. If None, defaults to the
            values of [-2.0, 2.0].

        ylims: A list of 2 floating point numbers. Declares the range of
            the `y` space that will be plotted. If None, defaults to the
            values of [-2.0, 2.0].

        num_points: Number of points to sample per dimension.

        xlabel: Label of the X axis.

        ylabel: Label of the Y axis.

        ip_rank: Declares the rank of the passed callable. Defaults to rank
            1 if not passed a value.

    Returns:
        Nothing is returned. The plot is not shown via plt.show() either,
        therefore it must be explicitly called by the called using `plt.show()`.

        This is done so that the phase plot and vector field plots can be
        visualized simultaneously.
    """

    if xlims is not None and len(xlims) != 2:
        raise ValueError('`xlims` must be a list of 2 floating point numbers')

    if ylims is not None and len(ylims) != 2:
        raise ValueError('`ylims` must be a list of 2 floating point numbers')

    if xlims is None:
        xlims = [-2., 2.]

    if ylims is None:
        ylims = [-2., 2.]

    if ip_rank is None:
        ip_rank = 1

    assert ip_rank >= 1, "`ip_rank` must be greater than or equal to 1."

    x = np.linspace(float(xlims[0]), float(xlims[1]), num=num_points)
    y = np.linspace(float(ylims[0]), float(ylims[1]), num=num_points)

    X, Y = np.meshgrid(x, y)

    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    t = tf.convert_to_tensor(0.0, dtype=tf.float64)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xi = X[i, j]
            yi = Y[i, j]
            inp = tf.stack([xi, yi])

            if ip_rank != 1:
                o = [1] * (ip_rank - 1) + [2]  # shape = [1, ..., 2]
                inp = tf.reshape(inp, o)

            out = func(t, inp).numpy()

            if ip_rank != 1:
                out = np.reshape(out, [-1])

            u[i, j] = out[0]
            v[i, j] = out[1]

    Q = plt.quiver(X, Y, u, v, color='black')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_vector_field(result, xlabel='X', ylabel='Y'):
    """
    Plots the vector field of the result of an integration call.

    Args:
        result: a tf.Tensor or a numpy ndarray describing the result.
            Can be any rank with excess 1 dimensions. However, the
            final dimension *must* have a dimension of 2.

        xlabel: Label of the X axis.

        ylabel: Label of the Y axis.

    Returns:
        Nothing is returned. The plot is not shown via plt.show() either,
        therefore it must be explicitly called by the called using `plt.show()`.

        This is done so that the phase plot and vector field plots can be
        visualized simultaneously.
    """
    if hasattr(result, 'numpy'):
        result = result.numpy()  # convert tensor back to numpy

    result = np.squeeze(result)

    if result.ndim > 2:
        raise ValueError("Passed tensor or ndarray must be at most a 2D tensor after squeeze.")

    plt.plot(result[:, 0], result[:, 1], 'b-')
    plt.plot([result[0, 0]], [result[0, 1]], 'o', label='start')  # start state
    plt.plot([result[-1, 0]], [result[-1, 1]], 's', label='end')  # end state

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
