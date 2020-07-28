import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

plt.style.use('seaborn-paper')


def plot_phase_portrait(func, t0=None, xlims=None, ylims=None, num_points=20,
                        xlabel='X', ylabel='Y', ip_rank=None):
    """
    Plots the phase portrait of a system of ODEs containing two dimensions.

    Args:
        func: Must be a callable function with the signature func(t, y)
            where t is a tf.float64 tensor representing the independent
            time dimension and y is a tensor of shape [2] if `ip_rank`
            if not specified, otherwise a tensor of rank = `ip_rank`.
            The function must emit exactly 2 outputs, in any shape as it
            will be flattened.

        t0: Initial timestep value. Can be None, which defaults to a value
            of 0.

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
            1 if not passed a value. All axis but one must have dimension
            equal to 1. All permutations are allowed, since it will be
            squeezed down to a vector of rank 1.
            Rank 1: Vector output. Shape = [N]
            Rank 2: Matrix output. Shape = [1, N] or [N, 1] etc.

    Returns:
        Nothing is returned. The plot is not shown via plt.show() either,
        therefore it must be explicitly called using `plt.show()`.

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

    t = t0 if t0 is not None else 0.0
    t = tf.convert_to_tensor(t, dtype=tf.float64)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xi = X[i, j]
            yi = Y[i, j]
            inp = tf.stack([xi, yi])

            # prepare input shape for the function
            if ip_rank != 1:
                o = [1] * (ip_rank - 1) + [2]  # shape = [1, ..., 2]
                inp = tf.reshape(inp, o)

            out = func(t, inp)

            # check whether function returns a Tensor or a ndarray
            if hasattr(out, 'numpy'):
                out = out.numpy()

            # reshape the results to be a vector
            out = np.squeeze(out)

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
            final dimension *must* have a rank of 2.

        xlabel: Label of the X axis.

        ylabel: Label of the Y axis.

    Returns:
        Nothing is returned. The plot is not shown via plt.show() either,
        therefore it must be explicitly called using `plt.show()`.

        This is done so that the phase plot and vector field plots can be
        visualized simultaneously.
    """
    if hasattr(result, 'numpy'):
        result = result.numpy()  # convert Tensor back to numpy

    result = np.squeeze(result)

    if result.ndim > 2:
        raise ValueError("Passed tensor or ndarray must be at most a 2D tensor after squeeze.")

    plt.plot(result[:, 0], result[:, 1], 'b-')
    plt.plot([result[0, 0]], [result[0, 1]], 'o', label='start')  # start state
    plt.plot([result[-1, 0]], [result[-1, 1]], 's', label='end')  # end state

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()


def plot_results(time, result, labels=None, dependent_vars=False, **fig_args):
    """
    Plots the result of an integration call.

    Args:
        time: a tf.Tensor or a numpy ndarray describing the time steps
            of integration. Can be any rank with excess 1 dimensions.
            However, the final dimension *must* be a vector of rank 1.

        result: a tf.Tensor or a numpy ndarray describing the result.
            Can be any rank with excess 1 dimensions. However, the
            final dimension *must* have a rank of 2.

        labels: A list of strings for the variable names on the plot.

        dependent_vars: If the resultant dimensions depend on each other,
            then a 2-d or 3-d plot can be made to display their interaction.

    Returns:
        A Matplotlib Axes object for dependent variables, otherwise noting.
        The plot is not shown via plt.show() either, therefore it must be
        explicitly called using `plt.show()`.

    """
    if hasattr(time, 'numpy'):
        time = time.numpy()  # convert Tensor back to numpy

    if hasattr(result, 'numpy'):
        result = result.numpy()  # convert Tensor back to numpy

    # remove excess dimensions
    time = np.squeeze(time)
    result = np.squeeze(result)

    if result.ndim == 1:
        result = np.expand_dims(result, -1)  # treat result as a matrix always

    if result.ndim != 2:
        raise ValueError("`result` must be a matrix of shape [:, 2/3] after "
                         "removal of excess dimensions.")

    num_vars = result.shape[-1]

    # setup labels
    if labels is not None:
        if type(labels) not in (list, tuple):
            labels = [labels]

        if len(labels) != num_vars:
            raise ValueError("If labels are provided, there must be one label "
                             "per variable in the result matrix. Found %d "
                             "labels for %d variables." % (len(labels), num_vars))

    else:
        labels = ["v%d" % (v_id + 1) for v_id in range(num_vars)]

    if not dependent_vars:
        for var_id, var_label in enumerate(labels):
            plt.plot(time, result[:, var_id], label=var_label)

        plt.legend()

    else:
        if num_vars not in (2, 3):
            raise ValueError("For dependent variable plotting, only 2 or 3 variables "
                             "are supported. Provided number of variables = %d" % num_vars)

        if num_vars == 2:
            fig = plt.figure(**fig_args)
            ax = fig.gca()

            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])

            ax.plot(result[:, 0], result[:, 1])

        elif num_vars == 3:
            from mpl_toolkits.mplot3d import Axes3D  # needed for plotting in 3d
            _ = Axes3D

            fig = plt.figure(**fig_args)
            ax = fig.gca(projection='3d')

            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])

            ax.plot(result[:, 0], result[:, 1], result[:, 2])

        return ax
