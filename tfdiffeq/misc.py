import six
import warnings
from typing import Iterable

import tensorflow as tf
from tensorflow.python import ops


def cast_double(x):
    if isinstance(x, Iterable):
        try:
            x = tf.cast(x, tf.float64)
        except Exception:
            xn = []
            for xi in x:
                xi = cast_double(xi)
                xn.append(xi)

            x = type(x)(xn)

    else:
        if hasattr(x, 'dtype') and x.dtype != tf.float64:
            x = tf.cast(x, tf.float64)

        elif type(x) != float:
            x = float(x)

    return x


def func_cast_double(func):
    """ Casts all Tensor arguments to float64 """

    @six.wraps(func)
    def wrapper(*args, **kwargs):
        cast_args = []
        for arg in args:
            if isinstance(arg, tf.Tensor) or isinstance(arg, tf.Variable):
                arg = cast_double(arg)
            elif type(arg) == tuple or type(tuple) == list:
                arg = cast_double(arg)

            cast_args.append(arg)

        result = func(*cast_args, **kwargs)
        return result

    return wrapper


def move_to_device(x, device):
    """ Utility function to move a tensor to a device """

    if device is None:
        return x

    # tf.Variables cannot be moved to a device
    if not isinstance(x, tf.Tensor):
        return x

    if isinstance(device, tf.Tensor):
        device = device.device

    # check if device is empty string
    if len(device) == 0:
        return x

    if '/' in device:
        device = device.replace('/', '')

    splits = device.split(':')[-2:]
    device, id = splits
    id = int(id)

    x_device = x.device.lower()

    if 'cpu' in device.lower() and 'cpu' not in x_device:
        with tf.device('cpu'):
            x = tf.identity(x)

    elif 'gpu' in device.lower() and 'gpu' not in x_device:
        with tf.device(device):
            x = tf.identity(x)

    return x


def _check_len(x):
    """ Utility function to get the length of the tensor """
    if hasattr(x, 'shape'):
        return x.shape[0]
    else:
        return len(x)


def _numel(x, dtype=None):
    """ Compute number of elements in the input tensor """
    if dtype is None:
        dtype = x.dtype
    return tf.cast(tf.reduce_prod(x.shape), dtype)


def _is_floating_tensor(x):
    return x.dtype in [tf.float16, tf.float32, tf.float64]


def _flatten(sequence):
    flat = [tf.reshape(p, [-1]) for p in sequence]
    return tf.concat(flat, 0) if len(flat) > 0 else tf.convert_to_tensor([])


def _flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [
        tf.reshape(p, [-1]) if p is not None else tf.reshape(tf.zeros_like(q), [-1])
        for p, q in zip(sequence, like_sequence)
    ]
    return tf.concat(flat, 0) if len(flat) > 0 else tf.convert_to_tensor([])


def _flatten_recover(sequence):
    shapes = [p.shape for p in sequence]
    numels = [tf.cast(tf.reduce_prod(shape), tf.int32).numpy() for shape in shapes]

    flat = [tf.reshape(p, [-1]) for p in sequence]
    out = tf.concat(flat, 0) if len(flat) > 0 else tf.convert_to_tensor([])

    def _recover_shapes(flat):
        params_splits = tf.split(flat, numels)
        param_list = [tf.reshape(p, shape)
                      for p, shape in zip(params_splits, shapes)]

        return param_list

    return out, _recover_shapes


def _flatten_convert_none_to_zeros_recover(sequence, like_sequence):
    shapes = [p.shape if p is not None else q.shape for p, q in zip(sequence, like_sequence)]
    numels = [tf.cast(tf.reduce_prod(shape), tf.int32).numpy() for shape in shapes]

    flat = [
        tf.reshape(p, [-1]) if p is not None else tf.reshape(tf.zeros_like(q), [-1])
        for p, q in zip(sequence, like_sequence)
    ]
    out = tf.concat(flat, 0) if len(flat) > 0 else tf.convert_to_tensor([])

    def _recover_shapes(flat):
        params_splits = tf.split(flat, numels)
        param_list = [tf.reshape(p, shape)
                      for p, shape in zip(params_splits, shapes)]

        return param_list

    return out, _recover_shapes


def _possibly_nonzero(x):
    return isinstance(x, tf.Tensor) or x != 0


def _scaled_dot_product(scale, xs, ys):
    """Calculate a scaled, vector inner product between lists of Tensors."""
    # Using _possibly_nonzero lets us avoid wasted computation.
    return sum([(scale * x) * y for x, y in zip(xs, ys) if _possibly_nonzero(x) or _possibly_nonzero(y)])


def _dot_product(xs, ys):
    """Calculate the vector inner product between two lists of Tensors."""
    return sum([x * y for x, y in zip(xs, ys)])


def _has_converged(y0, y1, rtol, atol):
    """Checks that each element is within the error tolerance."""
    error_tol = tuple(atol + rtol * tf.maximum(tf.abs(y0_), tf.abs(y1_)) for y0_, y1_ in zip(y0, y1))
    error = tuple(tf.abs(y0_ - y1_) for y0_, y1_ in zip(y0, y1))
    return all(tf.reduce_all(error_ < error_tol_) for error_, error_tol_ in zip(error, error_tol))


def _convert_to_tensor(a, dtype=None, device=None):
    if not isinstance(a, tf.Tensor):
        a = tf.convert_to_tensor(a)
    if dtype is not None:
        a = tf.cast(a, dtype)
    if device is not None:
        a = move_to_device(a, device)
    return a


def _is_finite(tensor):
    _check = tf.cast(tf.math.is_inf(tensor), tf.int64) + tf.cast(tf.math.is_nan(tensor), tf.int64)
    _check = tf.cast(_check, tf.bool)
    return not tf.reduce_any(_check)


def _decreasing(t):
    v = tf.reduce_all(t[1:] < t[:-1])
    return v


def _assert_increasing(t):
    assert tf.reduce_all(t[1:] > t[:-1]), 't must be strictly increasing or decrasing'


def _is_iterable(inputs):
    try:
        iter(inputs)
        return True
    except TypeError:
        return False


def _norm(x):
    """Compute RMS norm."""
    if isinstance(x, tf.Tensor):
        return tf.norm(x) / (_numel(x) ** 0.5)
    else:
        return tf.sqrt(sum(tf.norm(x_) ** 2 for x_ in x) / sum(_numel(x_) for x_ in x))


def _handle_unused_kwargs(solver, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn('{}: Unexpected arguments {}'.format(solver.__class__.__name__, unused_kwargs))


def _select_initial_step(fun, t0, y0, order, rtol, atol, f0=None):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    direction : float
        Integration direction.
    order : float
        Method order.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    t0 = move_to_device(t0, y0[0].device)
    t0 = cast_double(t0)

    y0 = cast_double(y0)

    if f0 is None:
        f0 = fun(t0, y0)

    f0 = cast_double(f0)

    if hasattr(y0, 'shape'):
        count = y0.shape[0]
    else:
        count = len(y0)

    rtol = rtol if _is_iterable(rtol) else [rtol] * count
    atol = atol if _is_iterable(atol) else [atol] * count

    rtol = [cast_double(r) for r in rtol]
    atol = [cast_double(a) for a in atol]

    scale = tuple(atol_ + tf.abs(y0_) * rtol_ for y0_, atol_, rtol_ in zip(y0, atol, rtol))
    scale = [cast_double(s) for s in scale]

    d0 = tuple(_norm(y0_ / scale_) for y0_, scale_ in zip(y0, scale))
    d1 = tuple(_norm(f0_ / scale_) for f0_, scale_ in zip(f0, scale))

    if max(d0).numpy() < 1e-5 or max(d1).numpy() < 1e-5:
        h0 = move_to_device(tf.convert_to_tensor(1e-6), t0)
    else:
        h0 = 0.01 * max(d0_ / d1_ for d0_, d1_ in zip(d0, d1))

    h0 = cast_double(h0)

    y1 = tuple(y0_ + h0 * f0_ for y0_, f0_ in zip(y0, f0))
    f1 = fun(t0 + h0, y1)
    f1 = cast_double(f1)

    d2 = tuple(_norm((f1_ - f0_) / scale_) / h0 for f1_, f0_, scale_ in zip(f1, f0, scale))

    if max(d1).numpy() <= 1e-15 and max(d2).numpy() <= 1e-15:
        h1 = tf.reduce_max([move_to_device(tf.convert_to_tensor(1e-6, dtype=tf.float64), h0.device), h0 * 1e-3])
    else:
        h1 = (0.01 / max(d1 + d2)) ** (1. / float(order + 1))

    return tf.reduce_min([100 * h0, h1])


def _compute_error_ratio(error_estimate, error_tol=None, rtol=None, atol=None, y0=None, y1=None):
    if error_tol is None:
        assert rtol is not None and atol is not None and y0 is not None and y1 is not None
        rtol if _is_iterable(rtol) else [rtol] * len(y0)
        atol if _is_iterable(atol) else [atol] * len(y0)
        y0 = cast_double(y0)

        error_tol = tuple(
            atol_ + rtol_ * tf.reduce_max([tf.abs(y0_), tf.abs(y1_)])
            for atol_, rtol_, y0_, y1_ in zip(atol, rtol, y0, y1)
        )
    error_ratio = tuple(error_estimate_ / error_tol_ for error_estimate_, error_tol_ in zip(error_estimate, error_tol))
    mean_sq_error_ratio = tuple(tf.reduce_mean(error_ratio_ * error_ratio_) for error_ratio_ in error_ratio)
    return mean_sq_error_ratio


def _optimal_step_size(last_step, mean_error_ratio, safety=0.9, ifactor=10.0, dfactor=0.2, order=5):
    """Calculate the optimal size for the next step."""
    mean_error_ratio = max(mean_error_ratio)  # Compute step size based on highest ratio.

    if mean_error_ratio == 0:
        return last_step * ifactor

    if mean_error_ratio < 1:
        dfactor = _convert_to_tensor(1, dtype=tf.float64, device=mean_error_ratio.device)

    error_ratio = tf.sqrt(mean_error_ratio)
    error_ratio = cast_double(error_ratio)
    error_ratio = move_to_device(error_ratio, last_step.device)

    exponent = tf.convert_to_tensor(1. / order)
    exponent = cast_double(exponent)
    exponent = move_to_device(exponent, last_step.device)

    factor = tf.reduce_max([1. / ifactor, tf.reduce_min([error_ratio ** exponent / safety, 1. / dfactor])])
    return last_step / factor


def _check_inputs(func, y0, t):
    tensor_input = False
    if isinstance(y0, tf.Tensor):
        tensor_input = True

        if not isinstance(y0, ops.EagerTensor):
            warnings.warn('Input is *not* an EagerTensor ! '
                          'Dummy op with zeros will be performed instead.')

            y0 = tf.convert_to_tensor(tf.zeros(y0.shape))

        y0 = (y0,)
        _base_nontuple_func_ = func
        func = lambda t, y: (_base_nontuple_func_(t, y[0]),)

    if ((type(y0) == tuple) or (type(y0) == list)):
        if not tensor_input:
            y0_type = type(y0)
            y0 = list(y0)

            for i in range(len(y0)):
                assert isinstance(y0[i], tf.Tensor), 'each element must be a tf.Tensor ' \
                                                     'but received {}'.format(type(y0[i]))

                if not isinstance(y0[i], ops.EagerTensor):
                    warnings.warn('Input %d (zero-based) is *not* an EagerTensor ! '
                                  'Dummy op with zeros will be performed instead.' % (i))

                    y0[i] = tf.convert_to_tensor(tf.zeros(y0[i].shape))

            y0 = y0_type(y0)  # return to same type
    else:
        raise ValueError('y0 must be either a tf.Tensor or a tuple')

    if _decreasing(t):
        t = -t
        _base_reverse_func = func
        func = lambda t, y: tuple(-f_ for f_ in _base_reverse_func(-t, y))

    for y0_ in y0:
        if not tf.debugging.is_numeric_tensor(y0_):
            raise TypeError('`y0` must be a floating point Tensor but is a {}'.format(y0_.dtype))
    if not tf.debugging.is_numeric_tensor(t):
        raise TypeError('`t` must be a floating point Tensor but is a {}'.format(t.dtype))

    return tensor_input, func, y0, t
