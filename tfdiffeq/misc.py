import warnings
import tensorflow as tf


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


def _checked_cast(x, y):
    if x.dtype != y.dtype:
        x = tf.cast(x, y.dtype)

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


def _handle_unused_kwargs(solver, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn('{}: Unexpected arguments {}'.format(solver.__class__.__name__, unused_kwargs))


@tf.function
def _rms_norm(tensor):
    result = tf.sqrt(tf.reduce_mean(tf.square(tensor)))
    return result


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


def _mixed_linf_rms_norm(shapes):
    def _norm(tensor):
        total = 0
        out = []
        for shape in shapes:
            next_total = total + _numel(shape)
            out.append(_rms_norm(tensor[total:next_total]))
            total = next_total
        assert total == _numel(tensor), "Shapes do not total to the full size of the tensor."
        return max(out)

    return _norm


@tf.function
def _linf_norm(tensor):
    return tf.reduce_max(tensor)


def _has_converged(y0, y1, rtol, atol):
    """Checks that each element is within the error tolerance."""
    error_tol = tuple(atol + rtol * tf.maximum(tf.abs(y0_), tf.abs(y1_))
                      for y0_, y1_ in zip(y0, y1))
    error = tuple(tf.abs(y0_ - y1_) for y0_, y1_ in zip(y0, y1))
    return all(tf.reduce_all(error_ < error_tol_) for error_, error_tol_ in zip(error, error_tol))


def _norm(x):
    """Compute RMS norm."""
    if isinstance(x, tf.Tensor):
        return tf.norm(x) / (_numel(x) ** 0.5)
    else:
        return tf.sqrt(sum(tf.norm(x_) ** 2 for x_ in x) / sum(_numel(x_) for x_ in x))


def _select_initial_step(func, t0, y0, order, rtol, atol, norm, f0=None):
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
    -------f
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    t_dtype = t0.dtype
    t0 = move_to_device(_checked_cast(t0, y0[0]), y0[0].device)
    if f0 is None:
        f0 = func(t0, y0)

    scale = atol + tf.abs(y0) * rtol

    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = move_to_device(tf.convert_to_tensor(1e-6, dtype=t0.dtype), t0)
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * f0
    f1 = func(t0 + h0, y1)
    d2 = norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = tf.reduce_max([move_to_device(tf.convert_to_tensor(
            1e-6, dtype=h0.dtype), h0.device), h0 * 1e-3])
    else:
        h1 = (0.01 / max(d1, d2)) ** (1. / float(order + 1))

    result = tf.reduce_min([100 * h0, h1])
    # result = _checked_cast(result, t_dtype)
    return result


# def _compute_error_ratio(error_estimate, error_tol=None, rtol=None, atol=None, y0=None, y1=None):
#     if error_tol is None:
#         assert rtol is not None and atol is not None and y0 is not None and y1 is not None
#         rtol if _is_iterable(rtol) else [rtol] * len(y0)
#         atol if _is_iterable(atol) else [atol] * len(y0)
#
#         error_tol = tuple(
#             atol_ + rtol_ * tf.reduce_max([tf.abs(y0_), tf.abs(y1_)])
#             for atol_, rtol_, y0_, y1_ in zip(atol, rtol, y0, y1)
#         )
#     error_ratio = tuple(error_estimate_ / error_tol_ for error_estimate_,
#                         error_tol_ in zip(error_estimate, error_tol))
#     mean_sq_error_ratio = tuple(tf.reduce_mean(error_ratio_ * error_ratio_)
#                                 for error_ratio_ in error_ratio)
#     return mean_sq_error_ratio


def _compute_error_ratio(error_estimate, rtol, atol, y0, y1, norm):
    error_tol = atol + rtol * tf.reduce_max(tf.abs(y0), tf.abs(y1))
    return norm(error_estimate / error_tol)


def _optimal_step_size(last_step, error_ratio, safety, ifactor, dfactor, order):
    """Calculate the optimal size for the next step."""
    if error_ratio == 0:
        return last_step * ifactor

    if error_ratio < 1:
        with tf.device(error_ratio.device):
            dfactor = tf.convert_to_tensor(1, dtype=last_step.dtype)

    # error_ratio = tf.sqrt(error_ratio)
    error_ratio = _checked_cast(error_ratio, last_step)
    error_ratio = move_to_device(error_ratio, last_step.device)

    with tf.device(last_step.device):
        exponent = tf.convert_to_tensor(1. / order, dtype=last_step.dtype)
        # exponent = tf.cast(exponent, last_step.dtype)
        # exponent = move_to_device(exponent, last_step.device)

    # factor = tf.reduce_max(
    #     [1. / ifactor, tf.reduce_min([error_ratio ** exponent / safety, 1. / dfactor])])
    # return last_step / factor
    factor = tf.reduce_min(ifactor, tf.reduce_max(safety / error_ratio ** exponent, dfactor))
    return last_step * factor


@tf.function
def _decreasing(t):
    return tf.reduce_all(t[1:] < t[:-1])


def _assert_increasing(name, t):
    assert tf.reduce_all(t[1:] > t[:-1]), '{} must be strictly increasing or decreasing'.format(name)


def _assert_one_dimensional(name, t):
    assert len(t.shape) == 1, "{} must be one dimensional".format(name)


def _assert_floating(name, t):
    if not _is_floating_tensor(t):
        raise TypeError('`{}` must be a floating point Tensor but is a {}'.format(name, t.dtype))


def _tuple_tol(name, tol, shapes):
    try:
        iter(tol)
    except TypeError:
        return tol
    tol = tuple(tol)
    assert len(tol) == len(shapes), "If using tupled {} it must have the same length as the tuple y0".format(name)
    tol = [tf.broadcast_to(tf.convert_to_tensor(tol_), _numel(shape)) for tol_, shape in zip(tol, shapes)]
    return tf.concat(tol, axis=0)


def _flat_to_shape(tensor, length, shapes):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + _numel(shape)
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(tf.reshape(tensor[..., total:next_total], (*length, *shape)))
        total = next_total
    return tuple(tensor_list)


class _TupleFunc(tf.keras.Model):
    def __init__(self, base_func, shapes, **kwargs):
        super(_TupleFunc, self).__init__(**kwargs)
        self.base_func = base_func
        self.shapes = shapes

    def call(self, t, y):
        f = self.base_func(t, _flat_to_shape(y, (), self.shapes))
        return tf.concat([tf.reshape(f_, [-1]) for f_ in f])


class _ReverseFunc(tf.keras.Model):
    def __init__(self, base_func, **kwargs):
        super(_ReverseFunc, self).__init__(**kwargs)
        self.base_func = base_func

    def call(self, t, y):
        return -self.base_func(-t, y)


def _check_inputs(func, y0, t, rtol, atol, method, options, SOLVERS):
    # Normalise to tensor (non-tupled) input
    shapes = None
    if not tf.is_tensor(y0):
        assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
        shapes = [y0_.shape for y0_ in y0]
        rtol = _tuple_tol('rtol', rtol, shapes)
        atol = _tuple_tol('atol', atol, shapes)
        y0 = tf.concat([tf.reshape(y0_, [-1]) for y0_ in y0])
        func = _TupleFunc(func, shapes)

    _assert_floating('y0', y0)

    # Normalise method and options
    if options is None:
        options = {}
    else:
        options = options.copy()

    if method is None:
        method = 'dopri5'

    if method not in SOLVERS:
        raise ValueError(
            'Invalid method "{}". Must be one of {}'.format(method, '{"' + '", "'.join(SOLVERS.keys()) + '"}.'))

    try:
        grid_points = options['grid_points']
    except KeyError:
        pass

    else:
        assert tf.is_tensor(grid_points), 'grid_points must be a tf.Tensor'
        _assert_one_dimensional('grid_points', grid_points)

        assert not grid_points.requires_grad, "grid_points cannot require gradient"
        _assert_floating('grid_points', grid_points)

    if 'norm' not in options:
        if shapes is None:
            # L2 norm over a single input
            options['norm'] = _rms_norm
        else:
            # Mixed Linf/L2 norm over tupled input (chosen mostly just for backward compatibility reasons)
            options['norm'] = _mixed_linf_rms_norm(shapes)

    # Normalise time
    assert tf.is_tensor(t), 't must be a tf.Tensor'

    _assert_one_dimensional('t', t)
    _assert_floating('t', t)

    if _decreasing(t):
        t = -t
        func = _ReverseFunc(func)
        try:
            grid_points = options['grid_points']
        except KeyError:
            pass
        else:
            options['grid_points'] = -grid_points

    # Can only do after having normalised time
    _assert_increasing('t', t)

    try:
        grid_points = options['grid_points']
    except KeyError:
        pass
    else:
        _assert_increasing('grid_points', grid_points)

    # Tol checking
    if isinstance(rtol, tf.Variable):
        raise ValueError("rtol cannot require gradient")

    if isinstance(atol, tf.Variable):
        raise ValueError("atol cannot require gradient")

    # Backward compatibility: Allow t and y0 to be on different devices
    if t.device != y0.device:
        warnings.warn("t is not on the same device as y0. Coercing to y0.device.")

        with tf.device(y0.device):
            t = tf.identity(t)
    # ~Backward compatibility

    return shapes, func, y0, t, rtol, atol, method, options

# def _check_inputs(func, y0, t):
#     tensor_input = False
#     if isinstance(y0, tf.Tensor):
#         tensor_input = True
#
#         if not tf.is_tensor(y0):
#             warnings.warn('Input is *not* an EagerTensor ! '
#                           'Dummy op with zeros will be performed instead.')
#
#             y0 = tf.convert_to_tensor(tf.zeros(y0.shape))
#
#         y0 = (y0,)
#         _base_nontuple_func_ = func
#         func = lambda t, y: (_base_nontuple_func_(t, y[0]),)
#
#     assert isinstance(y0, tuple), 'y0 must be either a tf.Tensor or a tuple'
#     if ((type(y0) == tuple) or (type(y0) == list)):
#         if not tensor_input:
#             y0_type = type(y0)
#             y0 = list(y0)
#
#             for i in range(len(y0)):
#                 assert isinstance(y0[i], tf.Tensor), 'each element must be a tf.Tensor ' \
#                                                      'but received {}'.format(type(y0[i]))
#             y0 = y0_type(y0)  # return to same type
#     else:
#         raise ValueError('y0 must be either a tf.Tensor or a tuple')
#
#     if _decreasing(t):
#         t = -t
#         _base_reverse_func = func
#         func = lambda t, y: tuple(-f_ for f_ in _base_reverse_func(-t, y))
#
#     for y0_ in y0:
#         if not tf.debugging.is_numeric_tensor(y0_):
#             raise TypeError('`y0` must be a floating point Tensor but is a {}'.format(y0_.dtype))
#     if not tf.debugging.is_numeric_tensor(t):
#         raise TypeError('`t` must be a floating point Tensor but is a {}'.format(t.dtype))
#
#     return tensor_input, func, y0, t
