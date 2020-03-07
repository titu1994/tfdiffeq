import warnings
from itertools import product
from typing import Iterable

import numpy as np
import tensorflow as tf

"""
PORTED FROM https://pytorch.org/docs/stable/_modules/torch/autograd/gradcheck.html
"""


def _numel(x):
    """ Compute number of elements in the input tensor """
    return tf.cast(tf.reduce_prod(x.shape), x.dtype)


def _requires_grad(x):
    return True  # get_gradient_function(x) is not None


def allclose(x, y, rtol=1e-5, atol=1e-8):
    return tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol)


def make_jacobian(input, num_out):
    if isinstance(input, tf.Tensor):
        if not tf.debugging.is_numeric_tensor(input):
            return None
        if not _requires_grad(input):
            return None
        return tf.Variable(tf.zeros([_numel(input), num_out], dtype=input.dtype), trainable=False)
    elif isinstance(input, Iterable) and not isinstance(input, str):
        jacobians = list(filter(
            lambda x: x is not None, (make_jacobian(elem, num_out) for elem in input)))
        if not jacobians:
            return None
        return type(input)(jacobians)
    else:
        return None


def iter_tensors(x, only_requiring_grad=False):
    if isinstance(x, tf.Tensor):
        if not only_requiring_grad:
            yield x
    elif isinstance(x, Iterable) and not isinstance(x, str):
        if hasattr(x, 'shape'):
            count = x.shape[0]
        else:
            count = len(x)

        for i in range(count):
            for result in iter_tensors(x[i], only_requiring_grad):
                yield result


# `input` is input to `fn`
# `target` is the Tensors wrt whom Jacobians are calculated (default=`input`)
#
# Note that `target` may not even be part of `input` to `fn`, so please be
# **very careful** in this to not clone `target`.
def get_numerical_jacobian(fn, input, target=None, eps=1e-3):
    if target is None:
        target = input
    output_size = _numel(fn(input))
    jacobian = make_jacobian(target, output_size)

    # It's much easier to iterate over flattened lists of tensors.
    # These are reference to the same objects in jacobian, so any changes
    # will be reflected in it as well.
    x_tensors = [t for t in iter_tensors(target, True)]
    j_tensors = [t for t in iter_tensors(jacobian)]

    for x_tensor, d_tensor in zip(x_tensors, j_tensors):
        # need data here to get around the version check because without .data,
        # the following code updates version but doesn't change content
        x_tensor = x_tensor.numpy()
        for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.shape])):
            orig = x_tensor[x_idx]
            x_tensor[x_idx] = orig - eps
            outa = tf.identity(fn(input))
            x_tensor[x_idx] = orig + eps
            outb = tf.identity(fn(input))
            x_tensor[x_idx] = orig

            r = (outb - outa) / (2 * eps)
            d_tensor[d_idx] = tf.reshape(r, [-1])

    return jacobian


def get_analytical_jacobian(input, output):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input)
        tape.watch(output)
        diff_input_list = list(iter_tensors(input, True))
        jacobian = make_jacobian(input, _numel(output))
        jacobian_reentrant = make_jacobian(input, _numel(output))
        grad_output = tf.zeros_like(output)
        flat_grad_output = tf.reshape(grad_output, [-1])
        reentrant = True
        correct_grad_sizes = True

    for i in range(tf.cast(_numel(flat_grad_output), tf.int64)):
        flat_grad_output *= 0.
        add_one = [0] * (flat_grad_output.shape[0])
        add_one[0] = 1
        flat_grad_output = flat_grad_output + add_one

        for jacobian_c in (jacobian, jacobian_reentrant):

            grads_input = tape.gradient(output, diff_input_list, grad_output)

            # grads_input = torch.autograd.grad(output, diff_input_list, grad_output,
            #                                   retain_graph=True, allow_unused=True)

            for jacobian_x, d_x, x in zip(jacobian_c, grads_input, diff_input_list):
                if d_x is not None and d_x.shape != x.shape:
                    correct_grad_sizes = False
                elif _numel(jacobian_x) != 0:
                    if d_x is None:
                        zeros = np.zeros(jacobian_x[:, i].shape.as_list())
                        tf.assign(jacobian_x[:, i], zeros)
                    else:
                        d_x_dense = tf.sparse.to_dense(d_x) if isinstance(d_x, tf.SparseTensor) else d_x
                        assert _numel(jacobian_x[:, i]) == _numel(d_x_dense)

                        tf.assign(jacobian_x[:, i], tf.reshape(d_x_dense, [-1]))

    for jacobian_x, jacobian_reentrant_x in zip(jacobian, jacobian_reentrant):
        if _numel(jacobian_x) != 0 and tf.not_equal(tf.reduce_max(tf.abs(jacobian_x - jacobian_reentrant_x)), 0.):
            reentrant = False

    return jacobian, reentrant, correct_grad_sizes


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


def _differentiable_outputs(x):
    return tuple(o for o in _as_tuple(x) if _requires_grad(o))


def gradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True):
    r"""Check gradients computed via small finite differences against analytical
    gradients w.r.t. tensors in :attr:`inputs` that are of floating point type
    and with ``requires_grad=True``.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    .. note::
        The default values are designed for :attr:`input` of double precision.
        This check will likely fail if :attr:`input` is of less precision, e.g.,
        ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` has overlapping memory, i.e.,
       different indices pointing to the same memory address (e.g., from
       :func:`torch.expand`), this check will likely fail because the numerical
       gradients computed by point perturbation at such indices will change
       values at all other indices that share the same memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.

    Returns:
        True if all differences satisfy allclose condition
    """
    tupled_inputs = _as_tuple(inputs)

    # Make sure that gradients are saved for all inputs
    any_input_requiring_grad = False
    for inp in tupled_inputs:
        if isinstance(inp, tf.Tensor):
            if _requires_grad(inp):
                if inp.dtype != tf.float64:
                    warnings.warn(
                        'At least one of the inputs that requires gradient '
                        'is not of double precision floating point. '
                        'This check will likely fail if all the inputs are '
                        'not of double precision floating point. ')
                any_input_requiring_grad = True
            # inp.retain_grad()
    if not any_input_requiring_grad:
        raise ValueError(
            'gradcheck expects at least one input tensor to require gradient, '
            'but none of the them have requires_grad=True.')

    output = _differentiable_outputs(func(*tupled_inputs))

    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    for i, o in enumerate(output):
        if not _requires_grad(o):
            continue

        def fn(input):
            return _as_tuple(func(*input))[i]

        analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(tupled_inputs, o)
        numerical = get_numerical_jacobian(fn, tupled_inputs, eps=eps)

        if not correct_grad_sizes:
            return fail_test('Analytical gradient has incorrect size')

        for j, (a, n) in enumerate(zip(analytical, numerical)):
            if _numel(a) != 0 or _numel(n) != 0:
                if not allclose(a, n, rtol, atol):
                    return fail_test('Jacobian mismatch for output %d with respect to input %d,\n'
                                     'numerical:%s\nanalytical:%s\n' % (i, j, n, a))

        if not reentrant:
            return fail_test('Backward is not reentrant, i.e., running backward with same '
                             'input and grad_output multiple times gives different values, '
                             'although analytical gradient matches numerical gradient')

    # check if the backward multiplies by grad_output
    with tf.GradientTape(persistent=True) as tape:
        output = _differentiable_outputs(func(*tupled_inputs))

    if any([_requires_grad(o) for o in output]):
        diff_input_list = list(iter_tensors(tupled_inputs, True))
        grads_input = tape.gradient(output, diff_input_list, [tf.zeros_like(o) for o in output])

        if not len(grads_input) == 0:
            raise RuntimeError("no Tensors requiring grad found in input")

        # grads_input = torch.autograd.grad(output, diff_input_list, [torch.zeros_like(o) for o in output],
        #                                   allow_unused=True)
        for gi, i in zip(grads_input, diff_input_list):
            if gi is None:
                continue
            if not tf.reduce_all(tf.equal(gi, 0)):
                return fail_test('backward not multiplied by grad_output')
            if gi.dtype != i.dtype:
                return fail_test("grad is incorrect type")
            if gi.shape != i.shape:
                return fail_test('grad is incorrect size')

    return True
