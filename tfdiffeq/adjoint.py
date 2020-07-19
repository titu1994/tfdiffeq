import numpy as np
from typing import Iterable

import tensorflow as tf
from tensorflow.python.eager.context import eager_mode

from tfdiffeq import odeint
from tfdiffeq.misc import (_flatten, _flatten_convert_none_to_zeros,
                           move_to_device, cast_double, func_cast_double,
                           _check_len, _numel, _convert_to_tensor)


class _Arguments(object):

    def __init__(self, func, method, options, rtol, atol, adjoint_method, adjoint_rtol, adjoint_atol, adjoint_options):
        self.func = func
        self.method = method
        self.options = options
        self.rtol = rtol
        self.atol = atol
        self.adjoint_method = adjoint_method
        self.adjoint_rtol = rtol
        self.adjoint_atol = atol
        self.adjoint_options = adjoint_options


_arguments = None


@tf.custom_gradient
def OdeintAdjointMethod(*args):
    global _arguments  # type: _Arguments
    # args = _arguments.args
    # kwargs = _arguments.kwargs
    func = _arguments.func
    method = _arguments.method
    options = _arguments.options
    rtol = _arguments.rtol
    atol = _arguments.atol

    y0, t = args[:-1], args[-1]

    # registers `t` as a Variable that needs a gred, then resets it to a Tensor
    # for the `odeint` function to work. This is done to force tf to allow us to
    # pass the gradient of t as output.
    # t = tf.get_variable('t', initializer=t)
    # t = tf.convert_to_tensor(t, dtype=t.dtype)

    ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)

    @func_cast_double
    def grad(*grad_output, variables=None):
        global _arguments
        flat_params = _flatten(variables)

        func = _arguments.func
        adjoint_method = _arguments.adjoint_method
        adjoint_rtol = _arguments.rtol
        adjoint_atol = _arguments.atol
        adjoint_options = _arguments.adjoint_options

        n_tensors = len(ans)
        f_params = tuple(variables)

        # TODO: use a tf.keras.Model and call odeint_adjoint to implement higher order derivatives.
        def augmented_dynamics(t, y_aug):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.

            y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]  # Ignore adj_time and adj_params.

            with tf.GradientTape() as tape:
                tape.watch(t)
                tape.watch(y)
                func_eval = func(t, y)
                func_eval = cast_double(func_eval)

            # gradys = tf.stack(list(-adj_y_ for adj_y_ in adj_y))
            gradys = list(-adj_y_ for adj_y_ in adj_y)

            if type(func_eval) in [list, tuple]:
                for eval_ix in range(len(func_eval)):
                    if len(gradys[eval_ix].shape) < len(func_eval[eval_ix].shape):
                        gradys[eval_ix] = tf.expand_dims(gradys[eval_ix], axis=0)

            else:
                for grad_ix in range(len(gradys)):
                    if len(gradys[grad_ix].shape) < len(func_eval.shape):
                        gradys[grad_ix] = tf.expand_dims(gradys[grad_ix], axis=0)

            vjp_t, *vjp_y_and_params = tape.gradient(
                func_eval,
                (t,) + y + f_params,
                output_gradients=gradys
            )

            vjp_y = vjp_y_and_params[:n_tensors]
            vjp_params = vjp_y_and_params[n_tensors:]

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = tf.zeros_like(t, dtype=t.dtype) if vjp_t is None else vjp_t
            vjp_y = tuple(tf.zeros_like(y_, dtype=y_.dtype)
                          if vjp_y_ is None else vjp_y_
                          for vjp_y_, y_ in zip(vjp_y, y))
            vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)

            if _check_len(f_params) == 0:
                vjp_params = tf.convert_to_tensor(0., dtype=vjp_y[0].dype)
                vjp_params = move_to_device(vjp_params, vjp_y[0].device)

            return (*func_eval, *vjp_y, vjp_t, vjp_params)

        T = ans[0].shape[0]
        if isinstance(grad_output, tf.Tensor) or isinstance(grad_output, tf.Variable):
            adj_y = [grad_output[-1]]
        else:
            adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
        # adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
        adj_params = tf.zeros_like(flat_params, dtype=flat_params.dtype)
        adj_time = move_to_device(tf.convert_to_tensor(0., dtype=t.dtype), t.device)
        time_vjps = []
        for i in range(T - 1, 0, -1):

            ans_i = tuple(ans_[i] for ans_ in ans)

            if isinstance(grad_output, tf.Tensor) or isinstance(grad_output, tf.Variable):
                grad_output_i = [grad_output[i]]
            else:
                grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)

            func_i = func(t[i], ans_i)
            func_i = cast_double(func_i)

            if not isinstance(func_i, Iterable):
                func_i = [func_i]

            # Compute the effect of moving the current time measurement point.
            dLd_cur_t = sum(
                tf.reshape(tf.matmul(tf.reshape(func_i_, [1, -1]), tf.reshape(grad_output_i_, [-1, 1])), [1])
                for func_i_, grad_output_i_ in zip(func_i, grad_output_i)
            )
            adj_time = cast_double(adj_time)
            adj_time = adj_time - dLd_cur_t
            time_vjps.append(dLd_cur_t)

            # Run the augmented system backwards in time.
            if isinstance(adj_params, Iterable):
                count = _numel(adj_params)

                if count == 0:
                    adj_params = move_to_device(tf.convert_to_tensor(0., dtype=adj_y[0].dtype), adj_y[0].device)

            aug_y0 = (*ans_i, *adj_y, adj_time, adj_params)

            aug_ans = odeint(
                augmented_dynamics,
                aug_y0,
                tf.convert_to_tensor([t[i], t[i - 1]]),
                rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
            )

            # Unpack aug_ans.
            adj_y = aug_ans[n_tensors:2 * n_tensors]
            adj_time = aug_ans[2 * n_tensors]
            adj_params = aug_ans[2 * n_tensors + 1]

            adj_y = tuple(adj_y_[1] if _check_len(adj_y_) > 0 else adj_y_ for adj_y_ in adj_y)
            if _check_len(adj_time) > 0: adj_time = adj_time[1]
            if _check_len(adj_params) > 0: adj_params = adj_params[1]

            adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_, grad_output_ in zip(adj_y, grad_output))

            del aug_y0, aug_ans

        time_vjps.append(adj_time)
        time_vjps = tf.concat(time_vjps[::-1], 0)

        # reshape the parameters back into the correct variable shapes
        var_flat_lens = [_numel(v, dtype=tf.int32).numpy() for v in variables]
        var_shapes = [v.shape for v in variables]

        adj_params_splits = tf.split(adj_params, var_flat_lens)
        adj_params_list = [tf.reshape(p, v_shape)
                           for p, v_shape in zip(adj_params_splits, var_shapes)]

        return (*adj_y, time_vjps), adj_params_list

    return ans, grad


def odeint_adjoint(func, y0, t, rtol=1e-6, atol=1e-12, method=None, options=None, adjoint_method=None, adjoint_rtol=None,
                   adjoint_atol=None, adjoint_options=None):
    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, tf.keras.Model):
        raise ValueError('func is required to be an instance of tf.keras.Model')

    if adjoint_method is None:
        adjoint_method = method

    if adjoint_rtol is None:
        adjoint_rtol = rtol

    if adjoint_atol is None:
        adjoint_atol = atol

    if adjoint_options is None:
        adjoint_options = options

    with eager_mode():
        tensor_input = False
        if tf.debugging.is_numeric_tensor(y0):
            class TupleFunc(tf.keras.Model):

                def __init__(self, base_func, **kwargs):
                    super(TupleFunc, self).__init__(**kwargs)
                    self.base_func = base_func

                def call(self, t, y):
                    return (self.base_func(t, y[0]),)

            tensor_input = True
            y0 = (y0,)
            func = TupleFunc(func)

        # build the function to get its variables
        # if not func.built:
        #     _ = func(t, y0)

        global _arguments
        _arguments = _Arguments(func, method, options, rtol, atol,
                                adjoint_method, adjoint_rtol, adjoint_atol, adjoint_options)

        ys = OdeintAdjointMethod(*y0, t)

        if tensor_input or type(ys) == tuple or type(ys) == list:
            ys = ys[0]

        return ys
