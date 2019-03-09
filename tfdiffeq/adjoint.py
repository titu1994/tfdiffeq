import numpy as np
from typing import Iterable

import tensorflow as tf
from tensorflow.python.eager.context import eager_mode

from tfdiffeq import odeint
from tfdiffeq.misc import (_flatten, _flatten_convert_none_to_zeros,
                           move_to_device, cast_double, func_cast_double,
                           _check_len, _numel, _convert_to_tensor)


class _Arguments(object):

    def __init__(self, func, method, options, rtol, atol):
        self.func = func
        self.method = method
        self.options = options
        self.rtol = rtol
        self.atol = atol


_arguments = None


@tf.custom_gradient
def OdeintAdjointMethod(*args):
    global _arguments
    # args = _arguments.args
    # kwargs = _arguments.kwargs
    func = _arguments.func
    method = _arguments.method
    options = _arguments.options
    rtol = _arguments.rtol
    atol = _arguments.atol

    assert len(args) >= 3, 'Internal error: all arguments required.'
    y0, t, flat_params = args[:-2], args[-2], args[-1]

    # registers `t` as a Variable that needs a gred, then resets it to a Tensor
    # for the `odeint` function to work. This is done to force tf to allow us to
    # pass the gradient of t as output.
    # t = tf.get_variable('t', initializer=t)
    # t = tf.convert_to_tensor(t, dtype=t.dtype)

    ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)

    @func_cast_double
    def grad(*grad_output, variables=None):
        global _arguments
        # t, flat_params, *ans = ctx.saved_tensors
        # ans = tuple(ans)
        # func, rtol, atol, method, options = ctx.func, ctx.rtol, ctx.atol, ctx.method, ctx.options
        func = _arguments.func
        method = _arguments.method
        options = _arguments.options
        rtol = _arguments.rtol
        atol = _arguments.atol

        print("Gradient Output : ", grad_output)
        print("Variables : ", variables)

        n_tensors = len(ans)
        f_params = tuple(variables)

        # TODO: use a tf.keras.Model and call odeint_adjoint to implement higher order derivatives.
        def augmented_dynamics(t, y_aug):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.

            y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]  # Ignore adj_time and adj_params.

            # t = tf.get_variable('t', initializer=t)
            # y = tuple(tf.Variable(y_) for y_ in y)

            with tf.GradientTape() as tape:
                tape.watch(t)
                tape.watch(y)
                func_eval = func(t, y)
                func_eval = cast_double(func_eval)

            # print('y', [y_.numpy().shape for y_ in y])
            # print('adj y', [a.numpy().shape for a in adj_y])

            vjp_t, *vjp_y_and_params = tape.gradient(func_eval, (t,) + y + f_params,
                                                     # list(-adj_y_ for adj_y_ in adj_y),
                                                     )

            vjp_y = vjp_y_and_params[:n_tensors]
            vjp_params = vjp_y_and_params[n_tensors:]
            # print('vjp_y', [v.numpy().shape if v is not None else None for v in vjp_y])
            # print()

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = tf.zeros_like(t, dtype=t.dtype) if vjp_t is None else vjp_t
            vjp_y = tuple(tf.zeros_like(y_, dtype=y_.dtype)
                          if vjp_y_ is None else vjp_y_
                          for vjp_y_, y_ in zip(vjp_y, y))
            vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)

            if _check_len(f_params) == 0:
                vjp_params = tf.convert_to_tensor(0., dtype=vjp_y[0].dype)
                vjp_params = move_to_device(vjp_params, vjp_y[0].device)

            # print('vjp_t grad', vjp_t.numpy())
            # print('vjp_params', [v.numpy() for v in vjp_params])
            # print('vjp y grads', [v.numpy().shape for v in vjp_y])
            # print("LEN FUNC EVALS : ", len(func_eval))
            # print()

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

            # print('ans i', [a.numpy().shape for a in ans_i])
            # print('adj y', [a.numpy().shape for a in adj_y])
            # print('adj time', adj_time.numpy().shape)
            # print('adj params', adj_params.numpy().shape)

            # print()

            aug_ans = odeint(
                augmented_dynamics,
                aug_y0,
                tf.convert_to_tensor([t[i], t[i - 1]]),
                rtol=rtol, atol=atol, method=method, options=options
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

        print()
        print('adj y', len(adj_y))
        print('time vjps', time_vjps.shape)
        print('adj params', adj_params.shape)
        print()

        # reshape the parameters back into the correct variable shapes
        var_flat_lens = [_numel(v, dtype=tf.int32).numpy() for v in variables]
        var_shapes = [v.shape for v in variables]

        adj_params_splits = tf.split(adj_params, var_flat_lens)
        adj_params_list = [tf.reshape(p, v_shape)
                           for p, v_shape in zip(adj_params_splits, var_shapes)]

        # add the time gradient (always the first tensor in list of variables)
        # adj_params.insert(0, time_vjps)

        # adj_y_grad_vars = list(zip(adj_y, grad_output))
        # time_grad_vars = list((time_vjps, t))
        model_vars = list(adj_params_list)  # list(zip(adj_params, variables))

        grad_vars = model_vars  # adj_y_grad_vars + time_grad_vars + model_vars
        # print('adj y grad', len(adj_y_grad_vars))
        # print('time grad', len(time_grad_vars))
        print('model grad', len(model_vars))
        print('model grad values', [v for v in grad_vars])
        print()

        # if len(adj_y) == 1:
        #     adj_y = adj_y[0]

        return (adj_y, model_vars)
        # print("Grad_vars : ", grad_vars)
        # return zip(*grad_vars)  # (*adj_y, time_vjps, adj_params, None, None)
        # return (*adj_y, time_vjps, adj_params)

    return ans, grad


def odeint_adjoint(func, y0, t, rtol=1e-6, atol=1e-12, method=None, options=None):
    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, tf.keras.Model):
        raise ValueError('func is required to be an instance of tf.keras.Model')

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
        if not func.built:
            _ = func(t, y0)

        flat_params = _flatten(func.variables)

        global _arguments
        _arguments = _Arguments(func, method, options, rtol, atol)

        ys = OdeintAdjointMethod(*y0, t, flat_params)

        if tensor_input or type(ys) == tuple or type(ys) == list:
            ys = ys[0]

        return ys
