import six
from typing import Iterable

import tensorflow as tf

from tfdiffeq.odeint import odeint, SOLVERS
from tfdiffeq.misc import (_check_inputs, _flat_to_shape, _mixed_linf_rms_norm, move_to_device, _check_len, _numel)


class _Arguments(object):

    def __init__(self, func, shapes, method, options, rtol, atol, adjoint_method, adjoint_rtol, adjoint_atol,
                 adjoint_options):
        self.func = func
        self.shapes = shapes
        self.method = method
        self.options = options
        self.rtol = rtol
        self.atol = atol
        self.adjoint_method = adjoint_method
        self.adjoint_rtol = adjoint_rtol
        self.adjoint_atol = adjoint_atol
        self.adjoint_options = adjoint_options


def grad_wrapper(func):
    """Necessary to fix tensorflow "variables in gradspec.args" error"""

    @six.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    return wrapper


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

    # registers `t` as a Variable that needs a grad, then resets it to a Tensor
    # for the `odeint` function to work. This is done to force tf to allow us to
    # pass the gradient of t as output.
    # t = tf.get_variable('t', initializer=t)
    # t = tf.convert_to_tensor(t, dtype=t.dtype)

    ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)

    @grad_wrapper
    def grad(*grad_output, variables=None):
        global _arguments

        func = _arguments.func
        shapes = _arguments.shapes
        adjoint_method = _arguments.adjoint_method
        adjoint_rtol = _arguments.rtol
        adjoint_atol = _arguments.atol
        adjoint_options = _arguments.adjoint_options

        n_tensors = len(ans)
        f_params = tuple(variables)

        ##################################
        #     Set up adjoint_options     #
        ##################################

        if adjoint_options is None:
            adjoint_options = {}
        else:
            adjoint_options = adjoint_options.copy()

        # We assume that any grid points are given to us ordered in the same direction as for the forward pass (for
        # compatibility with setting adjoint_options = options), so we need to flip them around here.
        try:
            grid_points = adjoint_options['grid_points']
        except KeyError:
            pass
        else:
            adjoint_options['grid_points'] = grid_points.flip(0)

        # Backward compatibility: by default use a mixed L-infinity/RMS norm over the input, where we treat t, each
        # element of y, and each element of adj_y separately over the Linf, but consider all the parameters
        # together.
        if 'norm' not in adjoint_options:
            if shapes is None:
                shapes = [ans[-1].shape]  # [-1] because y has shape (len(t), *y0.shape)
            # adj_t, y, adj_y, adj_params, corresponding to the order in aug_state below
            adjoint_shapes = [tf.TensorShape(())] + shapes + shapes + [tf.TensorShape([sum(_numel(param)
                                                                                           for param in f_params)])]
            adjoint_options['norm'] = _mixed_linf_rms_norm(adjoint_shapes)

        # ~Backward compatibility

        ##################################
        #    Set up backward ODE func    #
        ##################################

        # TODO: use a tf.keras.Model and call odeint_adjoint to implement higher order derivatives.
        def augmented_dynamics(t, y_aug):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.

            # y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]  # Ignore adj_time and adj_params.
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y = y_aug[0]
            adj_y = y_aug[1]
            # ignore gradients wrt time and parameters

            with tf.GradientTape() as tape:
                tape.watch(t)
                tape.watch(y)
                func_eval = func(t, y)
                # func_eval = tf.convert_to_tensor(func_eval)

            # gradys = -tf.stack(adj_y)
            # if type(func_eval) in [list, tuple]:
            #     for eval_ix in range(len(func_eval)):
            #         if len(gradys[eval_ix].shape) < len(func_eval[eval_ix].shape):
            #             gradys[eval_ix] = tf.expand_dims(gradys[eval_ix], axis=0)
            #
            # else:
            #     if len(gradys.shape) < len(func_eval.shape):
            #         gradys = tf.expand_dims(gradys, axis=0)
            gradys = -adj_y

            vjp_t, vjp_y, *vjp_params = tape.gradient(
                func_eval,
                (t, y) + f_params,
                output_gradients=gradys,
                unconnected_gradients=tf.UnconnectedGradients.ZERO
            )

            # vjp_y = vjp_y_and_params[:n_tensors]
            # vjp_params = vjp_y_and_params[n_tensors:]
            # vjp_params = _flatten(vjp_params)

            if _check_len(f_params) == 0:
                vjp_params = tf.convert_to_tensor(0., dtype=vjp_y[0].dype)
                vjp_params = move_to_device(vjp_params, vjp_y[0].device)

            return (vjp_t, func_eval, vjp_y, *vjp_params)

        ##################################
        #      Set up initial state      #
        ##################################

        T = ans[0].shape[0]

        # if isinstance(grad_output, (tf.Tensor, tf.Variable)):
        #     adj_y = [grad_output[-1]]
        # else:
        #     adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
        adj_y = grad_output[-1]

        adj_params = [tf.zeros_like(param, dtype=param.dtype) for param in f_params]
        adj_time = move_to_device(tf.convert_to_tensor(0., dtype=t.dtype), t.device)
        time_vjps = []
        for i in range(T - 1, 0, -1):

            ans_i = ans[i]

            # if isinstance(grad_output, (tf.Tensor, tf.Variable)):
            #     grad_output_i = [grad_output[i]]
            # else:
            #     grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)

            func_i = func(t[i], ans_i)

            # if not isinstance(func_i, Iterable):
            #     func_i = [func_i]

            # Compute the effect of moving the current time measurement point.
            # dLd_cur_t = sum(
            #     tf.reshape(tf.matmul(tf.reshape(func_i_, [1, -1]), tf.reshape(grad_output_i_, [-1, 1])), [1])
            #     for func_i_, grad_output_i_ in zip(func_i, grad_output_i)
            # )
            dLd_cur_t = tf.matmul(tf.reshape(func_i, [-1]), tf.reshape(grad_output[i], [-1]))

            adj_time = adj_time - dLd_cur_t
            time_vjps.append(dLd_cur_t)

            # Run the augmented system backwards in time.
            if isinstance(adj_params, Iterable):
                if _numel(adj_params) == 0:
                    adj_params = move_to_device(tf.convert_to_tensor(0., dtype=adj_y.dtype), adj_y.device)

            aug_y0 = (adj_time, adj_y, ans_i, *adj_params)

            aug_ans = odeint(
                augmented_dynamics,
                aug_y0,
                tf.convert_to_tensor([t[i], t[i - 1]]),
                rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
            )

            # Unpack aug_ans.
            # adj_y = aug_ans[n_tensors:2 * n_tensors]
            # adj_time = aug_ans[2 * n_tensors]
            # adj_params = aug_ans[2 * n_tensors + 1]

            aug_ans = [a[1] for a in aug_ans]  # extract just the t[i - 1] value
            aug_ans[1] = ans[i - 1]  # update to use our forward-pass estimate of the state
            aug_ans[2] += grad_output[i - 1]  # update any gradients wrt state at this time point

            # adj_y = tuple(adj_y_[1] if _check_len(adj_y_) > 0 else adj_y_ for adj_y_ in adj_y)
            # if _check_len(adj_time) > 0: adj_time = adj_time[1]
            # if _check_len(adj_params) > 0: adj_params = adj_params[1]

            # adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_, grad_output_ in zip(adj_y, grad_output))
            # del aug_y0, aug_ans

            adj_y = aug_ans[2]
            adj_params = aug_ans[3:]

        time_vjps.append(adj_time)
        time_vjps = tf.concat(time_vjps[::-1], 0)

        # # reshape the parameters back into the correct variable shapes
        # var_flat_lens = [_numel(v, dtype=tf.int32).numpy() for v in variables]
        # var_shapes = [v.shape for v in variables]
        #
        # adj_params_splits = tf.split(adj_params, var_flat_lens)
        # adj_params_list = [tf.reshape(p, v_shape)
        #                    for p, v_shape in zip(adj_params_splits, var_shapes)]
        return (adj_y, time_vjps), adj_params

    return ans, grad


def odeint_adjoint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None, adjoint_method=None, adjoint_rtol=None,
                   adjoint_atol=None, adjoint_options=None):
    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, tf.keras.Model):
        raise ValueError('func is required to be an instance of tf.keras.Model')

    # Must come before we default adjoint_options to options; using the same norm for both wouldn't make any sense.
    try:
        options['norm']
    except (TypeError, KeyError):
        pass
    else:
        try:
            adjoint_options['norm']
        except (TypeError, KeyError):
            raise ValueError("If specifying a custom `norm` for the forward pass, then must also specify a `norm` "
                             "for the adjoint (backward) pass.")

    # Must come before _check_inputs as we don't want to use normalised input (in particular any changes to options)
    if adjoint_method is None:
        adjoint_method = method

    if adjoint_rtol is None:
        adjoint_rtol = rtol

    if adjoint_atol is None:
        adjoint_atol = atol

    if adjoint_options is None:
        adjoint_options = options

    # tensor_input = False
    # if tf.debugging.is_numeric_tensor(y0):
    #     class TupleFunc(tf.keras.Model):
    #         def __init__(self, base_func, **kwargs):
    #             super(TupleFunc, self).__init__(dtype=base_func.dtype, **kwargs)
    #             self.base_func = base_func
    #
    #         def call(self, t, y):
    #             return (self.base_func(t, y[0]),)
    #
    #     tensor_input = True
    #     y0 = (y0,)
    #     func = TupleFunc(func)

    # Normalise to non-tupled input
    shapes, func, y0, t, rtol, atol, method, options = _check_inputs(func, y0, t, rtol, atol, method, options, SOLVERS)

    global _arguments
    _arguments = _Arguments(func, shapes, method, options, rtol, atol, adjoint_method, adjoint_rtol, adjoint_atol,
                            adjoint_options)

    ys = OdeintAdjointMethod(*y0, t)

    if shapes is not None:
        ys = _flat_to_shape(ys, (len(t),), shapes)
    return ys
