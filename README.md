# Tensorflow Ordinary Differential Equation Solvers

A library built to replicate the [TorchDiffEq](https://github.com/rtqichen/torchdiffeq) library built for the [Neural Ordinary Differential Equations paper by Chen et al](https://arxiv.org/abs/1806.07366).

All credits for the codebase go to [@rtqichen](https://github.com/rtqichen) for providing an excellent base to reimplement from.

Similar to the PyTorch codebase, this library provides ordinary differential equation (ODE) solvers implemented in Tensorflow Eager. For usage of ODE solvers in deep learning applications, see [Neural Ordinary Differential Equations paper](https://arxiv.org/abs/1806.07366).

As the solvers are implemented in Tensorflow, algorithms in this repository are fully supported to run on the GPU.

## Caveats

There are a few major limitations with this project : 

- Adjoint methods are not available. As Tensorflow Eager doesn't yet support custom gradient backpropogation on the level required by the Adjoint method, though the codebase is almost ported already, the `tf.custom_gradient` callback is not flexible enough to use for this purpose yet.
  - The code for adjoint methods has already been ported inside `adjoint.py`. However, it cannot be accessed as Tensorflow cannot handle custom gradients with variables created inside it (and greater number of gradients than original number of inputs).

- Speed is slightly slower than the PyTorch codebase. This is because there are several places where I had to place explicit casts to `tf.float64`. Runge-Kutta solvers require that level of precision for correct gradient computations. Yet, Tensorflow does not provide a convenient global switch to force all created tensors to double dtype. So explicit casts were unavoidable. 
  - Make sure to wrap the entire script in a `with tf.device('/gpu:0')` to make full utilization of the GPU.
  - Convenience methods `move_to_device`, `cast_double` and the wrapper `func_cast_double` are made available from the library to make things easier on this front.
  
- No equivalent tests for gradients checks. Tensorflow has no equivalent to `torch.autograd.checkgrad`, and therefore I could not port the tests. However, given that the non-adjoint codebase is a ditto-replica, and that the example scripts can replicate the original codebase results perfectly, I think the gradient computations are working correctly.
  - Tests will be updated when Tensorflow provides equivalent functionality. One could port the entire `torch.autograd.checkgrad` functionality from PyTorch, but it's too much work just to run tests.
  
# Basic Usage

Note: This is taken directly from the original PyTorch codebase. Almost all concepts apply here as well.

This library provides one main interface odeint which contains general-purpose algorithms for solving initial value problems (IVP), with gradients implemented for all main arguments. An initial value problem consists of an ODE and an initial value,

```
dy/dt = f(t, y)    y(t_0) = y_0.
```

The goal of an ODE solver is to find a continuous trajectory satisfying the ODE that passes through the initial condition.

To solve an IVP using the default solver:

```
from tfdiffeq import odeint

odeint(func, y0, t)
```

where `func` is any callable implementing the ordinary differential equation `f(t, x)`, `y0` is an any-D Tensor or a tuple of any-D Tensors representing the initial values, and `t` is a 1-D Tensor containing the evaluation points. The initial time is taken to be `t[0]`.

Backpropagation through odeint goes through the internals of the solver, but this is not supported for all solvers. Instead, we encourage the use of the adjoint method explained in [Neural Ordinary Differential Equations paper](https://arxiv.org/abs/1806.07366), which will allow solving with as many steps as necessary due to O(1) memory usage.

**NOTE: As of now, adjoint methods are not available in this port.**

# Keyword Arguments

- `rtol`: Relative tolerance.
- `atol`: Absolute tolerance.
- `method`: One of the solvers listed below.

### List of ODE Solvers:

### Adaptive-step:

 - `dopri5`: Runge-Kutta 4(5) [default].
 - `adams`: Adaptive-order implicit Adams.

### Fixed-step:

 - `euler`: Euler method.
 - `midpoint`: Midpoint method.
 - `rk4`: Fourth-order Runge-Kutta with 3/8 rule.
 - `explicit_adams`: Explicit Adams.
 - `fixed_adams`: Implicit Adams

## Compatibility

Since tensorflow doesn't yet support global setting of default datatype, the `tfdiffeq` library provides a few convenience methods.

- `move_do_device` : Attempts to move a `tf.Tensor` to a certain device. Can specify the device in the normal syntax of `cpu:0` or `gpu:x` where `x` must be replaced by any number representing the GPU ID. Falls back to CPU if GPU is unavailable.

- `cast_double` : Casts either a single `tf.Tensor` or a list of tensors to the `tf.float64` datatype.

- `func_cast_double` : A wrapper that casts all input arguments of the wrapped function to `tf.float64` dtype. Only affects arguments that are of type `tf.Tensor` or are a list of `tf.Tensor`.

# Examples

The scripts for the examples can be found in the `examples` folder, along with the weights and results for the `latent_ode.py` script as it takes some time to train. Two results have been replicated from the original codebase:

 - `ode_demo.py` : A basic example which contains a short implementation of learning a dynamics model to mimic a spiral ODE. Defaults automatically to allow visualization.
 
 The training should look similar to this:
 
![ode spiral demo](https://github.com/titu1994/tfdiffeq/blob/master/examples/demo1.gif?raw=true)

 - `latent_ode.py` : Another basic example which uses variational inference to learn a path along a spiral. 
 
 Results should be similar to below after 1200 iterations:
 
 ![ode spiral latent](https://github.com/titu1994/tfdiffeq/blob/master/examples/vis.png?raw=true)
 
 - The ODENet on MNIST experiment has not been performed yet, as without Adjoint methods, it takes an enormous amount of memory and time.
 
# Reference
If you found this library useful in your research, please consider citing

```
@article{chen2018neural,
  title={Neural Ordinary Differential Equations},
  author={Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David},
  journal={Advances in Neural Information Processing Systems},
  year={2018}
}
```
 
# Requirements

Install the required Tensorflow version along with the package using either

```
pip install .[tf]  # for cpu
pip install .[tf-gpu]  # for gpu
pip install .[tests]  # for cpu testing
```
 
 - Tensorflow 1.12.0 or above. Prefereably TF 2.0 when it comes out, as the entire codebase *requires* Eager Execution.
 - matplotlib
 - numpy
 - scipy (for tests)
 - six
 
 
