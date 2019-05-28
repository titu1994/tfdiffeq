# Tensorflow Ordinary Differential Equation Solvers
[![Build Status](https://travis-ci.org/titu1994/tfdiffeq.svg?branch=master)](https://travis-ci.org/titu1994/tfdiffeq)

A library built to replicate the [TorchDiffEq](https://github.com/rtqichen/torchdiffeq) library built for the [Neural Ordinary Differential Equations paper by Chen et al](https://arxiv.org/abs/1806.07366), running entirely on Tensorflow Eager Execution.

All credits for the codebase go to [@rtqichen](https://github.com/rtqichen) for providing an excellent base to reimplement from.

Similar to the PyTorch codebase, this library provides ordinary differential equation (ODE) solvers implemented in Tensorflow Eager. For usage of ODE solvers in deep learning applications, see [Neural Ordinary Differential Equations paper](https://arxiv.org/abs/1806.07366).

Supports Augmented Neural ODE Architectures from the paper [Augmented Neural ODEs](https://arxiv.org/abs/1904.01681) as well, which has been shown to solve certain problems that Neural ODEs may struggle with.


As the solvers are implemented in Tensorflow, algorithms in this repository fully support running on the GPU, and are differentiable. Also supports prebuilt ODENet and ConvODENet tf.keras Models that can be used as is or embedded in a larger architecture. 

## Caveats

There are a few major limitations with this project : 

- Adjoint methods are not available. As Tensorflow Eager doesn't yet support custom gradient backpropogation on the level required by the Adjoint method, though the codebase is almost ported already, the `tf.custom_gradient` callback is not flexible enough to use for this purpose yet.
  - The code for adjoint methods has already been ported inside `adjoint.py`. However, it cannot be accessed as Tensorflow cannot handle custom gradients with variables created inside it (and greater number of gradients than original number of inputs).

- Speed is almost the same as the PyTorch codebase (+- 2%), *if the solver is wrapped inside a `tf.device` block*. Runge-Kutta solvers require double dtype precision for correct gradient computations. Yet, Tensorflow does not provide a convenient global switch to force all created tensors to double dtype. So explicit casts are unavoidable.
  - Make sure to wrap the entire script in a `with tf.device('/gpu:0')` to make full utilization of the GPU. Or select the main components - the model, the optimizer, the dataset and the `odeint` call inside tf.device blocks locally.
  - Convenience methods `move_to_device`, `cast_double` and the wrapper `func_cast_double` are made available from the library to make things easier on this front.
  - If type errors are thrown, use `tfdiffeq.cast_double(...)` to correct them.

# Basic Usage

Note: This is taken directly from the original PyTorch codebase. Almost all concepts apply here as well.

> **NOTE**: There exists a Jupyter Notebook in the examples folder, `ode_usage.ipynb` which has examples of several
ODE solutions, explaining various methods and demonstrates visualization functions available in this library.

> **NOTE**: The Notebook can also be visualized on Google Colab : [Colaboratory Link](https://colab.research.google.com/github/titu1994/tfdiffeq/blob/master/examples/ode_usage.ipynb)

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

# Prebuilt Models
This library now supports prebuilt models inside the `tfdiffeq.models` namespace - specifically the Neural ODENet and Convolutional Neural ODENet. In addition, both of these models inherently support **Augmented Neural ODENets**. 

They can be used a models themselves, or can be inserted inside a larger stack of ODENet layers to build a deeper ODENet or ConvODENet model, depending on the usecase.

Usage : 
```python
import tensorflow as tf
from tfdiffeq.models import ODENet, ConvODENet

# Directly usable model
model = ODENet(hidden_dim, output_dim, augment_dim=0, time_dependent=False)
model = ConvODENet(num_filters, augment_dim=0, time_dependent=False)

# Used inside other models
x = Conv2D(...)(x)
x = Conv2D(...)(x)
x = Flatten()(x)
x = ODENet(...)(x)  # or dont use flatten and use ConvODENet directly
x = ODENet(...)(x)  # or dont use flatten and use ConvODENet directly
...
```

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
 - `huen`: Second-order Runge-Kutta.
 - `rk4`: Fourth-order Runge-Kutta with 3/8 rule.
 - `explicit_adams`: Explicit Adams.
 - `fixed_adams`: Implicit Adams

## Compatibility

Since tensorflow doesn't yet support global setting of default datatype, the `tfdiffeq` library provides a few convenience methods.

- `move_to_device` : Attempts to move a `tf.Tensor` to a certain device. Can specify the device in the normal syntax of `cpu:0` or `gpu:x` where `x` must be replaced by any number representing the GPU ID. Falls back to CPU if GPU is unavailable.

- `cast_double` : Casts either a single `tf.Tensor` or a list of tensors to the `tf.float64` datatype.

- `func_cast_double` : A wrapper that casts all input arguments of the wrapped function to `tf.float64` dtype. Only affects arguments that are of type `tf.Tensor` or are a list of `tf.Tensor`.

# Examples

The scripts for the examples can be found in the `examples` folder, along with the weights and results for the `latent_ode.py` script as it takes some time to train. Two results have been replicated from the original codebase:

 - `ode_demo.py` : A basic example which contains a short implementation of learning a dynamics model to mimic a spiral ODE.
 
 The training should look similar to this:
 
![ode spiral demo](https://github.com/titu1994/tfdiffeq/blob/master/images/demo1.gif?raw=true)

- `circular_ode_demo.py` : A basic example similar to above which contains a short implementation of learning a dynamics model to mimic a circular ODE.

 The training should look similar to this:

 ![ode circular demo](https://github.com/titu1994/tfdiffeq/blob/master/images/circular_ode.gif?raw=true)

- `lorenz_attractor.py` : A classic example of a chaotic solution for certain parameter sets and initial conditions.

Note this is just a stress test for the library, and scipy.integrate.odefun can solve this much much faster due to highly
optimized routines. This should take roughly 1 minute on a modern machine.

 ![lorenz attractor](https://github.com/titu1994/tfdiffeq/blob/master/images/lorenz.png?raw=true)

- `latent_ode.py` : Another basic example which uses variational inference to learn a path along a spiral.
 
 Results should be similar to below after 1200 iterations:
 
 ![ode spiral latent](https://github.com/titu1994/tfdiffeq/blob/master/images/vis.png?raw=true)

- `ODENet` on MNIST

 While the Adjoint method is not yet implemented, a smaller version of ODENet can be easily trained using the
 fixed grid solvers - `Euler` or `Huens` for a fast approximate solution. It has been observed that as MNIST is
 an extremely easy problem, RK45 (DOPRI5) works relatively well, whereas on more complex datasets like CIFAR 10/100
 it diverges in the first epoch.

 Reference :
 - [ANODE: Unconditionally Accurate Memory-Efficient Gradients for Neural ODEs](https://arxiv.org/abs/1902.10298)
 
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
 
 - Tensorflow 1.12.0 or above / TF 2. Prefereably TF 2.0 when it comes out, as the entire codebase *requires* Eager Execution.
 - matplotlib
 - numpy
 - scipy (for tests)
 - six
 
 
