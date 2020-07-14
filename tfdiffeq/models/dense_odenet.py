"""
Definition of ODENet and Augmented ODENet.
Ported from https://github.com/EmilienDupont/augmented-neural-odes/blob/master/anode/models.py
"""
import tensorflow as tf
from tfdiffeq import odeint

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver


class ODEFunc(tf.keras.Model):

    def __init__(self, hidden_dim, augment_dim=0,
                 time_dependent=False, non_linearity='relu',
                 **kwargs):
        """
        MLP modeling the derivative of ODE system.

        # Arguments:
            input_dim : int
                Dimension of data.
            hidden_dim : int
                Dimension of hidden layers.
            augment_dim: int
                Dimension of augmentation. If 0 does not augment ODE, otherwise augments
                it with augment_dim dimensions.
            time_dependent : bool
                If True adds time as input, making ODE time dependent.
            non_linearity : string
                One of 'relu' and 'softplus'
        """
        dynamic = kwargs.pop('dynamic', True)
        super(ODEFunc, self).__init__(**kwargs, dynamic=dynamic)
        self.augment_dim = augment_dim
        # self.data_dim = input_dim
        # self.input_dim = input_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent

        self.fc1 = tf.keras.layers.Dense(hidden_dim)
        self.fc2 = tf.keras.layers.Dense(hidden_dim)

        self.fc3 = None

        if non_linearity == 'relu':
            self.non_linearity = tf.keras.layers.ReLU()
        elif non_linearity == 'softplus':
            self.non_linearity = tf.keras.layers.Activation('softplus')
        else:
            self.non_linearity = tf.keras.layers.Activation(non_linearity)

    def build(self, input_shape):
        if len(input_shape) > 0:
            self.fc3 = tf.keras.layers.Dense(input_shape[-1])
            self.built = True

    @tf.function
    def call(self, t, x, training=None, **kwargs):
        """
        Forward pass. If time dependent, concatenates the time
        dimension onto the input before the call to the dense layer.

        # Arguments:
            t: Tensor. Current time. Shape (1,).
            x: Tensor. Shape (batch_size, input_dim).

        # Returns:
            Output tensor of forward pass.
        """

        # build the final layer if it wasnt built yet
        if self.fc3 is None:
            self.fc3 = tf.keras.layers.Dense(x.shape.as_list()[-1])

        # Forward pass of model corresponds to one function evaluation, so
        # increment counter
        self.nfe += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = tf.ones([x.shape[0], 1], dtype=t.dtype) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = tf.concat([t_vec, x], axis=-1)
            # Shape (batch_size, hidden_dim)
            # TODO: Remove cast when Keras supports double
            out = self.fc1(tf.cast(t_and_x, tf.float32))
        else:
            out = self.fc1(x)
        out = self.non_linearity(out)
        out = self.fc2(out)
        out = self.non_linearity(out)
        out = self.fc3(out)
        return out


class ODEBlock(tf.keras.Model):

    def __init__(self, odefunc, is_conv=False, tol=1e-3, adjoint=False, solver='dopri5', **kwargs):
        """
        Solves ODE defined by odefunc.

        # Arguments:
            odefunc : ODEFunc instance or Conv2dODEFunc instance
                Function defining dynamics of system.
            is_conv : bool
                If True, treats odefunc as a convolutional model.
            tol : float
                Error tolerance.
            adjoint : bool
                If True calculates gradient with adjoint solver, otherwise
                backpropagates directly through operations of ODE solver.
            solver: ODE solver. Defaults to DOPRI5.
        """
        dynamic = kwargs.pop('dynamic', True)
        super(ODEBlock, self).__init__(**kwargs, dynamic=dynamic)

        if adjoint:
            raise NotImplementedError("adjoint solver has not been implemented yet !")

        self.adjoint = adjoint
        self.is_conv = is_conv
        self.odefunc = odefunc
        self.tol = tol
        self.method = solver
        self.channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        if solver == "dopri5":
            self.options = {'max_num_steps': MAX_NUM_STEPS}
        else:
            self.options = None

    def call(self, x, training=None, eval_times=None, **kwargs):
        """
        Solves ODE starting from x.

        # Arguments:
            x: Tensor. Shape (batch_size, self.odefunc.data_dim)
            eval_times: None or tf.Tensor.
                If None, returns solution of ODE at final time t=1. If tf.Tensor
                then returns full ODE trajectory evaluated at points in eval_times.

        # Returns:
            Output tensor of forward pass.
        """

        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = tf.convert_to_tensor([0, 1], dtype=x.dtype)
        else:
            integration_time = tf.cast(eval_times, x.dtype)

        if self.odefunc.augment_dim > 0:
            if self.is_conv:
                # Add augmentation
                if self.channel_axis == 1:
                    batch_size, channels, height, width = x.shape
                    aug = tf.zeros([batch_size, self.odefunc.augment_dim,
                                    height, width], dtype=x.dtype)

                else:
                    batch_size, height, width, channels = x.shape

                    aug = tf.zeros([batch_size, height, width,
                                    self.odefunc.augment_dim], dtype=x.dtype)

                # Shape (batch_size, channels + augment_dim, height, width)
                x_aug = tf.concat([x, aug], axis=self.channel_axis)
            else:
                # Add augmentation
                aug = tf.zeros([x.shape[0], self.odefunc.augment_dim], dtype=x.dtype)
                # Shape (batch_size, data_dim + augment_dim)
                x_aug = tf.concat([x, aug], axis=-1)
        else:
            x_aug = x

        if self.adjoint:
            # TODO: Replace with odeint_adjoint once implemented !
            out = odeint(self.odefunc, x_aug, integration_time,
                         rtol=self.tol, atol=self.tol, method=self.method,
                         options=self.options)
        else:
            out = odeint(self.odefunc, x_aug, integration_time,
                         rtol=self.tol, atol=self.tol, method=self.method,
                         options=self.options)

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out

    def trajectory(self, x, timesteps):
        """Returns ODE trajectory.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)
        timesteps : int
            Number of timesteps in trajectory.
        """
        integration_time = tf.linspace(0., 1., timesteps)
        return self.call(x, eval_times=integration_time)


class ODENet(tf.keras.Model):

    def __init__(self, hidden_dim, output_dim,
                 augment_dim=0, time_dependent=False, non_linearity='relu',
                 tol=1e-3, adjoint=False, solver='dopri5', **kwargs):
        """
        An ODEBlock followed by a Linear layer.

        # Arguments:
            hidden_dim : int
                Dimension of hidden layers.
            output_dim : int
                Dimension of output after hidden layer. Should be 1 for regression or
                num_classes for classification.
            augment_dim: int
                Dimension of augmentation. If 0 does not augment ODE, otherwise augments
                it with augment_dim dimensions.
            time_dependent : bool
                If True adds time as input, making ODE time dependent.
            non_linearity : string
                One of 'relu' and 'softplus'
            tol : float
                Error tolerance.
            adjoint : bool
                If True calculates gradient with adjoint method, otherwise
                backpropagates directly through operations of ODE solver.
            solver: ODE solver. Defaults to DOPRI5.
        """
        dynamic = kwargs.pop('dynamic', True)
        super(ODENet, self).__init__(**kwargs, dynamic=dynamic)

        if adjoint:
            raise NotImplementedError("adjoint solver has not been implemented yet !")

        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.time_dependent = time_dependent
        self.tol = tol

        odefunc = ODEFunc(hidden_dim, augment_dim,
                          time_dependent, non_linearity)

        self.odeblock = ODEBlock(odefunc, tol=tol, adjoint=adjoint, solver=solver)
        self.linear_layer = tf.keras.layers.Dense(self.output_dim)

    # @tf.function
    def call(self, x, training=None, return_features=False):
        features = self.odeblock(x, training=training)

        # TODO: Remove cast when keras supports double
        pred = self.linear_layer(tf.cast(features, tf.float32))
        if return_features:
            return features, pred
        return pred
