"""
Definition of Convolutional ODENet and Augmented ODENet.
Ported from : https://github.com/EmilienDupont/augmented-neural-odes/blob/master/anode/conv_models.py
"""
import tensorflow as tf

from tfdiffeq.models.dense_odenet import ODEBlock

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver


class Conv2dTime(tf.keras.Model):
    """
    Implements time dependent 2d convolutions, by appending the time variable as
    an extra channel.
    """
    def __init__(self, dim_out, kernel_size=3, stride=1, padding=0, dilation=1,
                 bias=True, transpose=False):
        super(Conv2dTime, self).__init__()
        module = tf.keras.layers.Conv2DTranspose if transpose else tf.keras.layers.Conv2D

        padding = 'same'if padding == 1 else 'valid'
        self._layer = module(
            dim_out, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding=padding,
            dilation_rate=dilation,
            use_bias=bias
        )

        self.channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

    @tf.function
    def call(self, t, x, training=None, **kwargs):
        # TODO: Remove cast when Keras supports double
        t = tf.cast(t, x.dtype)

        if self.channel_axis == 1:
            # Shape (batch_size, 1, height, width)
            tt = tf.ones_like(x[:, :1, :, :], dtype=t.dtype) * t  # channel dim = 1

        else:
            # Shape (batch_size, height, width, 1)
            tt = tf.ones_like(x[:, :, :, :1], dtype=t.dtype) * t  # channel dim = -1

        ttx = tf.concat([tt, x], axis=self.channel_axis)  # concat at channel dim

        # TODO: Remove cast when Keras supports double
        ttx = tf.cast(ttx, tf.float32)

        return self._layer(ttx)


class Conv2dODEFunc(tf.keras.Model):

    def __init__(self, num_filters, augment_dim=0,
                 time_dependent=False, non_linearity='relu', **kwargs):
        """
        Convolutional block modeling the derivative of ODE system.

        # Arguments:
            num_filters : int
                Number of convolutional filters.
            augment_dim: int
                Number of augmentation channels to add. If 0 does not augment ODE.
            time_dependent : bool
                If True adds time as input, making ODE time dependent.
            non_linearity : string
                One of 'relu' and 'softplus'
        """
        dynamic = kwargs.pop('dynamic', True)
        super(Conv2dODEFunc, self).__init__(**kwargs, dynamic=dynamic)

        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations
        # self.channels += augment_dim
        self.num_filters = num_filters

        if time_dependent:
            self.conv1 = Conv2dTime(self.num_filters,
                                    kernel_size=1, stride=1, padding=0)
            self.conv2 = Conv2dTime(self.num_filters,
                                    kernel_size=3, stride=1, padding=1)
            self.conv3 = None

        else:
            self.conv1 = tf.keras.layers.Conv2D(self.num_filters,
                                                kernel_size=(1, 1), strides=(1, 1),
                                                padding='valid')
            self.conv2 = tf.keras.layers.Conv2D(self.num_filters,
                                                kernel_size=(3, 3), strides=(1, 1),
                                                padding='same')
            self.conv3 = None

        if non_linearity == 'relu':
            self.non_linearity = tf.keras.layers.ReLU()
        elif non_linearity == 'softplus':
            self.non_linearity = tf.keras.layers.Activation('softplus')
        else:
            self.non_linearity = tf.keras.layers.Activation(non_linearity)

    def build(self, input_shape):
        if len(input_shape) > 0:
            if self.time_dependent:
                self.conv3 = Conv2dTime(self.channels,
                                        kernel_size=1, stride=1, padding=0)
            else:
                self.conv3 = tf.keras.layers.Conv2D(self.channels,
                                                    kernel_size=(1, 1), strides=(1, 1),
                                                    padding='valid')

            self.built = True

    @tf.function
    def call(self, t, x, training=None, **kwargs):
        """
        Parameters
        ----------
        t : Tensor
            Current time.
        x : Tensor
            Shape (batch_size, input_dim)
        """
        # build the final layer if it wasnt built yet
        if self.conv3 is None:
            channel_dim = 1 if tf.keras.backend.image_data_format() == 'channel_first' else -1
            self.channels = x.shape.as_list()[channel_dim]

            if self.time_dependent:
                self.conv3 = Conv2dTime(self.channels,
                                        kernel_size=1, stride=1, padding=0)
            else:
                self.conv3 = tf.keras.layers.Conv2D(self.channels,
                                                    kernel_size=(1, 1), strides=(1, 1),
                                                    padding='valid')

        self.nfe += 1

        if self.time_dependent:
            out = self.conv1(t, x)
            out = self.non_linearity(out)
            out = self.conv2(t, out)
            out = self.non_linearity(out)
            out = self.conv3(t, out)
        else:
            # TODO: Remove cast to tf.float32 once Keras supports tf.float64
            x = tf.cast(x, tf.float32)
            out = self.conv1(x)
            out = self.non_linearity(out)
            out = self.conv2(out)
            out = self.non_linearity(out)
            out = self.conv3(out)

        return out


class Conv2dODENet(tf.keras.Model):
    """Creates an ODEBlock with a convolutional ODEFunc followed by a Linear
    layer.
    Parameters
    ----------
    img_size : tuple of ints
        Tuple of (channels, height, width).
    num_filters : int
        Number of convolutional filters.
    output_dim : int
        Dimension of output after hidden layer. Should be 1 for regression or
        num_classes for classification.
    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.
    time_dependent : bool
        If True adds time as input, making ODE time dependent.
    non_linearity : string
        One of 'relu' and 'softplus'
    tol : float
        Error tolerance.
    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    return_sequences : bool
        Whether to return the Convolution outputs, or the features after an
        affine transform.
    solver: ODE solver. Defaults to DOPRI5.
    """
    def __init__(self, num_filters, output_dim=1,
                 augment_dim=0, time_dependent=False, out_kernel_size=(1, 1),
                 non_linearity='relu', out_strides=(1, 1),
                 tol=1e-3, adjoint=False, solver='dopri5', **kwargs):

        dynamic = kwargs.pop('dynamic', True)
        super(Conv2dODENet, self).__init__(**kwargs, dynamic=dynamic)

        self.num_filters = num_filters
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.time_dependent = time_dependent
        self.tol = tol
        self.solver = solver
        self.output_kernel = out_kernel_size
        self.output_strides = out_strides

        odefunc = Conv2dODEFunc(num_filters, augment_dim,
                                time_dependent, non_linearity)

        self.odeblock = ODEBlock(odefunc, is_conv=True, tol=tol,
                                 adjoint=adjoint, solver=solver)

        self.output_layer = tf.keras.layers.Conv2D(self.output_dim,
                                                   kernel_size=out_kernel_size,
                                                   strides=out_strides,
                                                   padding='same')

    def call(self, x, training=None, return_features=False):
        features = self.odeblock(x, training=training)

        # TODO: Remove cast when Keras supports double
        pred = self.output_layer(tf.cast(features, tf.float32))

        if return_features:
            return features, pred
        else:
            return pred
