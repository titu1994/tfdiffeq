import argparse
import logging
import os
import time

import numpy as np
import tensorflow as tf

from tfdiffeq import odeint

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-4)
parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'euler', 'huen'], default='euler')
# parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--data_aug', type=eval, default=False, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--save', type=str, default='./mnist')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer
    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes
    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        dim = input_shape[self.axis].value

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape.as_list()),
                                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint,
                                         dtype=self.dtype,
                                         trainable=True)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.keras.backend.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = tf.keras.backend.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = tf.stack(group_shape)
        inputs = tf.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = tf.keras.backend.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = tf.keras.backend.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (tf.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = tf.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = tf.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = tf.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = tf.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        return input_shape


tf.keras.utils.get_custom_objects().update({'GroupNormalization': GroupNormalization})


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return tf.keras.layers.Conv2D(out_planes, kernel_size=(3, 3), strides=(stride, stride),
                                  padding='same', use_bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return tf.keras.layers.Conv2D(out_planes, kernel_size=(1, 1), strides=(stride, stride),
                                  use_bias=False)


def norm(dim):
    return GroupNormalization(min(32, dim))


class ResBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def call(self, x):
        shortcut = x

        out = tf.nn.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = tf.nn.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(tf.keras.Model):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = tf.keras.layers.Conv2DTranspose if transpose else tf.keras.layers.Conv2D

        padding = 'same'if padding == 1 else 'valid'
        self._layer = module(
            dim_out, kernel_size=(ksize, ksize), strides=(stride, stride), padding=padding,
            dilation_rate=dilation,
            use_bias=bias
        )

    def call(self, t, x):
        t = tf.cast(t, tf.float32)
        tt = tf.ones_like(x[:, :, :, :1]) * t  # channel dim = -1
        ttx = tf.concat([tt, x], -1)  # concat at channel dim
        return self._layer(ttx)


class ODEfunc(tf.keras.Model):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def call(self, t, x):
        self.nfe += 1
        x = tf.cast(x, tf.float32)  # needs an explicit cast
        out = self.norm1(x)
        out = tf.nn.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = tf.nn.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(tf.keras.Model):

    def __init__(self, odefunc, **kwargs):
        super(ODEBlock, self).__init__(**kwargs)
        self.odefunc = odefunc
        self.integration_time = tf.convert_to_tensor([0., 1.], dtype=tf.float32)

    def call(self, x):
        # self.integration_time = tf.cast(self.integration_time, x.dtype)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol,
                     method=args.method)
        return tf.cast(out[1], tf.float32)  # necessary cast

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(tf.keras.Model):

    def __init__(self):
        super(Flatten, self).__init__()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        # shape = tf.reduce_prod(tf.convert_to_tensor(x.shape[1:])).numpy()
        # return x.view(-1, shape)
        return self.flatten(x)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=100, perc=1.0):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)

    x_train = tf.cast(x_train, tf.float32) / 255.
    x_test = tf.cast(x_test, tf.float32) / 255.

    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(10000)

    if data_aug:
        def augment(x, y):
            x_aug = tf.image.resize_image_with_pad(x, 32, 32)
            x_aug = tf.image.random_crop(x_aug, [28, 28, 1])
            return x_aug, y

        train_dataset = train_dataset.apply(tf.data.experimental.map_and_batch(augment, batch_size,
                                                                               num_parallel_batches=1))
    else:
        train_dataset = train_dataset.batch(batch_size)

    # train_dataset = train_dataset.apply(tf.data.experimental.prefetch_to_device(device))

    # evaluate the train dataset
    eval_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    eval_dataset = eval_dataset.batch(test_batch_size)
    # eval_dataset = eval_dataset.apply(tf.data.experimental.prefetch_to_device(device))

    # test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(test_batch_size)
    # test_dataset = test_dataset.apply(tf.data.experimental.prefetch_to_device(device))

    return train_dataset, test_dataset, eval_dataset


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


# def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
#     initial_learning_rate = args.lr * batch_size / batch_denom
#
#     boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
#     vals = [initial_learning_rate * decay for decay in decay_rates]
#
#     def learning_rate_fn(itr):
#         lt = [itr < b for b in boundaries] + [True]
#         i = np.argmax(lt)
#         return vals[i]
#
#     return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    print("Evaluating dataset ...")
    samples = 0
    for i, (x, y) in enumerate(dataset_loader):
        target_class = np.argmax(y.numpy(), axis=1)
        predicted_class = np.argmax(model(x).numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
        samples += x.shape.as_list()[0]
    return total_correct / samples


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

device = 'gpu:' + str(args.gpu) if tf.test.is_gpu_available() else 'cpu:0'
with tf.device(device):
    is_odenet = args.network == 'odenet'

    if args.downsampling_method == 'conv':
        downsampling_layers = [
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'),
            norm(32),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same'),
            norm(32),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same'),
        ]
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'),
            ResBlock(32, 32, stride=2, downsample=conv1x1(32, 32, 2)),
            ResBlock(32, 32, stride=2, downsample=conv1x1(32, 32, 2)),
        ]

    feature_layers = [ODEBlock(ODEfunc(32))] if is_odenet else [ResBlock(32, 32) for _ in range(1)]

    fc_layers = [norm(32),
                 tf.keras.layers.ReLU(),
                 tf.keras.layers.AveragePooling2D((1, 1)),
                 Flatten(),
                 tf.keras.layers.Dense(10, activation='softmax')]

    layers = (*downsampling_layers, *feature_layers, *fc_layers)
    print("Nnm layers", len(layers))

    model = tf.keras.models.Sequential(layers)

    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    # data_gen = inf_generator(train_loader)
    # batches_per_epoch = 60000 // args.batch_size

    global_step = tf.train.get_or_create_global_step()
    num_steps_per_epoch = 60000 // args.batch_size
    boundary_epochs = [60, 100, 140]
    boundary_iterations = [num_steps_per_epoch * e for e in boundary_epochs]
    lrs = [0.001, 0.0005, 0.0001, 0.00005]

    learning_rate = tf.train.piecewise_constant(global_step, boundary_iterations, lrs)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    saver = tf.train.Checkpoint(model=model,
                                optimizer=optimizer,
                                global_step=global_step)

    for epoch in range(args.nepochs):
        for x, y in iter(train_loader):
            # optimizer.zero_grad()
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = tf.keras.losses.categorical_crossentropy(logits, y)

            if is_odenet:
                nfe_forward = feature_layers[0].nfe
                feature_layers[0].nfe = 0

            grads = tape.gradient(loss, model.variables)
            grad_vars = zip(grads, model.variables)

            optimizer.apply_gradients(grad_vars, global_step)

            if is_odenet:
                nfe_backward = feature_layers[0].nfe
                feature_layers[0].nfe = 0

            batch_time_meter.update(time.time() - end)
            if is_odenet:
                f_nfe_meter.update(nfe_forward)
                b_nfe_meter.update(nfe_backward)
            end = time.time()

        train_acc = accuracy(model, train_eval_loader)
        val_acc = accuracy(model, test_loader)

        if val_acc > best_acc:
            path = os.path.join(args.save, 'model')

            saver.save(path)
            best_acc = val_acc

        logger.info(
            "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
            "Train Acc {:.4f} | Test Acc {:.4f}".format(
                epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                b_nfe_meter.avg, train_acc, val_acc
            )
        )

    logger.info('Number of parameters: {}'.format(model.count_params()))
    logger.info('Model Info:')

    def summary(line):
        logger.info(line)
        print(line)

    model.summary(print_fn=summary)
