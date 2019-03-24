import tensorflow as tf
from tfdiffeq.misc import move_to_device

__all__ = ['ODEModel']


class ODEModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super(ODEModel, self).__init__(**kwargs)

        self.dy = None

    def __call__(self, inputs, *args, **kwargs):
        result = super(ODEModel, self).__call__(inputs, *args, **kwargs)

        # if the result is not passed inside call,
        # then assume the gradient is pre-allocated in dy.
        if result is None:
            return self.dy

        else:
            return result

    def allocate_gradients(self, y):
        """
        Creates a cached gradient tensor that is reused,
        instead of allocating memory per call.

        Args:
            y: tf.Tensor input to the Model.

        Returns:
            a cached tf.Tensor which is the gradient.
        """
        if self.dy is None:
            dy = tf.zeros_like(y, dtype=y.dtype)
            dy = move_to_device(dy, y)
            self.dy = tf.Variable(dy, trainable=False, dtype=dy.dtype)
