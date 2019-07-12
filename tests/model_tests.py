import sys
import unittest
import tensorflow as tf

import tfdiffeq
import tfdiffeq.models as models

if not tf.executing_eagerly():
    tf.enable_v2_behavior()


class TestModel(unittest.TestCase):

    def test_dense_odenet(self):
        model = models.ODENet(hidden_dim=10, output_dim=2)

        data = tf.zeros([10, 5])
        out = model(data)

        assert out.shape == (10, 2)
        assert len(model.layers) == 2

    def test_dense_odenet_time_dependent(self):
        model = models.ODENet(hidden_dim=10, output_dim=2, time_dependent=True)

        data = tf.zeros([10, 5])
        out = model(data)

        assert out.shape == (10, 2)
        assert len(model.layers) == 2

    def test_conv_odenet(self):
        model = models.Conv2dODENet(num_filters=10, output_dim=2)

        data = tf.zeros([10, 5, 5, 3])
        out = model(data)

        assert out.shape == (10, 5, 5, 2)
        assert len(model.layers) == 2

    def test_conv_odenet_time_dependent(self):
        model = models.Conv2dODENet(num_filters=10, output_dim=2, time_dependent=True)

        data = tf.zeros([10, 5, 5, 3])
        out = model(data)

        assert out.shape == (10, 5, 5, 2)
        assert len(model.layers) == 2


if __name__ == '__main__':
    unittest.main(__file__)
