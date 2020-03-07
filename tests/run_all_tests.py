import unittest

import tensorflow as tf
from tests.problems import *
from tests.api_tests import *
from tests.gradient_tests import *
from tests.odeint_tests import *
from tests.model_tests import *

if __name__ == '__main__':
    with tf.device('cpu'):
        unittest.main()
