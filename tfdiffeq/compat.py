import tensorflow as tf

_VERSION = None

if _VERSION is None:
    if tf.version.VERSION.startswith("1."):
        _VERSION = 1

    else:
        _VERSION = 2


def assign(tensor, val):
    """
    Compatibility assignment operation

    Args:
        tensor: Tensor, to be assigned value of T2.
        val: Tensor or python value, which will be assigned to T1.

    Returns:
        Assigned Tensor
    """

    if _VERSION == 1:
        tf.assign(tensor, val)
    else:
        tf.compat.v1.assign(tensor, val)

    return tensor
