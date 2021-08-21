from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import (Multiply, ReLU)


def relu(x, name=None):
    return ReLU(name=name)(x)


def hard_sigmoid(x, name=None):
    return ReLU(6., name=name)(x + 3.) * (1. / 6.)


def hard_swish(x, name=None):
    return Multiply(name=name)([hard_sigmoid(x, name=f'{name}/hard_sigmoid' if name is not None else None), x])


def softmax_axis(axis):
    def soft(x):
        return softmax(x, axis=axis)
    return soft
