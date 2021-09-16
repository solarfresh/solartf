from tensorflow.keras.activations import softmax
from tensorflow.keras import layers


class HardSigmoid(layers.Layer):
    def __init__(self, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        self.relu = layers.ReLU(6.)

    def call(self, x, mask=None, *args, **kwargs):
        return self.relu((x + 3.) * (1. / 6.))

    def get_config(self):
        base_config = super(HardSigmoid, self).get_config()
        return base_config


class HardSwish(layers.Layer):
    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)
        self.multiply = layers.Multiply()
        self.hard_sigmoid = HardSigmoid()

    def call(self, inputs, mask=None, *args, **kwargs):
        x = inputs
        x = self.hard_sigmoid(x)
        return self.multiply([x, inputs])

    def get_config(self):
        base_config = super(HardSwish, self).get_config()
        return base_config


# todo: functions will be instead by Layer classes
def relu(x, name=None):
    return layers.ReLU(name=name)(x)


def hard_sigmoid(x, name=None):
    return layers.ReLU(6., name=name)(x + 3.) * (1. / 6.)


def hard_swish(x, name=None):
    return layers.Multiply(name=name)([hard_sigmoid(x, name=f'{name}/hard_sigmoid' if name is not None else None), x])


def softmax_axis(axis):
    def soft(x):
        return softmax(x, axis=axis)
    return soft
