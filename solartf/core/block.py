from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from .activation import hard_sigmoid
from .layer import (InstanceNormalization, ReflectionPadding2D)
from .util import (correct_pad, get_filter_nb_by_depth)


class DownsampleBlock(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_initializer=None,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 padding="same",
                 gamma_initializer=None,
                 use_bias=False,
                 activation=None,
                 **kwargs):
        super(DownsampleBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias= use_bias
        self.activation = activation

        self.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02) \
            if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = RandomNormal(mean=0.0, stddev=0.02) \
            if gamma_initializer is None else gamma_initializer

        self.conv = layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            kernel_initializer=self.kernel_initializer,
            padding=self.padding,
            use_bias=self.use_bias,
        )
        self.inst_norm = InstanceNormalization(gamma_initializer=self.gamma_initializer)

    def build(self, input_shape):
        self.input_spec = layers.InputSpec(shape=input_shape)

    def call(self, x, *args, **kwargs):
        x = self.conv(x)
        x = self.inst_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'gamma_initializer': self.gamma_initializer,
        }
        base_config = super(DownsampleBlock, self).get_config()
        base_config.update(config)
        return base_config


def resnet_block(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = layers.Conv2D(num_filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        x = conv(x)
    return x


def inverted_res_block(x,
                       expansion,
                       filters,
                       kernel_size,
                       stride,
                       se_ratio,
                       activation,
                       name):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = K.int_shape(x)[channel_axis]

    if name:
        # Expand
        prefix = 'expanded_conv_{}/'.format(name)
        x = layers.Conv2D(get_filter_nb_by_depth(infilters * expansion),
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand/BatchNorm')(x)
        x = activation(x)

    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(x, kernel_size),
                                 name=prefix + 'depthwise/pad')(x)

    x = layers.DepthwiseConv2D(kernel_size,
                               strides=stride,
                               padding='same' if stride == 1 else 'valid',
                               use_bias=False,
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise/BatchNorm')(x)
    x = activation(x)

    if se_ratio:
        x = se_block(x, get_filter_nb_by_depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'project/BatchNorm')(x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
    return x


class ResidualBlock(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 kernel_initializer=None,
                 gamma_initializer=None,
                 use_bias=False,
                 activation=None,
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias= use_bias
        self.activation = activation

        self.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02) \
            if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = RandomNormal(mean=0.0, stddev=0.02) \
            if gamma_initializer is None else gamma_initializer

        self.reflect_padding = ReflectionPadding2D()
        self.conv = layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=(1, 1),
            kernel_initializer=self.kernel_initializer,
            padding='valid',
            use_bias=self.use_bias,
        )

        self.inst_norm = InstanceNormalization(gamma_initializer=self.gamma_initializer)

    def build(self, input_shape):
        self.input_spec = layers.InputSpec(shape=input_shape)

    def call(self, x, *args, **kwargs):
        input_tensor = x
        for _ in range(2):
            x = self.reflect_padding(x)
            x = self.conv(x)
            x = self.inst_norm(x)
            if self.activation is not None:
                x = self.activation(x)

        x = layers.add([input_tensor, x])
        return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'gamma_initializer': self.gamma_initializer,
        }
        base_config = super(ResidualBlock, self).get_config()
        base_config.update(config)
        return base_config


def se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(inputs)

    if K.image_data_format() == 'channels_first':
        x = layers.Reshape((filters, 1, 1))(x)
    else:
        x = layers.Reshape((1, 1, filters))(x)

    x = layers.Conv2D(
        get_filter_nb_by_depth(filters * se_ratio),
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv')(x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv_1')(x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


class UpsampleBlock(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 padding="same",
                 kernel_initializer=None,
                 gamma_initializer=None,
                 use_bias=False,
                 activation=None,
                 **kwargs):
        super(UpsampleBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias= use_bias
        self.activation = activation

        self.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02) \
            if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = RandomNormal(mean=0.0, stddev=0.02) \
            if gamma_initializer is None else gamma_initializer

        self.invert_conv = layers.Conv2DTranspose(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            kernel_initializer=self.kernel_initializer,
            padding=self.padding,
            use_bias=self.use_bias,
        )
        self.inst_norm = InstanceNormalization(gamma_initializer=self.gamma_initializer)

    def build(self, input_shape):
        self.input_spec = layers.InputSpec(shape=input_shape)

    def call(self, x, *args, **kwargs):
        x = self.invert_conv(x)
        x = self.inst_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'gamma_initializer': self.gamma_initializer,
        }
        base_config = super(UpsampleBlock, self).get_config()
        base_config.update(config)
        return base_config
