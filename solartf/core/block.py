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


class ResNetBlock(layers.Layer):
    def __init__(self,
                 num_filters_in=16,
                 num_filters_out=32,
                 kernel_size=3,
                 kernel_initializer='he_normal',
                 kernel_regularizer=l2(1e-4),
                 activation=None,
                 normalization=None,
                 stage_index=0,
                 block_index=0,
                 **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.num_filters_in = num_filters_in
        self.num_filters_out = num_filters_out
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.activation = activation
        self.normalization = normalization
        self.stage_index = stage_index
        self.block_index = block_index

        self.conv_in = layers.Conv2D(self.num_filters_in,
                                     kernel_size=kernel_size,
                                     strides=1,
                                     padding='same',
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_regularizer=self.kernel_regularizer)
        self.conv_middle = layers.Conv2D(
            self.num_filters_in,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer)
        self.conv_in_downsample = layers.Conv2D(
            self.num_filters_in,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer)
        self.conv_out = layers.Conv2D(self.num_filters_out,
                                      kernel_size=1,
                                      strides=1,
                                      padding='same',
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_regularizer=self.kernel_regularizer)
        self.conv_out_downsample = layers.Conv2D(
            self.num_filters_out,
            kernel_size=1,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer)

    def build(self, input_shape):
        self.input_spec = layers.InputSpec(shape=input_shape)

    def call(self, x, *args, **kwargs):

        activation = self.activation
        normalization = self.normalization
        conv = self.conv_in
        if self.stage_index == 0:
            if self.block_index == 0:
                activation = None
                normalization = None
        else:
            if self.block_index == 0:
                conv = self.conv_in_downsample

        # bottleneck residual unit
        y = self._resnet_conv(inputs=x,
                              conv=conv,
                              activation=activation,
                              normalization=normalization,)
        y = self._resnet_conv(inputs=y,
                              conv=self.conv_middle,
                              activation=self.activation,
                              normalization=self.normalization,)
        y = self._resnet_conv(inputs=y,
                              conv=self.conv_out,
                              activation=self.activation,
                              normalization=self.normalization,)
        if self.block_index == 0:
            # linear projection residual shortcut connection to match
            # changed dims
            x = self._resnet_conv(
                inputs=x,
                conv=self.conv_out_downsample if self.stage_index > 0 else self.conv_out,
                activation=None,
                normalization=None)
        return layers.add([x, y])

    def get_config(self):
        config = {}
        base_config = super(ResNetBlock, self).get_config()
        base_config.update(config)
        return base_config

    def _resnet_conv(self,
                     inputs,
                     conv,
                     normalization=None,
                     activation=None):

        x = inputs
        if normalization is not None:
            x = normalization(x)

        if activation is not None:
            x = activation(x)

        x = conv(x)

        return x


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
