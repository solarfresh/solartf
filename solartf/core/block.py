from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from . import activation as solar_activation
from . import layer as solartf_layers
from .util import (get_filter_nb_by_depth,)


class Conv2DBlock(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 use_bias=False,
                 batch_normalization=True,
                 normalize_axis=0,
                 normalize_epsilon=1e-3,
                 normalize_momentum=0.999,
                 activation=None,
                 **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.batch_normalization = batch_normalization
        self.normalize_axis = normalize_axis
        self.normalize_epsilon = normalize_epsilon
        self.normalize_momentum = normalize_momentum
        self.activation = activation

        self.conv = layers.Conv2D(filters=self.filters,
                                  kernel_size=self.kernel_size,
                                  strides=self.strides,
                                  padding=self.padding,
                                  use_bias=self.use_bias)

        if self.batch_normalization is not None:
            self.batch_normalization_layer = layers.BatchNormalization(
                axis=self.normalize_axis,
                epsilon=self.normalize_epsilon,
                momentum=self.normalize_momentum
            )

        if self.activation is not None:
            self.activation_layer = layers.Activation(activation)

    def build(self, input_shape):
        self.input_spec = layers.InputSpec(shape=input_shape)

    def call(self, x, *args, **kwargs):
        x = self.conv(x)

        if self.batch_normalization is not None:
            x = self.batch_normalization_layer(x)

        if self.activation is not None:
            x = self.activation_layer(x)

        return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias,
            'batch_normalization': self.batch_normalization,
            'normalize_axis': self.normalize_axis,
            'normalize_epsilon': self.normalize_epsilon,
            'normalize_momentum': self.normalize_momentum,
            'activation': self.activation
        }
        base_config = super(Conv2DBlock, self).get_config()
        base_config.update(config)
        return base_config


class DepthwiseConv2DBlock(layers.Layer):
    def __init__(self,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 use_bias=False,
                 batch_normalization=True,
                 normalize_axis=0,
                 normalize_epsilon=1e-3,
                 normalize_momentum=0.999,
                 activation=None,
                 **kwargs):
        super(DepthwiseConv2DBlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.batch_normalization = batch_normalization
        self.normalize_axis = normalize_axis
        self.normalize_epsilon = normalize_epsilon
        self.normalize_momentum = normalize_momentum
        self.activation = activation

        self.conv = layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias)

        if self.batch_normalization is not None:
            self.batch_normalization_layer = layers.BatchNormalization(
                axis=self.normalize_axis,
                epsilon=self.normalize_epsilon,
                momentum=self.normalize_momentum
            )

        if self.activation is not None:
            self.activation_layer = layers.Activation(activation)

    def build(self, input_shape):
        self.input_spec = layers.InputSpec(shape=input_shape)

    def call(self, x, *args, **kwargs):
        x = self.conv(x)

        if self.batch_normalization is not None:
            x = self.batch_normalization_layer(x)

        if self.activation is not None:
            x = self.activation_layer(x)

        return x

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias,
            'batch_normalization': self.batch_normalization,
            'normalize_axis': self.normalize_axis,
            'normalize_epsilon': self.normalize_epsilon,
            'normalize_momentum': self.normalize_momentum,
            'activation': self.activation
        }
        base_config = super(DepthwiseConv2DBlock, self).get_config()
        base_config.update(config)
        return base_config


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
        self.inst_norm = solartf_layers.InstanceNormalization(gamma_initializer=self.gamma_initializer)

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


class InvertedResBlock(layers.Layer):
    def __init__(self,
                 infilters,
                 expansion,
                 filters,
                 kernel_size,
                 strides,
                 se_ratio=None,
                 **kwargs):
        super(InvertedResBlock, self).__init__(**kwargs)
        self.channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        self.infilters = infilters
        self.expansion = expansion
        self.filters = filters
        self.kernel_size= kernel_size
        self.strides = strides
        self.se_ratio = se_ratio

        self.conv_expand = Conv2DBlock(
            filters=get_filter_nb_by_depth(self.infilters * self.expansion),
            kernel_size=1,
            padding='same',
            use_bias=False,
            normalize_axis=self.channel_axis,
            normalize_epsilon=1e-3,
            normalize_momentum=0.999,
            activation='relu'
        )
        if self.strides == 2:
            self.zero_padding = solartf_layers.CorrectZeroPadding(
                kernel_size=self.kernel_size,
            )

        self.depthwise_conv = DepthwiseConv2DBlock(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            use_bias=False,
            normalize_axis=self.channel_axis,
            normalize_epsilon=1e-3,
            normalize_momentum=0.999,
            activation='relu'
        )
        self.conv_out = Conv2DBlock(
            filters=self.filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            normalize_axis=self.channel_axis,
            normalize_epsilon=1e-3,
            normalize_momentum=0.999,
            activation=None
        )

        if self.se_ratio:
            self.se_block = SEBlock(
                filters=get_filter_nb_by_depth(self.infilters * self.expansion),
                se_ratio=self.se_ratio
            )

    def build(self, input_shape):
        self.input_spec = layers.InputSpec(shape=input_shape)

    def call(self, x, *args, **kwargs):
        shortcut = x

        x = self.conv_expand(x)
        # todo: it would make FPN wrong, and will be fixed in the future
        # if self.strides == 2:
        #     x = self.zero_padding(x)

        x = self.depthwise_conv(x)
        if self.se_ratio:
            x = self.se_block(x)

        x = self.conv_out(x)
        if self.strides == 1 and self.infilters == self.filters:
            x = layers.add([shortcut, x])

        return x

    def get_config(self):
        config = {
            'channel_axis': self.channel_axis,
            'infilters': self.infilters,
            'expansion': self.expansion,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.strides,
            'se_ratio': self.se_ratio
        }
        base_config = super(InvertedResBlock, self).get_config()
        base_config.update(config)
        return base_config


class ResNetBlock(layers.Layer):
    def __init__(self,
                 num_filters_in=16,
                 num_filters_out=32,
                 kernel_size=3,
                 kernel_initializer='he_normal',
                 kernel_regularizer=l2(1e-4),
                 stage_index=0,
                 block_index=0,
                 **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.num_filters_in = num_filters_in
        self.num_filters_out = num_filters_out
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        # todo: to correct activation functions
        self.activation_in = layers.Activation('relu')
        self.activation_middle = layers.Activation('relu')
        self.activation_out = layers.Activation('relu')
        self.normalization_in = layers.BatchNormalization()
        self.normalization_middle = layers.BatchNormalization()
        self.normalization_out = layers.BatchNormalization()
        self.stage_index = stage_index
        self.block_index = block_index

        self.conv_in = layers.Conv2D(self.num_filters_in,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_regularizer=self.kernel_regularizer)
        self.conv_middle = layers.Conv2D(
            self.num_filters_in,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer)
        self.conv_in_downsample = layers.Conv2D(
            self.num_filters_in,
            kernel_size=1,
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
        self.conv_shortcut_out = layers.Conv2D(
            self.num_filters_out,
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

        activation = self.activation_in
        normalization = self.normalization_in
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
                              activation=self.activation_middle,
                              normalization=self.normalization_middle,)
        y = self._resnet_conv(inputs=y,
                              conv=self.conv_out,
                              activation=self.activation_out,
                              normalization=self.normalization_out,)
        if self.block_index == 0:
            # linear projection residual shortcut connection to match
            # changed dims
            x = self._resnet_conv(
                inputs=x,
                conv=self.conv_out_downsample if self.stage_index > 0 else self.conv_shortcut_out,
                activation=None,
                normalization=None)
        return layers.add([x, y])

    def get_config(self):
        config = {
            'num_filters_in': self.self.num_filters_in,
            'num_filters_out': self.num_filters_out,
            'kernel_size': self.kernel_size,
            'stage_index': self.stage_index,
            'block_index': self.block_index
        }
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

        self.reflect_padding = solartf_layers.ReflectionPadding2D()
        self.conv = layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=(1, 1),
            kernel_initializer=self.kernel_initializer,
            padding='valid',
            use_bias=self.use_bias,
        )

        self.inst_norm = solartf_layers.InstanceNormalization(gamma_initializer=self.gamma_initializer)

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


class SEBlock(layers.Layer):
    def __init__(self,
                 filters,
                 se_ratio,
                 **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.filters = filters
        self.se_ratio = se_ratio

        self.global_pool = layers.GlobalAveragePooling2D()
        if K.image_data_format() == 'channels_first':
            self.reshape = layers.Reshape((self.filters, 1, 1))
        else:
            self.reshape = layers.Reshape((1, 1, self.filters))

        self.conv_in = layers.Conv2D(
            filters=get_filter_nb_by_depth(self.filters * self.se_ratio),
            kernel_size=1,
            padding='same',
        )
        self.activation_relu = layers.ReLU()
        self.conv_out = layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            padding='same',
        )
        self.activation_hard_sigmoid = solar_activation.HardSigmoid()

    def build(self, input_shape):
        self.input_spec = layers.InputSpec(shape=input_shape)

    def call(self, inputs, *args, **kwargs):
        x = self.global_pool(inputs)
        x = self.reshape(x)
        x = self.conv_in(x)
        x = self.activation_relu(x)
        x = self.conv_out(x)
        x = self.activation_hard_sigmoid(x)
        return layers.multiply([inputs, x])

    def get_config(self):
        config = {
        }
        base_config = super(SEBlock, self).get_config()
        base_config.update(config)
        return base_config


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
        self.inst_norm = solartf_layers.InstanceNormalization(gamma_initializer=self.gamma_initializer)

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
