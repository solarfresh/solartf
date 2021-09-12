from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, Activation, BatchNormalization, Conv2D, Conv2DTranspose,
                                     DepthwiseConv2D, GlobalAveragePooling2D, Multiply,
                                     ReLU, Reshape, ZeroPadding2D)
from tensorflow.keras.regularizers import l2
from .activation import hard_sigmoid
from .layer import (InstanceNormalization, ReflectionPadding2D)
from .util import (correct_pad, get_filter_nb_by_depth)


def downsample_block(
    x,
    filters,
    activation,
    kernel_initializer=None,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=None,
    use_bias=False,
):

    kernel_initializer = RandomNormal(mean=0.0, stddev=0.02) \
        if kernel_initializer is None else kernel_initializer
    gamma_initializer = RandomNormal(mean=0.0, stddev=0.02) \
        if gamma_initializer is None else gamma_initializer

    x = Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
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
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
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
        x = Conv2D(get_filter_nb_by_depth(infilters * expansion),
                   kernel_size=1,
                   padding='same',
                   use_bias=False,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(axis=channel_axis,
                               epsilon=1e-3,
                               momentum=0.999,
                               name=prefix + 'expand/BatchNorm')(x)
        x = activation(x)

    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(x, kernel_size),
                          name=prefix + 'depthwise/pad')(x)

    x = DepthwiseConv2D(kernel_size,
                        strides=stride,
                        padding='same' if stride == 1 else 'valid',
                        use_bias=False,
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=channel_axis,
                           epsilon=1e-3,
                           momentum=0.999,
                           name=prefix + 'depthwise/BatchNorm')(x)
    x = activation(x)

    if se_ratio:
        x = se_block(x, get_filter_nb_by_depth(infilters * expansion), se_ratio, prefix)

    x = Conv2D(filters,
               kernel_size=1,
               padding='same',
               use_bias=False,
               name=prefix + 'project')(x)
    x = BatchNormalization(axis=channel_axis,
                           epsilon=1e-3,
                           momentum=0.999,
                           name=prefix + 'project/BatchNorm')(x)

    if stride == 1 and infilters == filters:
        x = Add(name=prefix + 'Add')([shortcut, x])
    return x


def residual_block(
    x,
    activation,
    kernel_initializer=None,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=None,
    use_bias=False,
):
    prefix = 'residual_block/'
    dim = x.shape[-1]
    input_tensor = x

    kernel_initializer = RandomNormal(mean=0.0, stddev=0.02) \
        if kernel_initializer is None else kernel_initializer
    gamma_initializer = RandomNormal(mean=0.0, stddev=0.02) \
        if gamma_initializer is None else gamma_initializer

    x = ReflectionPadding2D()(input_tensor)
    x = Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = Add(name=prefix + 'Add')([input_tensor, x])
    return x


def se_block(inputs, filters, se_ratio, prefix):
    x = GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(inputs)

    if K.image_data_format() == 'channels_first':
        x = Reshape((filters, 1, 1))(x)
    else:
        x = Reshape((1, 1, filters))(x)

    x = Conv2D(get_filter_nb_by_depth(filters * se_ratio),
               kernel_size=1,
               padding='same',
               name=prefix + 'squeeze_excite/Conv')(x)
    x = ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = Conv2D(filters,
               kernel_size=1,
               padding='same',
               name=prefix + 'squeeze_excite/Conv_1')(x)
    x = hard_sigmoid(x)
    x = Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


def upsample_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=None,
    gamma_initializer=None,
    use_bias=False,
):

    kernel_initializer = RandomNormal(mean=0.0, stddev=0.02) \
        if kernel_initializer is None else kernel_initializer
    gamma_initializer = RandomNormal(mean=0.0, stddev=0.02) \
        if gamma_initializer is None else gamma_initializer

    x = Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x
