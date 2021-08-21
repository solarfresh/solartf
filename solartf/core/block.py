from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Add, BatchNormalization, Conv2D, DepthwiseConv2D,
                                     GlobalAveragePooling2D, Multiply, ReLU, Reshape,
                                     ZeroPadding2D)
from .activation import hard_sigmoid
from .util import (correct_pad, get_filter_nb_by_depth)


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
