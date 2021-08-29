from tensorflow.keras.layers import (Conv2D, Reshape)


def anchor_detect_head_block(inputs,
                             filters,
                             reshape=None,
                             conv_block=None,
                             activation=None,
                             name=None):

    if conv_block is not None:
        x = conv_block(inputs, name=f'{name}/conv_block')
    else:
        x = inputs

    x = Conv2D(filters,
               kernel_size=1,
               padding='same',
               use_bias=False,
               name=f'{name}/conv' if name is not None else None)(x)

    if reshape is not None:
        x = Reshape(reshape)(x)

    if activation is not None:
        x = activation(x)

    return x
