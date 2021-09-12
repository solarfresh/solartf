import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Conv2D,
                                     UpSampling2D, Dense, Dropout, Flatten, LeakyReLU)
from tensorflow.keras.regularizers import l2
from .activation import (hard_swish, relu)
from .block import (downsample_block, inverted_res_block, residual_block, resnet_block, upsample_block)
from .layer import (GaussianBlur, InstanceNormalization, ReflectionPadding2D)
from .util import get_filter_nb_by_depth


class Discriminator(tf.keras.Model):
    def __init__(self,
                 init_filters=64,
                 kernel_initializer=None,
                 num_downsampling=3,
                 name=None):
        super(Discriminator, self).__init__()

        self.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02) \
            if kernel_initializer is None else kernel_initializer

        self.init_filters = init_filters
        self.num_downsampling = num_downsampling
        self.name = name

    def call(self, inputs, training=None, mask=None):
        x = Conv2D(
            self.init_filters,
            (4, 4),
            strides=(2, 2),
            padding="same",
            kernel_initializer=self.kernel_initializer,
        )(inputs)
        x = LeakyReLU(0.2)(x)

        num_filters = self.init_filters
        for num_downsample_block in range(3):
            num_filters *= 2
            if num_downsample_block < 2:
                x = downsample_block(
                    x,
                    filters=num_filters,
                    activation=LeakyReLU(0.2),
                    kernel_size=(4, 4),
                    strides=(2, 2),
                )
            else:
                x = downsample_block(
                    x,
                    filters=num_filters,
                    activation=LeakyReLU(0.2),
                    kernel_size=(4, 4),
                    strides=(1, 1),
                )

        x = Conv2D(1, (4, 4),
                   strides=(1, 1),
                   padding="same",
                   kernel_initializer=self.kernel_initializer)(x)

        return Model(inputs=inputs, outputs=x, name=self.name)

    def get_config(self):
        return super(Discriminator, self).get_config()


class LeNet5(tf.keras.Model):
    def __init__(self, n_classes, dropout_rate=0.5):
        super(LeNet5, self).__init__()
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate

    def call(self, inputs, training=None, mask=None):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        activation = hard_swish

        x = Conv2D(16,
                   kernel_size=3,
                   strides=(2, 2),
                   padding='same',
                   use_bias=False,
                   name=f'lenet5_conv_0')(inputs)
        x = activation(x, name=f'lenet5_activation_0')

        x = Conv2D(32,
                   kernel_size=3,
                   padding='same',
                   use_bias=False,
                   name=f'lenet5_conv_1')(x)
        x = activation(x, name=f'lenet5_activation_1')

        x = Flatten()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(self.n_classes, activation="softmax")(x)

        return x

    def get_config(self):
        return super(LeNet5, self).get_config()


class FPN(tf.keras.Model):
    def __init__(self,
                 n_filters=40,
                 prefix=None,
                 kernel_initializer='he_normal',
                 l2_regularization=0.0005,
                 use_bias=True):
        super(FPN, self).__init__()
        self.n_filters = n_filters
        self.kernel_initializer = kernel_initializer
        self.l2_reg = l2_regularization
        self.use_bias = use_bias
        if prefix is None:
            self.prefix = 'fpn'

    def call(self, inputs, training=None, mask=None):
        n_features = len(inputs)
        feature_map_list = []
        for index in range(n_features):
            if index == 0:
                p = Conv2D(self.n_filters, (1, 1),
                           kernel_regularizer=l2(self.l2_reg),
                           name=f'{self.prefix}_c{index}_p{index}')(inputs[index])
            else:
                p = Add(name=f'{self.prefix}_p{index}_add')([
                    UpSampling2D(2)(p),
                    Conv2D(self.n_filters, (1, 1),
                           padding='same',
                           kernel_regularizer=l2(self.l2_reg),
                           name=f'{self.prefix}_c{index}_p{index}')(inputs[index])
                ])

            feature_map_list.append(Conv2D(self.n_filters, (3, 3),
                                           padding="SAME",
                                           kernel_regularizer=l2(self.l2_reg),
                                           name=f'{self.prefix}_p_conv_{index}')(p))

        return feature_map_list

    def get_config(self):
        return super(FPN, self).get_config()


class MobileNetV3(tf.keras.Model):
    def __init__(self,
                 alpha=1.0,
                 ref_filter_nb=32,
                 model_type='small',
                 minimalistic=False):
        super(MobileNetV3, self).__init__()
        self.minimalistic = minimalistic
        self.alpha = alpha
        self.model_type = model_type
        self.ref_filter_nb = ref_filter_nb

    def call(self, inputs, training=None, mask=None):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        if self.minimalistic:
            kernel = 3
            activation = relu
            se_ratio = None
        else:
            kernel = 5
            activation = hard_swish
            se_ratio = 0.25

        x = GaussianBlur(kernel_size=3, mu=0., sigma=1.5)(inputs)

        x = Conv2D(4 * self.ref_filter_nb // 10,
                   kernel_size=3,
                   strides=(2, 2),
                   padding='same',
                   use_bias=True,
                   name=f'mobilenetv3_{self.model_type}_conv_0')(x)
        x = BatchNormalization(axis=channel_axis,
                               epsilon=1e-3,
                               momentum=0.999,
                               name=f'mobilenetv3_{self.model_type}_conv/bn_0')(x)
        x = activation(x, name=f'mobilenetv3_{self.model_type}_activation_0')

        if self.model_type == 'small':
            x = self.mobilenet_v3_small_stack_fn(x, kernel, activation, se_ratio)
            last_point_ch = 1024
        else:
            raise NotImplementedError(f'{self.model_type} is not implemented yet...')

        last_conv_ch = get_filter_nb_by_depth(K.int_shape(x)[channel_axis] * 6)
        if self.alpha > 1.0:
            last_point_ch = get_filter_nb_by_depth(last_point_ch * self.alpha)

        x = Conv2D(last_conv_ch,
                   kernel_size=1,
                   padding='same',
                   use_bias=True,
                   name=f'mobilenetv3_{self.model_type}_conv_1')(x)
        x = BatchNormalization(axis=channel_axis,
                               epsilon=1e-3,
                               momentum=0.999,
                               name=f'mobilenetv3_{self.model_type}_conv/bn_1')(x)
        x = activation(x, name=f'mobilenetv3_{self.model_type}_activation_1')
        x = Conv2D(last_point_ch,
                   kernel_size=1,
                   padding='same',
                   use_bias=True,
                   name=f'mobilenetv3_{self.model_type}_conv_2')(x)
        x = activation(x, name=f'mobilenetv3_{self.model_type}_activation_2')

        return Model(inputs, x)

    def mobilenet_v3_small_stack_fn(self, x, kernel, activation, se_ratio):
        x = inverted_res_block(x, 1, get_filter_nb_by_depth(self.ref_filter_nb, alpha=self.alpha),
                               3, 2, se_ratio, relu, 0)
        x = inverted_res_block(x, 72. / 16, get_filter_nb_by_depth(self.ref_filter_nb, alpha=self.alpha),
                               3, 2, None, relu, 1)
        x = inverted_res_block(x, 88. / 24, get_filter_nb_by_depth(self.ref_filter_nb, alpha=self.alpha),
                               3, 1, None, relu, 2)
        x = inverted_res_block(x, 4, get_filter_nb_by_depth(self.ref_filter_nb, alpha=self.alpha),
                               kernel, 2, se_ratio, activation, 3)
        x = inverted_res_block(x, 6, get_filter_nb_by_depth(self.ref_filter_nb, alpha=self.alpha),
                               kernel, 1, se_ratio, activation, 4)
        x = inverted_res_block(x, 6, get_filter_nb_by_depth(self.ref_filter_nb, alpha=self.alpha),
                               kernel, 1, se_ratio, activation, 5)
        x = inverted_res_block(x, 3, get_filter_nb_by_depth(self.ref_filter_nb, alpha=self.alpha),
                               kernel, 1, se_ratio, activation, 6)
        x = inverted_res_block(x, 3, get_filter_nb_by_depth(self.ref_filter_nb, alpha=self.alpha),
                                kernel, 1, se_ratio, activation, 7)
        x = inverted_res_block(x, 6, get_filter_nb_by_depth(self.ref_filter_nb, alpha=self.alpha),
                               kernel, 2, se_ratio, activation, 8)
        x = inverted_res_block(x, 6, get_filter_nb_by_depth(self.ref_filter_nb, alpha=self.alpha),
                               kernel, 1, se_ratio, activation, 9)
        x = inverted_res_block(x, 6, get_filter_nb_by_depth(self.ref_filter_nb, alpha=self.alpha),
                               kernel, 1, se_ratio, activation, 10)
        return x

    def get_config(self):
        return super(MobileNetV3, self).get_config()


class ResNetV2(tf.keras.Model):
    def __init__(self,
                 num_res_blocks=3,
                 num_stage=3,
                 num_filters_in=16):
        super(ResNetV2, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.num_stage = num_stage
        self.num_filters_in = num_filters_in
        self.depth = self.num_res_blocks * 9 + 2

    def call(self, inputs, training=None, mask=None):
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = resnet_block(inputs=inputs,
                         num_filters=self.num_filters_in,
                         conv_first=True)

        # Instantiate the stack of residual units
        num_filters_in = self.num_filters_in
        for stage in range(self.num_stage):
            for res_block in range(self.num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2  # downsample

                # bottleneck residual unit
                y = resnet_block(inputs=x,
                                 num_filters=num_filters_in,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=activation,
                                 batch_normalization=batch_normalization,
                                 conv_first=False)
                y = resnet_block(inputs=y,
                                 num_filters=num_filters_in,
                                 conv_first=False)
                y = resnet_block(inputs=y,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_block(inputs=x,
                                     num_filters=num_filters_out,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = Add()([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return Model(inputs, x)

    def get_config(self):
        return super(ResNetV2, self).get_config()


class ResnetGenerator(tf.keras.Model):
    def __init__(self,
                 init_filters=64,
                 num_downsampling_blocks=2,
                 num_residual_blocks=9,
                 num_upsample_blocks=2,
                 kernel_initializer=None,
                 gamma_initializer=None,
                 name=None,):
        super(ResnetGenerator, self).__init__()

        self.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02) \
            if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = RandomNormal(mean=0.0, stddev=0.02) \
            if gamma_initializer is None else gamma_initializer

        self.init_filters = init_filters
        self.num_downsampling_blocks = num_downsampling_blocks
        self.num_residual_blocks = num_residual_blocks
        self.num_upsample_blocks = num_upsample_blocks
        self.name = name

    def call(self, inputs, training=None, mask=None):
        x = ReflectionPadding2D(padding=(3, 3))(inputs)
        x = Conv2D(self.init_filters, (7, 7), kernel_initializer=self.kernel_initializer, use_bias=False)(x)
        x = InstanceNormalization(gamma_initializer=self.gamma_initializer)(x)
        x = Activation("relu")(x)

        # Downsampling
        filters = self.init_filters
        for _ in range(self.num_downsampling_blocks):
            filters *= 2
            x = downsample_block(x, filters=filters, activation=Activation("relu"))

        # Residual blocks
        for _ in range(self.num_residual_blocks):
            x = residual_block(x, activation=Activation("relu"))

        # Upsampling
        for _ in range(self.num_upsample_blocks):
            filters //= 2
            x = upsample_block(x, filters, activation=Activation("relu"))

        # Final block
        x = ReflectionPadding2D(padding=(3, 3))(x)
        x = Conv2D(3, (7, 7), padding="valid")(x)
        x = Activation("tanh")(x)

        return Model(inputs, x, name=self.name)

    def get_config(self):
        return super(ResnetGenerator, self).get_config()
