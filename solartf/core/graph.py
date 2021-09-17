import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from .activation import (hard_swish, relu)
from . import layer as solartf_layers
from . import block as solartf_blocks
from .util import get_filter_nb_by_depth


class Discriminator(tf.keras.Model):
    def __init__(self,
                 init_filters=64,
                 kernel_initializer=None,
                 num_downsampling=3,
                 prefix=None):
        super(Discriminator, self).__init__()

        self.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02) \
            if kernel_initializer is None else kernel_initializer

        self.init_filters = init_filters
        self.num_downsampling = num_downsampling
        self.prefix = 'disc' if prefix is None else prefix

        self.activate_leaky_relu = layers.LeakyReLU(0.2)
        self.init_conv = layers.Conv2D(
            self.init_filters,
            (4, 4),
            strides=(2, 2),
            padding="same",
            kernel_initializer=self.kernel_initializer,
        )
        self.final_conv = layers.Conv2D(
            1, (4, 4),
            strides=(1, 1),
            padding="same",
            kernel_initializer=self.kernel_initializer)

        self.downsample_blocks = []
        num_filters = self.init_filters
        for num_downsample_block in range(self.num_downsampling - 1):
            num_filters *= 2
            self.downsample_blocks.append(solartf_blocks.DownsampleBlock(
                filters=num_filters,
                activation=self.activate_leaky_relu,
                kernel_size=(4, 4),
                strides=(2, 2),
            ))

        self.downsample_blocks.append(solartf_blocks.DownsampleBlock(
            filters=num_filters,
            activation=self.activate_leaky_relu,
            kernel_size=(4, 4),
            strides=(1, 1),
        ))

    def call(self, inputs):
        x = self.init_conv(inputs)
        x = self.activate_leaky_relu(x)

        for downsample_block in self.downsample_blocks:
            x = downsample_block(x)

        return self.final_conv(x)

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

        x = layers.Conv2D(16,
                          kernel_size=3,
                          strides=(2, 2),
                          padding='same',
                          use_bias=False,
                          name=f'lenet5_conv_0')(inputs)
        x = activation(x, name=f'lenet5_activation_0')

        x = layers.Conv2D(32,
                          kernel_size=3,
                          padding='same',
                          use_bias=False,
                          name=f'lenet5_conv_1')(x)
        x = activation(x, name=f'lenet5_activation_1')

        x = layers.Flatten()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.n_classes, activation="softmax")(x)

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
                p = layers.Conv2D(self.n_filters, (1, 1),
                                  kernel_regularizer=l2(self.l2_reg),
                                  name=f'{self.prefix}_c{index}_p{index}')(inputs[index])
            else:
                p = layers.Add(name=f'{self.prefix}_p{index}_add')([
                    layers.UpSampling2D(2)(p),
                    layers.Conv2D(self.n_filters, (1, 1),
                                  padding='same',
                                  kernel_regularizer=l2(self.l2_reg),
                                  name=f'{self.prefix}_c{index}_p{index}')(inputs[index])
                ])

            feature_map_list.append(layers.Conv2D(self.n_filters, (3, 3),
                                                  padding="SAME",
                                                  kernel_regularizer=l2(self.l2_reg),
                                                  name=f'{self.prefix}_p_conv_{index}')(p))

        return feature_map_list

    def get_config(self):
        return super(FPN, self).get_config()


class MobileNetV3Small(tf.keras.Model):
    def __init__(self,
                 last_point_ch=1024,
                 alpha=1.0,):
        super(MobileNetV3Small, self).__init__()
        self.alpha = alpha
        if self.alpha > 1.0:
            self.last_point_ch = get_filter_nb_by_depth(last_point_ch * self.alpha)
        else:
            self.last_point_ch = last_point_ch

        self.kernel_size = 3

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        self.conv_init = solartf_blocks.Conv2DBlock(
            filters=16,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            use_bias=True,
            normalize_axis=channel_axis,
            normalize_epsilon=1e-3,
            normalize_momentum=0.999,
            activation='relu'
        )
        self.inverted_res_blocks = [
            solartf_blocks.InvertedResBlock(
                expansion=1,
                infilters=16,
                filters=get_filter_nb_by_depth(16, alpha=self.alpha),
                kernel_size=3,
                strides=2,
                se_ratio=None,
            ),
            solartf_blocks.InvertedResBlock(
                expansion=72. / 16.,
                infilters=get_filter_nb_by_depth(16, alpha=self.alpha),
                filters=get_filter_nb_by_depth(24, alpha=self.alpha),
                kernel_size=3,
                strides=2,
                se_ratio=None,
            ),
            solartf_blocks.InvertedResBlock(
                expansion=88. / 24.,
                infilters=get_filter_nb_by_depth(24, alpha=self.alpha),
                filters=get_filter_nb_by_depth(24, alpha=self.alpha),
                kernel_size=3,
                strides=1,
                se_ratio=None,
            ),
            solartf_blocks.InvertedResBlock(
                expansion=4.,
                infilters=get_filter_nb_by_depth(24, alpha=self.alpha),
                filters=get_filter_nb_by_depth(40, alpha=self.alpha),
                kernel_size=self.kernel_size,
                strides=2,
                se_ratio=None,
            ),
            solartf_blocks.InvertedResBlock(
                expansion=6.,
                infilters=get_filter_nb_by_depth(40, alpha=self.alpha),
                filters=get_filter_nb_by_depth(40, alpha=self.alpha),
                kernel_size=self.kernel_size,
                strides=1,
                se_ratio=None,
            ),
            solartf_blocks.InvertedResBlock(
                expansion=6.,
                infilters=get_filter_nb_by_depth(40, alpha=self.alpha),
                filters=get_filter_nb_by_depth(40, alpha=self.alpha),
                kernel_size=self.kernel_size,
                strides=1,
                se_ratio=None,
            ),
            solartf_blocks.InvertedResBlock(
                expansion=3.,
                infilters=get_filter_nb_by_depth(40, alpha=self.alpha),
                filters=get_filter_nb_by_depth(48, alpha=self.alpha),
                kernel_size=self.kernel_size,
                strides=1,
                se_ratio=None,
            ),
            solartf_blocks.InvertedResBlock(
                expansion=3.,
                infilters=get_filter_nb_by_depth(48, alpha=self.alpha),
                filters=get_filter_nb_by_depth(48, alpha=self.alpha),
                kernel_size=self.kernel_size,
                strides=1,
                se_ratio=None,
            ),
            solartf_blocks.InvertedResBlock(
                expansion=6.,
                infilters=get_filter_nb_by_depth(48, alpha=self.alpha),
                filters=get_filter_nb_by_depth(96, alpha=self.alpha),
                kernel_size=self.kernel_size,
                strides=2,
                se_ratio=None,
            ),
            solartf_blocks.InvertedResBlock(
                expansion=6.,
                infilters=get_filter_nb_by_depth(96, alpha=self.alpha),
                filters=get_filter_nb_by_depth(96, alpha=self.alpha),
                kernel_size=self.kernel_size,
                strides=1,
                se_ratio=None,
            ),
            solartf_blocks.InvertedResBlock(
                expansion=6.,
                infilters=get_filter_nb_by_depth(96, alpha=self.alpha),
                filters=get_filter_nb_by_depth(96, alpha=self.alpha),
                kernel_size=self.kernel_size,
                strides=1,
                se_ratio=None,
            )
        ]
        self.conv_middle = solartf_blocks.Conv2DBlock(
            filters=get_filter_nb_by_depth(96, alpha=self.alpha),
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=True,
            normalize_axis=channel_axis,
            normalize_epsilon=1e-3,
            normalize_momentum=0.999,
            activation='relu'
        )
        self.conv_out = solartf_blocks.Conv2DBlock(
            filters=self.last_point_ch,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=True,
            batch_normalization=None,
            activation='relu'
        )

    def call(self, inputs, training=None, mask=None):
        x = self.conv_init(inputs)

        for inverted_res_block in self.inverted_res_blocks:
            x = inverted_res_block(x)

        x = self.conv_middle(x)
        x = self.conv_out(x)

        return x

    def get_config(self):
        config = {
            'last_point_ch': self.last_point_ch,
            'alpha': self.alpha,
        }
        base_config = super(MobileNetV3Small, self).get_config()
        base_config.update(config)
        return base_config


class ResNetV2(tf.keras.Model):
    def __init__(self,
                 num_res_blocks=3,
                 num_stage=3,
                 num_filters_in=16,
                 activation=None,
                 normalization=None,):
        super(ResNetV2, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.num_stage = num_stage
        self.num_filters_in = num_filters_in
        self.depth = self.num_res_blocks * 9 + 2
        self.init_conv = layers.Conv2D(
            self.num_filters_in,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(1e-4)
        )
        self.activation_init = layers.Activation("relu") if activation is None else activation
        self.normalization_init = layers.BatchNormalization() if normalization is None else normalization
        self.activation_final = layers.Activation("relu") if activation is None else activation
        self.normalization_final = layers.BatchNormalization() if normalization is None else normalization

        self.resnet_blocks = []
        num_filters_in = self.num_filters_in
        for stage_index in range(self.num_stage):
            for block_index in range(self.num_res_blocks):
                if stage_index == 0:
                    num_filters_out = num_filters_in * 4
                else:
                    num_filters_out = num_filters_in * 2

                self.resnet_blocks.append(solartf_blocks.ResNetBlock(
                    num_filters_in=num_filters_in,
                    num_filters_out=num_filters_out,
                    stage_index=stage_index,
                    block_index=block_index,))
            num_filters_in = num_filters_out

    def call(self, inputs, training=None, mask=None):
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths

        x = self.init_conv(inputs)
        x = self.normalization_init(x)
        x = self.activation_init(x)

        # Instantiate the stack of residual units
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x)

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = self.normalization_final(x)
        x = self.activation_final(x)
        return x

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
                 prefix=None):
        super(ResnetGenerator, self).__init__()

        self.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02) \
            if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = RandomNormal(mean=0.0, stddev=0.02) \
            if gamma_initializer is None else gamma_initializer

        self.init_filters = init_filters
        self.num_downsampling_blocks = num_downsampling_blocks
        self.num_residual_blocks = num_residual_blocks
        self.num_upsample_blocks = num_upsample_blocks
        self.prefix = 'resnet_gen' if prefix is None else prefix

        self.reflection_padding = solartf_layers.ReflectionPadding2D(padding=(3, 3))
        self.init_conv = layers.Conv2D(self.init_filters, (7, 7), kernel_initializer=self.kernel_initializer, use_bias=False)
        self.final_conv = layers.Conv2D(3, (7, 7), padding="valid")
        self.inst_norm = solartf_layers.InstanceNormalization(gamma_initializer=self.gamma_initializer)
        self.active_relu = layers.Activation("relu")
        self.active_tanh = layers.Activation("tanh")

        filters = self.init_filters
        self.downsample_blocks = []
        for _ in range(self.num_downsampling_blocks):
            filters *= 2
            self.downsample_blocks.append(solartf_blocks.DownsampleBlock(
                filters=filters,
                activation=self.active_relu))

        self.residual_block = solartf_blocks.ResidualBlock(
            filters=filters,
            activation=self.active_relu)

        self.upsample_blocks = []
        for _ in range(self.num_upsample_blocks):
            filters //= 2
            self.upsample_blocks.append(solartf_blocks.UpsampleBlock(
                filters=filters,
                activation=self.active_relu))

    def call(self, inputs):
        x = self.reflection_padding(inputs)
        x = self.init_conv(x)
        x = self.inst_norm(x)
        x = self.active_relu(x)

        # Downsampling
        for downsample_block in self.downsample_blocks:
            x = downsample_block(x)

        # Residual blocks
        for _ in range(self.num_residual_blocks):
            x = self.residual_block(x)

        # Upsampling
        for upsample_block in self.upsample_blocks:
            x = upsample_block(x)

        # Final block
        x = self.reflection_padding(x)
        x = self.final_conv(x)
        x = self.active_tanh(x)

        return x

    def get_config(self):
        return super(ResnetGenerator, self).get_config()
