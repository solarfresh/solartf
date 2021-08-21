import tensorflow as tf
from typing import List
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                                     Conv2D, Conv2DTranspose, Dense, Dropout, Flatten,
                                     Reshape, ZeroPadding2D)
from .activation import (hard_swish, relu)
from .block import (inverted_res_block, head_block)
from .layer import (AnchorBoxes, GaussianBlur)
from .util import get_filter_nb_by_depth


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
                 forward='up',
                 prefix=None,
                 kernel_initializer='he_normal',
                 l2_regularization=0.0005,
                 use_bias=True):
        super(FPN, self).__init__()
        self.n_filters = n_filters
        self.forward = forward
        self.kernel_initializer = kernel_initializer
        self.l2_reg = l2_regularization
        self.use_bias = use_bias
        if prefix is not None:
            self.prefix = prefix
        elif self.forward == 'up':
            self.prefix = 'fpn_up'
        elif self.forward == 'down':
            self.prefix = 'fpn_down'
        else:
            raise ValueError(f'The argument forward {self.forward} is not implemented...')

    def call(self, inputs, training=None, mask=None):
        n_features = len(inputs)
        feature_map_list = []
        for index in range(n_features):
            if index == 0:
                p = Conv2D(self.n_filters, (1, 1), name=f'{self.prefix}_c{index}_p{index}')(inputs[index])
            else:
                p = Add(name=f'{self.prefix}_p{index}_add')([
                    self._forward_conv(p, index=index),
                    Conv2D(self.n_filters, (1, 1), padding='same', name=f'{self.prefix}_c{index}_p{index}')(inputs[index])
                ])

            feature_map_list.append(Conv2D(self.n_filters, (3, 3), padding="SAME", name=f'{self.prefix}_p_conv_{index}')(p))

        return feature_map_list

    def _forward_conv(self, x, index):
        if self.forward == 'up':
            return Conv2DTranspose(filters=self.n_filters,
                                   kernel_size=(3, 3),
                                   strides=(2, 2),
                                   padding='same',
                                   kernel_initializer=self.kernel_initializer,
                                   # kernel_regularizer=l2(self.l2_reg),
                                   name=f'{self.prefix}_up_p{index}',
                                   use_bias=self.use_bias)(x)
        elif self.forward == 'down':
            return Conv2D(filters=self.n_filters,
                          kernel_size=(3, 3),
                          strides=(2, 2),
                          padding='same',
                          kernel_initializer=self.kernel_initializer,
                          # kernel_regularizer=l2(self.l2_reg),
                          name=f'{self.prefix}_p{index}',
                          use_bias=self.use_bias)(x)
        else:
            raise NotImplementedError(f'forward {self.forward} is not implemented...')

    def get_config(self):
        return super(FPN, self).get_config()


class MobileNetSingleScaleDetectionNeck(tf.keras.Model):
    def __init__(self,
                 l2_regularization=0.0005,
                 momentum=0.99,
                 epsilon=0.00001):
        super(MobileNetSingleScaleDetectionNeck, self).__init__()
        self.l2_reg = l2_regularization
        self.momentum = momentum
        self.epsilon = epsilon

    def call(self, inputs, training=None, mask=None):
        conv_1 = self.conv(inputs, filter_nb=40, name='1')
        conv_2 = self.conv(conv_1, filter_nb=128, name='2')
        conv_3 = self.conv(conv_2, filter_nb=128, name='3')
        conv_4 = self.conv(conv_3, filter_nb=64, name='4')

        return conv_4

    def conv(self,
             inputs,
             filter_nb,
             kernel_initializer='he_normal',
             use_bias=True,
             name=''):

        conv_1 = Conv2D(filter_nb, (1, 1),
                        padding='same',
                        kernel_initializer=kernel_initializer,
                        name=f'ssd_neck_conv_{name}_1',
                        use_bias=use_bias)(inputs)
        conv_1 = BatchNormalization(momentum=self.momentum,
                                    epsilon=self.epsilon,
                                    name=f'ssd_neck_conv/bn_{name}_1')(conv_1)
        conv_1 = Activation(hard_swish, name=f'ssd_neck_relu_{name}_1')(conv_1)
        conv_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name=f'ssd_neck_zero_padding_{name}')(conv_1)

        conv_2 = Conv2D(2 * filter_nb, (3, 3),
                        strides=(2, 2),
                        padding='valid',
                        kernel_initializer=kernel_initializer,
                        name=f'ssd_neck_conv_{name}_2',
                        use_bias=use_bias)(conv_1)
        conv_2 = BatchNormalization(momentum=self.momentum,
                                    epsilon=self.epsilon,
                                    name=f'ssd_neck_conv/bn_{name}_2')(conv_2)
        conv_2 = Activation(hard_swish, name=f'ssd_neck_relu_{name}_2')(conv_2)
        return conv_2

    def get_config(self):
        return super(MobileNetSingleScaleDetectionNeck, self).get_config()


class SSDNeck(tf.keras.Model):
    def __init__(self,
                 neck_filter_nb=64,
                 fpn_filters_nb=64,
                 momentum=0.99,
                 epsilon=0.00001):
        super(SSDNeck, self).__init__()
        self.neck_filter_nb = neck_filter_nb
        self.momentum = momentum
        self.epsilon = epsilon
        self.fpn_up = FPN(forward='up', n_filters=fpn_filters_nb)
        self.fpn_down = FPN(forward='down', n_filters=fpn_filters_nb)

    def call(self, inputs, training=None, mask=None):
        return self.fpn_up.call([inputs[index] for index in range(len(inputs) - 1, -1, -1)])

    def conv(self,
             inputs,
             filter_nb,
             kernel_initializer='he_normal',
             use_bias=True,
             name=''):

        conv_1 = Conv2D(filter_nb, (1, 1),
                        padding='same',
                        kernel_initializer=kernel_initializer,
                        name=f'ssd_neck_conv_{name}_1',
                        use_bias=use_bias)(inputs)
        conv_1 = BatchNormalization(momentum=self.momentum,
                                    epsilon=self.epsilon,
                                    name=f'ssd_neck_conv/bn_{name}_1')(conv_1)
        conv_1 = Activation(hard_swish, name=f'ssd_neck_relu_{name}_1')(conv_1)
        conv_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name=f'ssd_neck_zero_padding_{name}')(conv_1)

        conv_2 = Conv2D(2 * filter_nb, (3, 3),
                        strides=(2, 2),
                        padding='valid',
                        kernel_initializer=kernel_initializer,
                        name=f'ssd_neck_conv_{name}_2',
                        use_bias=use_bias)(conv_1)
        conv_2 = BatchNormalization(momentum=self.momentum,
                                    epsilon=self.epsilon,
                                    name=f'ssd_neck_conv/bn_{name}_2')(conv_2)
        conv_2 = Activation(hard_swish, name=f'ssd_neck_relu_{name}_2')(conv_2)
        return conv_2

    def get_config(self):
        return super(SSDNeck, self).get_config()


class SSDHead(tf.keras.Model):

    def __init__(self,
                 n_classes: int,
                 image_shape,
                 n_boxes: List[int],
                 scales,
                 aspect_ratios,
                 step_shapes,
                 variances,
                 conv_block=None,
                 n_detect_filter=32,
                 offset_shapes=None,
                 l2_regularization=0.0005):
        """
        :param n_classes: number of classes or labels
        :param n_boxes: number of boxes generated from 6 feature layers
        """
        super(SSDHead, self).__init__()

        self.n_classes = n_classes
        self.conv_block = conv_block
        self.n_boxes = n_boxes
        self.n_features = len(n_boxes)

        self.image_shape = image_shape
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        if step_shapes is None:
            self.step_shapes = [None] * self.n_features
        else:
            self.step_shapes = step_shapes

        if offset_shapes is None:
            self.offset_shapes = [None] * self.n_features
        else:
            self.offset_shapes = offset_shapes
        self.variances = variances

        self.n_detect_filter = n_detect_filter
        self.l2_reg = l2_regularization

    def call(self, inputs, training=None, mask=None):
        mbox_conf_list = []
        mbox_loc_list = []
        mbox_priorbox_list = []

        for index in range(self.n_features):
            mbox_conf_list.append(head_block(inputs[index],
                                             filters=self.n_classes * self.n_boxes[index],
                                             reshape=(-1, self.n_classes),
                                             conv_block=self.conv_block,
                                             activation=Activation('softmax'),
                                             name=f'mbox_conf_{index}'))
            mbox_loc_list.append((head_block(inputs[index],
                                             filters=4 * self.n_boxes[index],
                                             reshape=(-1, 4),
                                             conv_block=self.conv_block,
                                             activation=None,
                                             name=f'mbox_loc_{index}')))
            mbox_priorbox_list.append(self.mbox_priorbox_block(inputs[index],
                                                               n_boxes=self.n_boxes[index],
                                                               scale=self.scales[index],
                                                               aspect_ratios=self.aspect_ratios[index],
                                                               step_shape=self.step_shapes[index],
                                                               offset_shape=self.offset_shapes[index],
                                                               variances=self.variances))

        predictions = {'mbox_conf': Concatenate(axis=1, name='mbox_conf')(mbox_conf_list),
                       'mbox_loc': Concatenate(axis=1, name='mbox_loc')(mbox_loc_list),
                       'mbox_priorbox': Concatenate(axis=1, name='mbox_priorbox')(mbox_priorbox_list),
                       'fpn_output': inputs[-1]}

        return predictions

    def mbox_priorbox_block(self, inputs, n_boxes, scale, aspect_ratios, step_shape, offset_shape, variances):
        x = Conv2D(n_boxes, kernel_size=1, padding='same', use_bias=False, trainable=False)(inputs)

        x = AnchorBoxes(image_shape=self.image_shape[:2],
                        scale=scale,
                        aspect_ratios=aspect_ratios,
                        step_shape=step_shape,
                        offset_shape=offset_shape,
                        variances=variances)(x)

        return Reshape((-1, 8))(x)

    def get_config(self):
        return super(SSDHead, self).get_config()
