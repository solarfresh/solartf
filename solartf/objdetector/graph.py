import tensorflow as tf
from tensorflow.keras.layers import (Activation, Concatenate, Conv2D, Reshape)
from typing import List
from .block import anchor_detect_head_block
from .layer import AnchorBoxes


class AnchorDetectHead(tf.keras.Model):
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
        super(AnchorDetectHead, self).__init__()

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
            mbox_conf_list.append(anchor_detect_head_block(inputs[index],
                                                           filters=self.n_classes * self.n_boxes[index],
                                                           reshape=(-1, self.n_classes),
                                                           conv_block=self.conv_block,
                                                           activation=Activation('softmax'),
                                                           name=f'mbox_conf_{index}'))
            mbox_loc_list.append((anchor_detect_head_block(inputs[index],
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
        return super(AnchorDetectHead, self).get_config()
