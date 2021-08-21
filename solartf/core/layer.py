"""
A custom Keras layer to generate anchor boxes.

Copyright (C) 2017 Pierluigi Ferrari

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
# import__init__.py

from __future__ import division
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (InputSpec, Layer)
from .data.label.bbox import NumpyBBoxes
from .model.output import BBoxOutput
from .model.anchor import GridAnchorGenerator


class AnchorBoxes(Layer):
    """
    A Keras layer to create an output tensor containing anchor box coordinates
    and variances based on the input tensor and the passed arguments.

    A set of 2D anchor boxes of different aspect ratios is created for each spatial unit of
    the input tensor. The number of anchor boxes created per unit depends on the arguments
    `aspect_ratios` and `two_boxes_for_ar1`, in the default case it is 4. The boxes
    are parameterized by the coordinate tuple `(xmin, xmax, ymin, ymax)`.

    The logic implemented by this layer is identical to the logic in the module
    `ssd_box_encode_decode_utils.py`.

    The purpose of having this layer in the network is to make the model self-sufficient
    at inference time. Since the model is predicting offsets to the anchor boxes
    (rather than predicting absolute box coordinates directly), one needs to know the anchor
    box coordinates in order to construct the final prediction boxes from the predicted offsets.
    If the model's output tensor did not contain the anchor box coordinates, the necessary
    information to convert the predicted offsets back to absolute coordinates would be missing
    in the model output. The reason why it is necessary to predict offsets to the anchor boxes
    rather than to predict absolute box coordinates directly is explained in `README.md`.

    Input shape:
        4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
        or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.

    Output shape:
        5D tensor of shape `(batch, height, width, n_boxes, 8)`. The last axis contains
        the four anchor box coordinates and the four variance values for each box.
    """

    def __init__(self,
                 image_shape,
                 scale,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 step_shape=None,
                 offset_shape=None,
                 variances=[1.0, 1.0, 1.0, 1.0],
                 **kwargs):

        if K.backend() != 'tensorflow':
            raise TypeError("This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        self.anchor_generator = GridAnchorGenerator()

        self.image_shape = image_shape
        self.aspect_ratios = aspect_ratios
        self.scale = scale
        self.step_shape = step_shape
        self.offset_shape = offset_shape
        self.variances = variances
        self.n_boxes = len(aspect_ratios)
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, *args, **kwargs):

        if K.image_data_format() == 'channels_first':
            # Not yet relevant since TensorFlow is the only supported backend right now,
            # but it can't harm to have this in here for the future
            batch_size, feature_map_channels, feature_map_height, feature_map_width = x.shape
        else:
            batch_size, feature_map_height, feature_map_width, feature_map_channels = x.shape

        # The shape is `(feature_map_height, feature_map_width, n_boxes, 4)` with the centroid presentation
        anchor_tensor = self.anchor_generator.generate_anchors(
            image_shape=self.image_shape,
            feature_map_shape=(feature_map_height, feature_map_width),
            aspect_ratios=self.aspect_ratios,
            scale=self.scale,
            step_shape=self.step_shape,
            offset_shape=self.offset_shape
        )

        bboxes = NumpyBBoxes(anchor_tensor, coord='centroid', method='numpy')
        anchor_tensor = bboxes.to_array(coord='corner')

        # Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
        # as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(anchor_tensor) # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor += self.variances # Long live broadcasting
        # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
        anchor_tensor = np.concatenate((anchor_tensor, variances_tensor), axis=-1)

        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
        anchor_tensor = np.expand_dims(anchor_tensor, axis=0)
        anchor_tensor = K.tile(K.constant(anchor_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return anchor_tensor

    def compute_output_shape(self, input_shape):
        if K.image_data_format() == 'channels_first':
            # Not yet relevant since TensorFlow is the only supported backend right now,
            # but it can't harm to have this in here for the future
            batch_size, feature_map_channels, feature_map_height, feature_map_width = input_shape
        else:
            batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape

        return batch_size, feature_map_height, feature_map_width, self.n_boxes, 8

    def get_config(self):
        config = {
            'image_shape': self.image_shape,
            'scale': self.scale,
            'aspect_ratios': list(self.aspect_ratios),
            'variances': list(self.variances),
        }
        base_config = super(AnchorBoxes, self).get_config()
        base_config.update(config)
        return base_config


class DecodeDetections(Layer):

    def __init__(self,
                 bbox_encoding,
                 normalize=False,
                 **kwargs):
        super(DecodeDetections, self).__init__(**kwargs)
        self.bbox_encoding = bbox_encoding
        self.normalize = normalize
        self.bbox_output = BBoxOutput(method=self.bbox_encoding, normalize=normalize, package='tensorflow')

    def call(self, cls, bbox):
        decoded_bbox = self.bbox_output.decoder(bbox).to_array(coord='corner')
        return tf.concat((cls, decoded_bbox), axis=-1)


class GaussianBlur(Layer):
    def __init__(self, kernel_size: int, mu: float, sigma: float, **kwargs):
        super(GaussianBlur, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.mu = mu
        self.sigma = sigma

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        # self.mu = self.add_weight('mu',
        #                           shape=(1,),
        #                           initializer='random_normal',
        #                           trainable=True)
        # self.sigma = self.add_weight('sigma',
        #                              shape=(1,),
        #                              initializer='random_normal',
        #                              trainable=True)
        super(GaussianBlur, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        if K.image_data_format() == 'channels_first':
            # Not yet relevant since TensorFlow is the only supported backend right now,
            # but it can't harm to have this in here for the future
            batch_size, depth, height, width = x.shape
        else:
            batch_size, height, width, depth = x.shape

        return tf.nn.depthwise_conv2d(x, self.gaussian_kernel(depth), strides=[1, 1, 1, 1], padding='SAME')

    def gaussian_kernel(self, depth):
        x = tf.range(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1, dtype=tf.float32)
        g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(self.sigma, 2))))
        g_norm2d = tf.pow(tf.reduce_sum(g), 2)
        g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d

        return tf.tile(g_kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, depth, 1])

    def compute_output_shape(self, input_shape):
        if K.image_data_format() == 'channels_first':
            # Not yet relevant since TensorFlow is the only supported backend right now,
            # but it can't harm to have this in here for the future
            batch_size, depth, height, width = input_shape
        else:
            batch_size, height, width, depth = input_shape

        return batch_size, height, width, depth

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'mu': self.mu,
            'sigma': self.sigma
        }
        base_config = super(GaussianBlur, self).get_config()
        base_config.update(config)
        return base_config
