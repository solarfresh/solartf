from __future__ import division
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (InputSpec, Layer)


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
