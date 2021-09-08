import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv2DTranspose, Dense, GlobalAveragePooling2D, Input, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from solartf.core.model import TFModelBase
from solartf.core.loss import MonteCarloEstimateLoss


class CVAE(TFModelBase):
    def __init__(self,
                 input_shape,
                 latent_dim,
                 backbone,
                 n_stage=5,
                 n_decoder_filter_in=32):

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.backbone = backbone
        self.n_stage = n_stage
        self.n_decoder_filter_in = n_decoder_filter_in
        self.decoder = self._build_decoder()

    def data_preprocess(self, inputs, training=True):
        image_input_list = inputs['image_input_list']
        batch_image_input = np.stack([image_input.image_array.copy().astype(np.float32)
                                      for image_input in image_input_list], axis=0)

        batch_decoded_image = np.stack([image_input.image_array.copy().astype(np.float32) / 255.
                                        for image_input in image_input_list], axis=0)

        return batch_image_input, {
            'decoded_image': batch_decoded_image,
            'logpz': np.zeros(shape=(self.latent_dim * 3),),
        }

    def data_postprocess(self, outputs, meta):
        return outputs

    def build_model(self):
        image_input = Input(shape=self.input_shape, name='image_input')
        backbone = self.backbone.call(image_input)
        x = backbone.layers[-1].output
        x = GlobalAveragePooling2D()(x)
        latent = Dense(units=self.latent_dim + self.latent_dim, name='latent')(x)
        mean, logvar = tf.split(latent, num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        output = {
            'decoded_image': self.decoder(z),
            'logpz': tf.concat((z, mean, logvar), name='logpz', axis=-1)
        }
        self.model = Model(image_input, output)
        return self

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None):
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04) \
            if optimizer is None else optimizer

        mc_loss = MonteCarloEstimateLoss()
        loss = {
            'decoded_image': mc_loss.logpx_loss,
            'logpz': mc_loss.logpz_loss
        } if loss is None else loss

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)

        return self

    def _build_decoder(self):
        height, width, depth = self.input_shape
        height = height // 2 ** self.n_stage
        width = width // 2 ** self.n_stage

        latent = Input(shape=(self.latent_dim,))
        x = Dense(units=height * width * self.n_decoder_filter_in, activation=tf.nn.relu)(latent)
        x = Reshape(target_shape=(height, width, self.n_decoder_filter_in))(x)

        for idx in range(self.n_stage):
            n_filters = self.n_decoder_filter_in * 2 * (idx + 1)
            x = Conv2DTranspose(filters=n_filters,
                                kernel_size=3,
                                strides=2,
                                padding='same',
                                activation='relu')(x)

        x = tf.keras.layers.Conv2DTranspose(filters=depth,
                                            kernel_size=3,
                                            strides=1,
                                            padding='same',
                                            name='decoded_image')(x)
        x = tf.sigmoid(x)

        return Model(latent, x)

    @staticmethod
    def reparameterize(mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean
