import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input,)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from solartf.core.graph import (ResnetGenerator, Discriminator)
from solartf.core.layer import IntensityNormalization
from solartf.core.model import TFModelBase

from .loss import CycleGANLoss


class CycleGan(TFModelBase):
    def __init__(self,
                 input_shape_x,
                 input_shape_y,
                 generator_G=None,
                 generator_F=None,
                 discriminator_X=None,
                 discriminator_Y=None):

        self.input_shape_x = input_shape_x
        self.input_shape_y = input_shape_y

        self.gen_G = generator_G if generator_G is not None else ResnetGenerator(prefix='generator_G')
        self.gen_F = generator_F if generator_F is not None else ResnetGenerator(prefix='generator_F')
        self.disc_X = discriminator_X if discriminator_X is not None else Discriminator(prefix='discriminator_X')
        self.disc_Y = discriminator_Y if discriminator_Y is not None else Discriminator(prefix='discriminator_Y')

        self.gen_x_shape = None
        self.gen_y_shape = None
        self.disc_x_shape = None
        self.disc_y_shape = None

    def data_preprocess(self, inputs, training=True):
        real_x_list = inputs['real_x']
        real_y_list = inputs['real_y']

        batch_real_x = np.stack([image_input.image_array.copy().astype(np.float32)
                                 for image_input in real_x_list], axis=0)
        batch_real_y = np.stack([image_input.image_array.copy().astype(np.float32)
                                 for image_input in real_y_list], axis=0)

        inputs = {
            'real_x': batch_real_x,
            'real_y': batch_real_y
        }

        outputs = {
            'gen_x': np.zeros(shape=self.gen_x_shape),
            'gen_y': np.zeros(shape=self.gen_y_shape),
            'disc_x': np.zeros(shape=self.disc_x_shape),
            'disc_y': np.zeros(shape=self.disc_y_shape)
        }

        return inputs, outputs

    def data_postprocess(self, outputs, meta):
        return outputs

    def build_model(self):
        real_x = Input(shape=self.input_shape_x, name='real_x')
        real_y = Input(shape=self.input_shape_y, name='real_y')
        # real_x = IntensityNormalization()(real_x)
        # real_y = IntensityNormalization()(real_y)

        fake_y = self.gen_G(real_x)
        fake_x = self.gen_F(real_y)

        cycled_x = self.gen_F(fake_y)
        cycled_y = self.gen_G(fake_x)

        same_x = self.gen_F(real_x)
        same_y = self.gen_G(real_y)

        disc_real_x = self.disc_X(real_x)
        disc_fake_x = self.disc_X(fake_x)

        disc_real_y = self.disc_Y(real_y)
        disc_fake_y = self.disc_Y(fake_y)

        gen_x = tf.concat((real_x, cycled_x, same_x), axis=-1)
        gen_y = tf.concat((real_y, cycled_y, same_y), axis=-1)
        disc_x = tf.concat((disc_real_x, disc_fake_x), axis=-1)
        disc_y = tf.concat((disc_real_y, disc_fake_y), axis=-1)

        self.gen_x_shape = gen_x.shape
        self.gen_y_shape = gen_y.shape
        self.disc_x_shape = disc_x.shape
        self.disc_y_shape = disc_y.shape

        inputs = {
            'real_x': real_x,
            'real_y': real_y
        }

        outputs = {
            'gen_x': gen_x,
            'gen_y': gen_y,
            'disc_x': disc_x,
            'disc_y': disc_y
        }

        self.model = Model(inputs, outputs)

        return self

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None):
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04) \
            if optimizer is None else optimizer

        cycle_gan_loss = CycleGANLoss()
        loss = {
            'gen_x': cycle_gan_loss.gen_loss,
            'gen_y': cycle_gan_loss.gen_loss,
            'disc_x': cycle_gan_loss.disc_loss,
            'disc_y':cycle_gan_loss.disc_loss
        } if loss is None else loss

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)

        return self
