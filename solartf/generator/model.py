import tensorflow as tf
from tensorflow.keras.layers import (Input,)
from tensorflow.keras.models import Model
from solartf.core.graph import (ResnetGenerator, Discriminator)
from solartf.core.model import TFModelBase


class CycleGan(TFModelBase):
    def __init__(self,
                 input_shape_x,
                 input_shape_y,
                 generator_G=None,
                 generator_F=None,
                 discriminator_X=None,
                 discriminator_Y=None,
                 lambda_cycle=10.0,
                 lambda_identity=0.5,):

        self.input_shape_x = input_shape_x
        self.input_shape_y = input_shape_y

        self.gen_G = generator_G if generator_G is not None else ResnetGenerator(name='generator_G')
        self.gen_F = generator_F if generator_F is not None else ResnetGenerator(name='generator_F')
        self.disc_X = discriminator_X if discriminator_X is not None else Discriminator(name='discriminator_X')
        self.disc_Y = discriminator_Y if discriminator_Y is not None else Discriminator(name='discriminator_Y')
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def data_preprocess(self, inputs, training=True):
        pass

    def data_postprocess(self, outputs, meta):
        return outputs

    def build_model(self):
        real_x = Input(shape=self.input_shape_x, name='real_x')
        real_y = Input(shape=self.input_shape_y, name='real_y')

        fake_y = self.gen_G(real_x)
        fake_x = self.gen_G(real_y)

        cycled_x = self.gen_F(fake_y)
        cycled_y = self.gen_G(fake_x)

        same_x = self.gen_F(real_x)
        same_y = self.gen_G(real_y)

        disc_real_x = self.disc_X(real_x)
        disc_fake_x = self.disc_X(fake_x)

        disc_real_y = self.disc_Y(real_y)
        disc_fake_y = self.disc_Y(fake_y)

        gen_x = tf.concat((cycled_x, same_x), axis=-1)
        gen_y = tf.concat((cycled_y, same_y), axis=-1)
        disc_x = tf.concat((disc_real_x, disc_fake_x), axis=-1)
        disc_y = tf.concat((disc_real_y, disc_fake_y), axis=-1)

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
        return self

