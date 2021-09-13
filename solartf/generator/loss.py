import tensorflow as tf
from tensorflow.keras.losses import (MeanAbsoluteError, MeanSquaredError)


class CycleGANLoss:
    def __init__(self,
                 lambda_cycle=10.0,
                 lambda_identity=0.5,
                 img_sim_loss=None,
                 adv_loss=None):
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.img_sim_loss = MeanAbsoluteError() if img_sim_loss is None else img_sim_loss
        self.adv_loss = MeanSquaredError() if adv_loss is None else adv_loss

    def gen_loss(self, y_true, y_pred):
        real_img, cycled_img, same_img = tf.split(y_pred, num_or_size_splits=3, axis=-1)
        cycle_loss = self.img_sim_loss(real_img, cycled_img) * self.lambda_cycle
        id_loss = self.img_sim_loss(real_img, same_img) * self.lambda_cycle * self.lambda_identity
        return cycle_loss + id_loss

    def disc_loss(self, y_true, y_pred):
        disc_real, disc_fake = tf.split(y_pred, num_or_size_splits=2, axis=-1)
        real_loss = self.adv_loss(tf.ones_like(disc_real), disc_real)
        fake_loss = self.adv_loss(tf.zeros_like(disc_fake), disc_fake)
        gen_loss = self.adv_loss(tf.ones_like(disc_fake), disc_fake)
        return gen_loss + (real_loss + fake_loss) * 0.5

