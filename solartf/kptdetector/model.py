import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.losses import (binary_crossentropy, mse)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from solartf.core import graph
from solartf.core.model import TFModelBase


class TFKeypointNet(TFModelBase):
    def __init__(self,
                 input_shape,
                 n_classes,
                 backbone=None,
                 dropout_rate=.3):
        self.input_shape = input_shape
        self.n_classes = n_classes

        self.backbone = graph.ResNetV2(
            num_res_blocks=3,
            num_stage=3,
            num_filters_in=16,
        ) if backbone is None else backbone
        self.dropout_rate = dropout_rate

    def data_preprocess(self, inputs, training=True):
        image_input_list = inputs['image_input_list']
        kpt_input_list = inputs['kpt_input_list']

        batch_image_input = np.stack([image_input.image_array
                                      for image_input in image_input_list], axis=0)
        batch_cls_output = np.stack([kpt_input.labels for kpt_input in kpt_input_list], axis=0)
        batch_kpt_output = np.stack([kpt_input.points_tensor for kpt_input in kpt_input_list], axis=0).astype(np.float)
        height, width, _ = self.input_shape
        batch_kpt_output[..., 0] = batch_kpt_output[..., 0] / width
        batch_kpt_output[..., 1] = batch_kpt_output[..., 1] / height

        return batch_image_input, {'cls': batch_cls_output, 'kpt': batch_kpt_output}

    def data_postprocess(self, outputs, meta):
        return outputs

    def build_model(self):
        img_height, img_width, _ = self.input_shape
        image_input = layers.Input(shape=self.input_shape, name='image_input')
        x = self.backbone(image_input)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.GlobalAveragePooling2D()(x)
        cls = layers.Dense(units=self.n_classes, activation='sigmoid', name='cls_output')(x)
        x = layers.Dense(units=512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        kpt = layers.Dense(units=self.n_classes * 2, activation='sigmoid')(x)
        kpt = layers.Reshape(target_shape=(-1, 2), name='kpt_output')(kpt)

        predictions = {'cls': cls, 'kpt': kpt}
        self.model = Model(image_input, predictions)

        return self

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None):
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04) \
            if optimizer is None else optimizer
        loss = {
            'cls': binary_crossentropy,
            'kpt': mse
        } if loss is None else loss

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)

        return self
