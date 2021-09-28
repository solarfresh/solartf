import numpy as np
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
                 cls_activations=None,
                 backbone=None,
                 dropout_rate=.3):
        """

        :param input_shape:
        :param n_classes: int, list, or tuple. different labels for each categories
        :param cls_activations:
        :param backbone:
        :param dropout_rate:
        """
        self.input_shape = input_shape
        if isinstance(n_classes, int):
            self.n_classes = [n_classes]
        else:
            self.n_classes = n_classes

        self.n_cls = len(self.n_classes)

        if cls_activations is None:
            self.cls_activations = ['sigmoid']
        else:
            self.cls_activations = cls_activations

        self.backbone = graph.ResNetV2(
            num_res_blocks=3,
            num_stage=3,
            num_filters_in=16,
        ) if backbone is None else backbone
        self.dropout_rate = dropout_rate

    def data_preprocess(self, inputs, training=True):
        image_input_list = inputs['image_input_list']
        kpt_input_list = inputs['kpt_input_list']
        classes_input_list = inputs['classes_input_list']
        batch_size = len(classes_input_list)

        batch_image_input = np.stack([image_input.image_array
                                      for image_input in image_input_list], axis=0)
        batch_kpt_output = np.stack([kpt_input.points_tensor for kpt_input in kpt_input_list], axis=0).astype(np.float)
        height, width, _ = self.input_shape
        batch_kpt_output[..., 0] = batch_kpt_output[..., 0] / width
        batch_kpt_output[..., 1] = batch_kpt_output[..., 1] / height
        batch_output = {'kpt': batch_kpt_output}
        for idx in range(self.n_cls-1):
            n_classes = self.n_classes[idx]
            cls_output = np.zeros(shape=(batch_size, n_classes), dtype=np.float)
            for b_idx, classes in enumerate(classes_input_list):
                cls_output[b_idx, classes[idx]] = 1.
            batch_output.update({
                f'cls_{idx}': cls_output
            })

        batch_output.update({
            f'cls_{self.n_cls-1}': np.stack([kpt_input.labels
                                             for kpt_input in kpt_input_list], axis=0).astype(np.float)
        })

        return batch_image_input, batch_output

    def data_postprocess(self, outputs, meta):
        return outputs

    def build_model(self):
        img_height, img_width, _ = self.input_shape
        image_input = layers.Input(shape=self.input_shape, name='image_input')
        x = self.backbone(image_input)
        if isinstance(x, list):
            x = x[-1]
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(units=512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        kpt = layers.Dense(units=self.n_classes[-1] * 2, activation='sigmoid')(x)
        kpt = layers.Reshape(target_shape=(-1, 2), name='kpt_output')(kpt)
        predictions = {'kpt': kpt}
        for idx in range(self.n_cls):
            n_classes = self.n_classes[idx]
            cls_activation = self.cls_activations[idx]
            cls = layers.Dense(units=n_classes, activation=cls_activation, name=f'cls_{idx}_output')(x)
            predictions.update({f'cls_{idx}': cls})

        self.model = Model(image_input, predictions)

        return self

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None):
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04) \
            if optimizer is None else optimizer
        loss = {
            'cls_0': binary_crossentropy,
            'kpt': mse
        } if loss is None else loss

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)

        return self
