from tensorflow.keras.layers import Input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from solartf.core.graph import LeNet5
from solartf.core.model import TFModelBase


class TFLeNet5(TFModelBase):
    def __init__(self,
                 input_shape,
                 n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes

        self.lenet5 = LeNet5(self.n_classes, dropout_rate=.5)

    def data_preprocess(self, inputs, training=True):
        return inputs

    def data_postprocess(self, outputs, meta):
        return outputs

    def build_model(self):
        img_height, img_width, _ = self.input_shape
        image_input = Input(shape=self.input_shape, name='lenet5_input')
        classifier = self.lenet5.call(image_input)
        self.model = Model(image_input, classifier)

        return self

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None):
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04) \
            if optimizer is None else optimizer
        loss = categorical_crossentropy if loss is None else loss

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)

        return self
