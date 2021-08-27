import os
import tensorflow as tf
from tensorflow.keras import (losses, optimizers)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Concatenate
from solartf.kptdetector.pipeline import KeypointDetectPipeline
from solartf.kptdetector.config import ResNetV2Config
from solartf.kptdetector.model import TFResNet
from solartf.kptdetector.processor import KeypointAugmentation


class Config(ResNetV2Config):

    # train or freeze or show or partial_freeze or inference
    STATUS = 'train'
    # MODEL_WEIGHT_PATH = '/Users/huangshangyu/Downloads/model/' \
    #                     'epoch-00790_mbox_conf_compute_classification_metric-0.9944_val_mbox_conf_compute_classification_metric-0.9944_mbox_loc_compute_localization_loss-0.7338_val_mbox_loc_compute_localization_loss-0.7263.h5'
    MODEL_WEIGHT_PATH = None

    MODEL_FREEZE_DIR = '/Users/huangshangyu/Downloads/model'
    MODEL_FREEZE_NAME = 'kptdetector.pb'

    IN_MEMORY = False

    DATA_ROOT = '/Users/huangshangyu/Downloads/experiment/facekpt'
    TRAIN_IMAGE_PATH = os.path.join(DATA_ROOT, 'train', 'images')
    TRAIN_LABEL_PATH = os.path.join(DATA_ROOT, 'train', 'annotations')

    VALID_IMAGE_PATH = os.path.join(DATA_ROOT, 'valid', 'images')
    VALID_LABEL_PATH = os.path.join(DATA_ROOT, 'valid', 'annotations')

    TEST_IMAGE_PATH = os.path.join(DATA_ROOT, 'test', 'images')
    TEST_LABEL_PATH = os.path.join(DATA_ROOT, 'test', 'annotations')

    TRAIN_LOSS = {'cls': losses.binary_crossentropy,
                  'kpt': losses.mse}
    TRAIN_LOSS_WEIGHTS = [1., 1.]

    TRAIN_BATCH_SIZE = 32
    TRAIN_STEP_PER_EPOCH = 10
    TRAIN_AUGMENT = [KeypointAugmentation(brightness_ratio=(.5, 2.),
                                          flip_orientation='horizontal_random',
                                          scale_ratio=(.8, 1.2),
                                          degree=(-5., 5.),
                                          h_shift=(-10, 10),
                                          v_shift=(-10, 10))]

    VALID_BATCH_SIZE = 32
    VALID_STEP_PER_EPOCH = 1
    VALID_AUGMENT = [KeypointAugmentation(brightness_ratio=(.5, 2.),
                                          flip_orientation='horizontal_random',
                                          scale_ratio=(.8, 1.2),
                                          degree=(-5., 5.),
                                          h_shift=(-10, 10),
                                          v_shift=(-10, 10))]

    TRAIN_OPTIMIZER = optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

    IMAGE_SHAPE = (64, 64, 3)
    CLASS_NUMBER = 4
    MODEL = TFResNet(input_shape=IMAGE_SHAPE,
                     n_classes=CLASS_NUMBER,
                     num_res_blocks=3,
                     num_filters_in=16,
                     dropout_rate=.3)

    TRAIN_MODEL_CHECKPOINT_PATH = '/Users/huangshangyu/Downloads/model/kptdetector-{epoch:05d}' \
                                  + ''.join([f'_{key}-{{{key}:.4f}}_val_{key}-{{val_{key}:.4f}}'
                                             for key in ['loss', 'cls_output_loss', 'kpt_output_loss']]) + '.h5'
    TRAIN_CALLBACKS = [
        ModelCheckpoint(filepath=TRAIN_MODEL_CHECKPOINT_PATH,
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=False,
                        save_weights_only=False,
                        mode='auto',
                        period=10),
    ]


if __name__ == '__main__':
    config = Config()
    trainer = KeypointDetectPipeline(config)

    if config.STATUS == 'inference':
        trainer.inference()

    if config.STATUS == 'train':
        trainer.train()

    if config.STATUS == 'show':
        trainer.load_model()
        trainer.model.model.summary()

    if config.STATUS == 'freeze':
        trainer.load_model()
        #
        # trainer.model.model = tf.keras.models.Model(ssd_image_input, predictions)
        # trainer.model.model.summary()
        #
        # trainer.model.freeze_graph(save_dir=config.MODEL_FREEZE_DIR, model_name=config.MODEL_FREEZE_NAME)

    if config.STATUS == 'partial_freeze':
        trainer.load_model()
        for layer in trainer.model.model.layers:
            if 'mbox_loc' in layer.name or 'fpn_output' in layer.name:
                layer.trainable = False

        trainer.model.save_model('/Users/huangshangyu/Downloads/model/ssd_adaptive_loc_freeze.h5')
