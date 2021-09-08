import os

import cv2
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from solartf.data.image.processor import ImageAugmentation
from solartf.core.graph import ResNetV2
from solartf.autoencoder.config import CVAEConfig
from solartf.autoencoder.model import CVAE
from solartf.autoencoder.pipeline import CVAEPipeline


class Config(CVAEConfig):
    # inference, train, show
    STATUS = 'inference'
    IMAGE_TYPE = 'gray'
    MODEL_WEIGHT_PATH = '/Users/huangshangyu/Downloads/model/autoencoder/' \
                        'epoch-00020_loss-106.3888_val_loss-105.9110.h5'
    SAVE_DIR = '/Users/huangshangyu/Downloads/mnist_png-master/mnist_png/tmp'
    # MODEL_WEIGHT_PATH = None

    IMAGE_SHAPE = (28, 28, 1)
    LATENT_DIM = 2

    TRAIN_INITIAL_EPOCH = 0
    TRAIN_EPOCH = TRAIN_INITIAL_EPOCH + 500
    TRAIN_OPTIMIZER = optimizers.Adam(learning_rate=0.0001)

    TRAIN_BATCH_SIZE = 32
    TRAIN_STEP_PER_EPOCH = 1000
    # TRAIN_AUGMENT = [ImageAugmentation(brightness_ratio=None,
    #                                    flip_orientation='horizontal_random',
    #                                    scale_ratio=(.8, 1.2),
    #                                    degree=(-5., 5.),
    #                                    h_shift=(-10, 10),
    #                                    v_shift=(-10, 10))]
    TRAIN_AUGMENT = []

    VALID_BATCH_SIZE = 32
    VALID_STEP_PER_EPOCH = 100
    # VALID_AUGMENT = [ImageAugmentation(brightness_ratio=None,
    #                                    flip_orientation='horizontal_random',
    #                                    scale_ratio=(.8, 1.2),
    #                                    degree=(-5., 5.),
    #                                    h_shift=(-10, 10),
    #                                    v_shift=(-10, 10))]
    VALID_AUGMENT = []

    # DATA_ROOT = '/Users/huangshangyu/Downloads/experiment/maskimage/tmp/'
    DATA_ROOT = '/Users/huangshangyu/Downloads/mnist_png-master/mnist_png'
    TRAIN_IMAGE_PATH = os.path.join(DATA_ROOT, 'train')
    VALID_IMAGE_PATH = os.path.join(DATA_ROOT, 'test')
    TEST_IMAGE_PATH = os.path.join(DATA_ROOT, 'train')

    MODEL = CVAE(input_shape=IMAGE_SHAPE,
                 latent_dim=LATENT_DIM,
                 backbone=ResNetV2(num_res_blocks=2,
                                   num_stage=2,
                                   num_filters_in=16),
                 n_stage=2,
                 n_decoder_filter_in=16)

    TEST_BATCH_SIZE = 1

    TRAIN_MODEL_CHECKPOINT_PATH = '/Users/huangshangyu/Downloads/model/autoencoder/epoch-{epoch:05d}' \
                                  + ''.join([f'_{key}-{{{key}:.4f}}_val_{key}-{{val_{key}:.4f}}'
                                             for key in ['loss']]) + '.h5'

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
    trainer = CVAEPipeline(config)

    if config.STATUS == 'train':
        trainer.train()

    if config.STATUS == 'show':
        trainer.load_model()
        trainer.model.model.summary()

    if config.STATUS == 'inference':
        for fname, image_array in trainer.inference():
            cv2.imwrite(os.path.join(config.SAVE_DIR, fname), image_array)
