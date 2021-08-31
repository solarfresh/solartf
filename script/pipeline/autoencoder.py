import os
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from solartf.data.image.processor import ImageAugmentation
from solartf.core.graph import ResNetV2
from solartf.autoencoder.config import CVAEConfig
from solartf.autoencoder.model import CVAE
from solartf.autoencoder.pipeline import CVAEPipeline


class Config(CVAEConfig):
    STATUS = 'train'
    MODEL_WEIGHT_PATH = '/Users/huangshangyu/Downloads/model/autoencoder/' \
                        'epoch-00500_loss-1310.8757_val_loss-1241.8926.h5'
    # MODEL_WEIGHT_PATH = None

    IMAGE_SHAPE = (32, 32, 3)
    LATENT_DIM = 8

    TRAIN_INITIAL_EPOCH = 500
    TRAIN_EPOCH = TRAIN_INITIAL_EPOCH + 500
    TRAIN_OPTIMIZER = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

    TRAIN_BATCH_SIZE = 32
    TRAIN_STEP_PER_EPOCH = 10
    TRAIN_AUGMENT = [ImageAugmentation(brightness_ratio=(.5, 2.),
                                       flip_orientation='horizontal_random',
                                       scale_ratio=(.8, 1.2),
                                       degree=(-5., 5.),
                                       h_shift=(-10, 10),
                                       v_shift=(-10, 10))]

    VALID_BATCH_SIZE = 32
    VALID_STEP_PER_EPOCH = 1
    VALID_AUGMENT = [ImageAugmentation(brightness_ratio=(.5, 2.),
                                       flip_orientation='horizontal_random',
                                       scale_ratio=(.8, 1.2),
                                       degree=(-5., 5.),
                                       h_shift=(-10, 10),
                                       v_shift=(-10, 10))]

    DATA_ROOT = '/Users/huangshangyu/Downloads/experiment/maskimage/tmp/'
    TRAIN_IMAGE_PATH = os.path.join(DATA_ROOT, 'train')
    VALID_IMAGE_PATH = os.path.join(DATA_ROOT, 'valid')
    TEST_IMAGE_PATH = os.path.join(DATA_ROOT, 'test')

    MODEL = CVAE(input_shape=IMAGE_SHAPE,
                 latent_dim=LATENT_DIM,
                 backbone=ResNetV2(num_res_blocks=3,
                                   num_stage=3,
                                   num_filters_in=16),
                 n_stage=3,
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
