from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from solartf.data.image.processor import ImageAugmentation
from solartf.generator.config import CycleGANConfig
from solartf.generator.pipeline import CycleGANPipeline


class Config(CycleGANConfig):
    # train, show
    STATUS = 'show'
    # MODEL_WEIGHT_PATH = ''
    MODEL_WEIGHT_PATH = None

    TRAIN_INITIAL_EPOCH = 0
    TRAIN_EPOCH = TRAIN_INITIAL_EPOCH + 500

    TRAIN_IMAGE_PATH_X = '/Users/huangshangyu/Downloads/experiment/maskimage/Crowd_Human/No_Mask'
    TRAIN_IMAGE_PATH_Y = '/Users/huangshangyu/Downloads/experiment/maskimage/MAFA'
    TRAIN_AUGMENT = [ImageAugmentation(brightness_ratio=None,
                                       flip_orientation='horizontal_random',
                                       scale_ratio=(.8, 1.2),
                                       degree=(-5., 5.),
                                       h_shift=(-10, 10),
                                       v_shift=(-10, 10))]

    VALID_IMAGE_PATH_X = '/Users/huangshangyu/Downloads/experiment/maskimage/Crowd_Human/No_Mask'
    VALID_IMAGE_PATH_Y = '/Users/huangshangyu/Downloads/experiment/maskimage/MAFA'
    VALID_AUGMENT = [ImageAugmentation(brightness_ratio=None,
                                       flip_orientation='horizontal_random',
                                       scale_ratio=(.8, 1.2),
                                       degree=(-5., 5.),
                                       h_shift=(-10, 10),
                                       v_shift=(-10, 10))]

    TRAIN_OPTIMIZER = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

    TRAIN_MODEL_CHECKPOINT_PATH = '/Users/huangshangyu/Downloads/model/cyclegan/epoch-{epoch:05d}' \
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
    trainer = CycleGANPipeline(config)

    if config.STATUS == 'train':
        trainer.train()

    if config.STATUS == 'show':
        trainer.load_model()
        trainer.model.model.summary()
