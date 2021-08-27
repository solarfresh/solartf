from tensorflow.keras import (losses, optimizers)
from tensorflow.keras.callbacks import ModelCheckpoint
from .model import TFLeNet5


class LeNet5Config:
    IMAGE_SHAPE = (32, 32, 3)
    IMAGE_TYPE = 'bgr'
    CLASS_NUMBER = 2

    TRAIN_IMAGE_PATH = ''
    TRAIN_LABEL_PATH = ''
    TRAIN_AUGMENT = None
    TRAIN_SHUFFLE = True

    VALID_IMAGE_PATH = ''
    VALID_LABEL_PATH = ''
    VALID_AUGMENT = None
    VALID_SHUFFLE = True

    TEST_IMAGE_PATH = ''
    TEST_LABEL_PATH = ''
    TEST_SHUFFLE = False

    TRAIN_OPTIMIZER = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

    TRAIN_LOSS = [losses.categorical_crossentropy]

    # TRAIN_MODEL_CHECKPOINT_PATH = '/home/data/models/lenet5/epoch-{epoch:05d}' \
    #                               + ''.join([f'_{key}-{{{key}:.4f}}_val_{key}-{{val_{key}:.4f}}'
    #                                          for key in ['loss', 'accuracy']]) + '.h5'
    TRAIN_MODEL_CHECKPOINT_PATH = ''
    TRAIN_CALLBACKS = [
        ModelCheckpoint(filepath=TRAIN_MODEL_CHECKPOINT_PATH,
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=False,
                        mode='auto',
                        period=1),
    ]

    TRAIN_INITIAL_EPOCH = 0
    TRAIN_EPOCH = 20000
    TRAIN_VERBOSE = 1
    TRAIN_METRIC = ['accuracy']
    TRAIN_CLASS_WEIGHT = None

    TRAIN_BATCH_SIZE = 32
    TRAIN_STEP_PER_EPOCH = 20

    VALID_BATCH_SIZE = 32
    VALID_STEP_PER_EPOCH = 20
    VALID_FREQUENCY = 1

    TEST_BATCH_SIZE = 1

    MODEL = TFLeNet5(input_shape=IMAGE_SHAPE,
                     n_classes=CLASS_NUMBER)
    MODEL_WEIGHT_PATH = None
    MODEL_CUSTOM_OBJECTS = None
    TRAIN_LOSS_WEIGHTS = None

    MAX_QUEUE_SIZE = 20
    USE_MULTIPROCESSING = None
    WORKER_NUMBER = 1
