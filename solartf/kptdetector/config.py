from tensorflow.keras import (losses, optimizers)
from tensorflow.keras.callbacks import ModelCheckpoint
from solartf.core import graph
from .model import TFKeypointNet


class ResNetV2Config:
    IMAGE_SHAPE = (64, 64, 3)
    IMAGE_TYPE = 'bgr'
    CLASS_NUMBER = [4]
    CLASS_ACTIVATION = ['sigmoid']

    TRAIN_IMAGE_PATH = ''
    TRAIN_AUGMENT = None
    TRAIN_SHUFFLE = True

    VALID_IMAGE_PATH = ''
    VALID_AUGMENT = None
    VALID_SHUFFLE = True

    TEST_IMAGE_PATH = ''
    TEST_SHUFFLE = False

    TRAIN_OPTIMIZER = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

    TRAIN_LOSS = {'cls': losses.binary_crossentropy,
                  'kpt': losses.mse}
    TRAIN_LOSS_WEIGHTS = [1., 1.]
    TRAIN_METRIC = None

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
    TRAIN_CLASS_WEIGHT = None

    TRAIN_BATCH_SIZE = 32
    TRAIN_STEP_PER_EPOCH = 20

    VALID_BATCH_SIZE = 32
    VALID_STEP_PER_EPOCH = 20
    VALID_FREQUENCY = 1

    TEST_BATCH_SIZE = 1

    MODEL = TFKeypointNet(
        input_shape=IMAGE_SHAPE,
        n_classes=CLASS_NUMBER,
        cls_activations=CLASS_ACTIVATION,
        backbone=graph.ResNetV2(
            num_res_blocks=3,
            num_stage=3,
            num_filters_in=16,
        ),
        dropout_rate=.3
    )
    CUSTOM_OBJECTS = {
        'ResNetV2': graph.ResNetV2
    }
    MODEL_WEIGHT_PATH = None
    MODEL_CUSTOM_OBJECTS = None

    MAX_QUEUE_SIZE = 20
    USE_MULTIPROCESSING = None
    WORKER_NUMBER = 1
