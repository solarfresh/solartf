from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from .loss import CycleGANLoss


class CycleGANConfig:
    IMAGE_SHAPE_X = (226, 226, 3)
    IMAGE_SHAPE_Y = (226, 226, 3)
    IMAGE_TYPE_X = 'bgr'
    IMAGE_TYPE_Y = 'bgr'

    TRAIN_IMAGE_PATH_X = ''
    TRAIN_IMAGE_PATH_Y = ''
    TRAIN_AUGMENT = None
    TRAIN_SHUFFLE = True

    VALID_IMAGE_PATH_X = ''
    VALID_IMAGE_PATH_Y = ''
    VALID_AUGMENT = None
    VALID_SHUFFLE = True

    TEST_IMAGE_PATH_X = ''
    TEST_IMAGE_PATH_Y = ''
    TEST_SHUFFLE = False

    TRAIN_OPTIMIZER = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

    cycle_gan_loss = CycleGANLoss()
    TRAIN_LOSS = {
        'gen_x': cycle_gan_loss.gen_loss,
        'gen_y': cycle_gan_loss.gen_loss,
        'disc_x': cycle_gan_loss.disc_loss,
        'disc_y': cycle_gan_loss.disc_loss
    }
    TRAIN_LOSS_WEIGHTS = None

    # TRAIN_MODEL_CHECKPOINT_PATH = '/home/data/models/ssd_xentropy_diou_epoch-{epoch:05d}' \
    #                               + ''.join([f'_{key}-{{{key}:.4f}}_val_{key}-{{val_{key}:.4f}}'
    #                                          for key in ['loss', 'mbox_conf_loss', 'mbox_loc_loss']]) + '.h5'
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

    TRAIN_METRIC = None

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

    MODEL_WEIGHT_PATH = None
    MODEL_CUSTOM_OBJECTS = None

    MAX_QUEUE_SIZE = 20
    USE_MULTIPROCESSING = None
    WORKER_NUMBER = 1

