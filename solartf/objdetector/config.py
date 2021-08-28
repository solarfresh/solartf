from tensorflow.keras import (losses, optimizers)
from tensorflow.keras.callbacks import ModelCheckpoint
from solartf.core.loss import (SSDMultipleLoss, log_loss, IoUFamilyLoss)
# from .model import MobileNetV3TFModel


class AnchorDetectConfig:

    IN_MEMORY = False

    IMAGE_SHAPE = (320, 320, 3)
    IMAGE_TYPE = 'bgr'
    CLASS_NUMBER = 2
    BBOX_EXCLUDE = {'region_level': ['visualable', 'full']}

    TRAIN_IMAGE_PATH = ''
    TRAIN_LABEL_PATH = ''
    TRAIN_SHUFFLE = True
    TRAIN_AUGMENT = []

    VALID_IMAGE_PATH = ''
    VALID_LABEL_PATH = ''
    VALID_SHUFFLE = True
    VALID_AUGMENT = []

    TEST_IMAGE_PATH = ''
    TEST_LABEL_PATH = ''
    TEST_SHUFFLE = False
    TEST_AUGMENT = []

    FEATURE_MAP_SHAPES = [(20, 20), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]
    ANCHOR_ASPECT_RATIOS = [[1. / .5, 1., 1. / 1.5, 1. / 2., 1. / 3.],
                            [1. / .67, 1. / 1.33, 1. / 2.1, 1. / 3.26],
                            [1. / .5, 1., 1. / 1.5, 1. / 2.],
                            [1. / .5, 1., 1. / 1.5, 1. / 2., 1. / 3.],
                            [1. / .5, 1., 1. / 2.],
                            [1.0]]
    ANCHOR_SCALES = [0.096, 0.098, 0.197, 0.310, 0.496, 0.986]
    ANCHOR_STEP_SHAPES = [16, 32, 64, 100, 160, 320]
    ANCHOR_OFFSET_SHAPES = [None] * 6

    POSITIVE_IOU_THRESHOLD = .2
    NEGATIVE_IOU_THRESHOLD = .05
    BBOX_LABEL_METHOD = 'threshold'

    OUTPUT_ENCODER_VARIANCES = [1., 1., 1., 1.]
    OUTPUT_ENCODER_LOC_TYPE = 'corner'
    OUTPUT_ENCODER_NORMALIZE = False

    # MODEL = MobileNetV3TFModel(input_shape=IMAGE_SHAPE,
    #                            n_classes=CLASS_NUMBER,
    #                            aspect_ratios=ANCHOR_ASPECT_RATIOS,
    #                            feature_map_shapes=FEATURE_MAP_SHAPES,
    #                            scales=ANCHOR_SCALES,
    #                            step_shapes=ANCHOR_STEP_SHAPES,
    #                            offset_shapes=ANCHOR_OFFSET_SHAPES,
    #                            variances=OUTPUT_ENCODER_VARIANCES,
    #                            bbox_encoding=OUTPUT_ENCODER_LOC_TYPE,
    #                            bbox_normalize=OUTPUT_ENCODER_NORMALIZE,
    #                            pos_iou_threshold=POSITIVE_IOU_THRESHOLD,
    #                            neg_iou_threshold=NEGATIVE_IOU_THRESHOLD,
    #                            bbox_labeler_method=BBOX_LABEL_METHOD)

    iou_family_loss = IoUFamilyLoss(coord=OUTPUT_ENCODER_LOC_TYPE)
    ssd_multi_loss = SSDMultipleLoss(mbox_conf_loss=log_loss,
                                     mbox_loc_loss=iou_family_loss.tf_diou_loss,
                                     pos_weights=[1.],
                                     neg_pos_ratio=3.,
                                     n_neg_min=0)

    TRAIN_BATCH_SIZE = 32
    TRAIN_EPOCH = 20000
    TRAIN_VERBOSE = 1
    TRAIN_OPTIMIZER = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

    TRAIN_LOSS = {'mbox_conf': ssd_multi_loss.compute_classification_loss,
                  'mbox_loc': ssd_multi_loss.compute_localization_loss,
                  'mbox_priorbox': losses.mse}
    TRAIN_LOSS_WEIGHTS = [1., 1., 0]

    TRAIN_MODEL_CHECKPOINT_PATH = '/home/data/models/ssd_xentropy_diou_epoch-{epoch:05d}' \
                                  + ''.join([f'_{key}-{{{key}:.4f}}_val_{key}-{{val_{key}:.4f}}'
                                             for key in ['loss', 'mbox_conf_loss', 'mbox_loc_loss']]) + '.h5'
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
    TRAIN_CLASS_WEIGHT = None
    TRAIN_INITIAL_EPOCH = 0
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
