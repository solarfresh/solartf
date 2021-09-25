import tensorflow as tf
from tensorflow.keras import (losses, optimizers)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Concatenate
from solartf.core.loss import (SSDMultipleLoss, smooth_L1_loss, IoUFamilyLoss)
from solartf.objdetector.layer import DecodeDetections
from solartf.objdetector.pipeline import AnchorDetectPipeline
from solartf.objdetector.config import AnchorDetectConfig
from solartf.objdetector.model import MobileNetV3SmallTFModel


class Config(AnchorDetectConfig):

    # train or freeze or show or partial_freeze or inference
    STATUS = 'show'
    # MODEL_WEIGHT_PATH = '/Users/huangshangyu/Downloads/model/' \
    #                     'epoch-00790_mbox_conf_compute_classification_metric-0.9944_val_mbox_conf_compute_classification_metric-0.9944_mbox_loc_compute_localization_loss-0.7338_val_mbox_loc_compute_localization_loss-0.7263.h5'
    MODEL_WEIGHT_PATH = None

    MODEL_FREEZE_DIR = '/Users/huangshangyu/Downloads/model'
    MODEL_FREEZE_NAME = 'ssd_multiple.pb'

    IN_MEMORY = False

    TRAIN_IMAGE_PATH = '/Users/huangshangyu/Downloads/processed_head_count_dataset/mask_faces_images'
    TRAIN_LABEL_PATH = '/Users/huangshangyu/Projects/ViewSonic/AITeam/Dataset/train'

    VALID_IMAGE_PATH = '/Users/huangshangyu/Downloads/processed_head_count_dataset/mask_faces_images'
    VALID_LABEL_PATH = '/Users/huangshangyu/Projects/ViewSonic/AITeam/Dataset/valid'

    TEST_IMAGE_PATH = '/Users/huangshangyu/Downloads/processed_head_count_dataset/mask_faces_images'
    TEST_LABEL_PATH = '/Users/huangshangyu/Projects/ViewSonic/AITeam/Dataset/train'

    OUTPUT_ENCODER_LOC_TYPE = 'iou'
    OUTPUT_ENCODER_NORMALIZE = False
    OUTPUT_ENCODER_VARIANCES = [.1, .1, .1, .1]

    iou_family_loss = IoUFamilyLoss(coord=OUTPUT_ENCODER_LOC_TYPE)
    ssd_multi_loss = SSDMultipleLoss(mbox_conf_loss=losses.categorical_crossentropy,
                                     mbox_loc_loss=iou_family_loss.tf_diou_loss,
                                     pos_weights=[1.],
                                     neg_pos_ratio=3.,
                                     n_neg_min=0)

    TRAIN_LOSS = {'mbox_conf': ssd_multi_loss.compute_classification_loss,
                  'mbox_loc': ssd_multi_loss.compute_localization_loss,
                  'mbox_priorbox': smooth_L1_loss,}

    TRAIN_BATCH_SIZE = 1
    VALID_BATCH_SIZE = 1
    VALID_STEP_PER_EPOCH = 1

    TRAIN_OPTIMIZER = optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
    TRAIN_LOSS_WEIGHTS = [1., 1., 10., 0]

    IMAGE_SHAPE = (320, 320, 3)

    AUGMENT = False
    FEATURE_MAP_SHAPES = [(20, 20), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]
    ANCHOR_ASPECT_RATIOS = [[(1.2, 1.2), (.8, .8), (1., 2.), (.5, 1.)],
                            [(.8, .8), (.8, 1.2)],
                            [(1.2, 1.2), (.8, .8), (1., 2.), (.5, 1.)],
                            [(.8, .8), (.8, 1.2)],
                            [(.8, .8), (.8, 1.2)],
                            [(.8, 1.2)]]
    ANCHOR_SCALES = [0.096, 0.098, 0.197, 0.310, 0.496, 0.986]
    ANCHOR_STEP_SHAPES = [16, 32, 64, 100, 160, 320]
    ANCHOR_OFFSET_SHAPES = [.5] * 6

    POSITIVE_IOU_THRESHOLD = .01
    NEGATIVE_IOU_THRESHOLD = .0
    BBOX_LABEL_METHOD = 'threshold'

    CLASS_NUMBER = 2
    MODEL = MobileNetV3SmallTFModel(
        input_shape=IMAGE_SHAPE,
        n_classes=CLASS_NUMBER,
        aspect_ratios=ANCHOR_ASPECT_RATIOS,
        feature_map_shapes=FEATURE_MAP_SHAPES,
        scales=ANCHOR_SCALES,
        step_shapes=ANCHOR_STEP_SHAPES,
        offset_shapes=ANCHOR_OFFSET_SHAPES,
        variances=OUTPUT_ENCODER_VARIANCES,
        bbox_encoding=OUTPUT_ENCODER_LOC_TYPE,
        bbox_normalize=OUTPUT_ENCODER_NORMALIZE,
        pos_iou_threshold=POSITIVE_IOU_THRESHOLD,
        neg_iou_threshold=NEGATIVE_IOU_THRESHOLD,
        bbox_labeler_method=BBOX_LABEL_METHOD)

    TRAIN_MODEL_CHECKPOINT_PATH = '/Users/huangshangyu/Downloads/model/ssd_xentropy_diou_epoch-{epoch:05d}' \
                                  + ''.join([f'_{key}-{{{key}:.4f}}_val_{key}-{{val_{key}:.4f}}'
                                             for key in ['loss', 'mbox_conf_loss', 'mbox_loc_loss']]) + '.h5'
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
    trainer = AnchorDetectPipeline(config)

    if config.STATUS == 'inference':
        trainer.inference()

    if config.STATUS == 'train':
        trainer.train()

    if config.STATUS == 'show':
        trainer.load_model()
        trainer.model.model.summary()

    if config.STATUS == 'freeze':
        trainer.load_model()

        mbox_conf = trainer.model.model.get_layer('mbox_conf').output
        mbox_loc = trainer.model.model.get_layer('mbox_loc').output
        mbox_priorbox = trainer.model.model.get_layer('mbox_priorbox').output
        ssd_image_input = trainer.model.model.get_layer('detect_image_input').input

        bboxes = Concatenate(axis=2, name='encoded_bboxes')([mbox_loc, mbox_priorbox])

        detection_layer = DecodeDetections(config.OUTPUT_ENCODER_LOC_TYPE,
                                           normalize=config.OUTPUT_ENCODER_NORMALIZE,
                                           name='ssd_output')
        predictions = detection_layer(mbox_conf, bboxes)

        trainer.model.model = tf.keras.models.Model(ssd_image_input, predictions)
        trainer.model.model.summary()

        trainer.model.freeze_graph(save_dir=config.MODEL_FREEZE_DIR, model_name=config.MODEL_FREEZE_NAME)

    if config.STATUS == 'partial_freeze':
        trainer.load_model()
        for layer in trainer.model.model.layers:
            if 'mbox_loc' in layer.name or 'fpn_output' in layer.name:
                layer.trainable = False

        trainer.model.save_model('/Users/huangshangyu/Downloads/model/ssd_adaptive_loc_freeze.h5')
