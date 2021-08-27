import cv2
import os
import tensorflow as tf
from tensorflow.keras import (losses, optimizers)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Concatenate
from solartf.kptdetector.pipeline import KeypointDetectPipeline
from solartf.kptdetector.config import ResNetV2Config
from solartf.kptdetector.model import TFResNet
from solartf.kptdetector.processor import KeypointAugmentation
from solartf.core.loss import smooth_L1_loss


class Config(ResNetV2Config):

    # train or freeze or show or partial_freeze or inference
    STATUS = 'inference'
    MODEL_WEIGHT_PATH = '/Users/huangshangyu/Downloads/model/kptdetector/' \
                        'kptdetector-00060_loss-0.4823_val_loss-0.4841_cls_output_loss-0.0236_val_cls_output_loss-0.0258_kpt_output_loss-0.0197_val_kpt_output_loss-0.0207.h5'
    # MODEL_WEIGHT_PATH = None
    TRAIN_INITIAL_EPOCH = 0
    TRAIN_EPOCH = 200

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
                  'kpt': smooth_L1_loss}
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

    TRAIN_OPTIMIZER = optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

    IMAGE_SHAPE = (64, 64, 3)
    CLASS_NUMBER = 4
    MODEL = TFResNet(input_shape=IMAGE_SHAPE,
                     n_classes=CLASS_NUMBER,
                     num_res_blocks=3,
                     num_filters_in=16,
                     dropout_rate=.3)

    TRAIN_MODEL_CHECKPOINT_PATH = '/Users/huangshangyu/Downloads/model/kptdetector/kptdetector-{epoch:05d}' \
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
        results, image_arrays = trainer.inference()
        for result, image_array in zip(results, image_arrays):
            for idx in range(config.CLASS_NUMBER):
                kpt = (result[f'cls_{idx}_kptx_pred'], result[f'cls_{idx}_kpty_pred'])
                image_array = cv2.circle(image_array, kpt, radius=0, color=(0, 0, 255), thickness=10)
                cv2.imshow('Prediction', image_array)

            key = cv2.waitKey(5000)
            if key == ord('q') or key == 27:
                break

    if config.STATUS == 'train':
        trainer.train()

    if config.STATUS == 'show':
        trainer.load_model()
        trainer.model.model.summary()

    if config.STATUS == 'freeze':
        trainer.load_model()

        cls_output = trainer.model.model.get_layer('cls_output').output
        kpt_output = trainer.model.model.get_layer('kpt_output').output
        image_input = trainer.model.model.get_layer('image_input').input
        kpt_detect = Concatenate(axis=2, name='kpt_detect')([cls_output, kpt_output])

        trainer.model.model = tf.keras.models.Model(image_input, kpt_detect)
        trainer.model.model.summary()

        trainer.model.freeze_graph(save_dir=config.MODEL_FREEZE_DIR, model_name=config.MODEL_FREEZE_NAME)

    if config.STATUS == 'partial_freeze':
        trainer.load_model()
        for layer in trainer.model.model.layers:
            if 'mbox_loc' in layer.name or 'fpn_output' in layer.name:
                layer.trainable = False

        trainer.model.save_model('/Users/huangshangyu/Downloads/model/ssd_adaptive_loc_freeze.h5')
