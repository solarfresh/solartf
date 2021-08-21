import os
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from solartf.classifier.model import TFLeNet5
from solartf.classifier.config import LeNet5Config
from solartf.classifier.pipeline import ClassificationPipeline
from solartf.data.image.processor import ImageAugmentation
from solartf.metric.score import ClassificationMetrics
from solartf.metric.plot import ClassificationMetricPlot


class Config(LeNet5Config):
    STATUS = 'train'
    CLASS_NUMBER = 4
    LABEL_NAME = ['cmfd', 'crowd', 'lobby', 'mafa']
    TRAIN_INITIAL_EPOCH = 0
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

    # MODEL_WEIGHT_PATH = None
    DATA_ROOT = '/Users/huangshangyu/Downloads/experiment/maskimage/tmp/'
    TRAIN_IMAGE_PATH = os.path.join(DATA_ROOT, 'train')
    VALID_IMAGE_PATH = os.path.join(DATA_ROOT, 'valid')
    # TEST_IMAGE_PATH = '/Users/huangshangyu/Downloads/classroom/sample_FLOW_AI_0604/maskimage/clsdata/valid'
    TEST_IMAGE_PATH = os.path.join(DATA_ROOT, 'test')

    MODEL = TFLeNet5
    MODEL_WEIGHT_PATH = os.path.join(DATA_ROOT, 'models',
                                     'epoch-02000_loss-0.5718_val_loss-0.2581_accuracy-0.8687_val_accuracy-0.9375.h5')
    # MODEL_WEIGHT_PATH = None

    TEST_BATCH_SIZE = 1

    TRAIN_MODEL_CHECKPOINT_PATH = os.path.join(DATA_ROOT, 'models', 'epoch-{epoch:05d}'
                                               + ''.join([f'_{key}-{{{key}:.4f}}_val_{key}-{{val_{key}:.4f}}'
                                                          for key in ['loss', 'accuracy']]) + '.h5')

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
    trainer = ClassificationPipeline(config)

    if config.STATUS == 'train':
        trainer.train()

    if config.STATUS == 'inference':
        results = trainer.inference()
        y_score = results[[f'class_{i}_score' for i in range(config.CLASS_NUMBER)]].values
        results.to_csv(os.path.join(config.DATA_ROOT, 'models', 'results.csv'), index=False)
        y_test = results['class_id']
        y_test = pd.get_dummies(y_test).values
        clf_metrics = ClassificationMetrics(y_test, y_score, label_name=config.LABEL_NAME)
        clf_plot = ClassificationMetricPlot(clf_metrics, cm_norm=False)
        clf_plot.show()
