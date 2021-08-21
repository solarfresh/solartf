import os
import numpy as np
import pandas as pd
from solartf.core.pipeline import TFPipelineBase
from .generator import ClassifierDirectoryGenerator


class ClassificationPipeline(TFPipelineBase):

    def inference(self, dataset_type='test'):
        self.load_model().load_dataset()

        results = []
        for inputs, outputs in self.dataset[dataset_type]:
            fnames = [os.path.basename(image_input.image_path)
                      for image_input in inputs]
            image_arrays = np.stack([image_input.image_array
                                     for image_input in inputs], axis=0)
            predict_results = self.model.predict(image_arrays)
            for fname, gt, predict in zip(fnames, outputs, predict_results):
                result = {
                    'fname': fname,
                    'class_id': np.argmax(gt),
                    'label': np.argmax(predict)
                }
                for index, score in enumerate(predict):
                    result.update({f'class_{index}_score': score})

                results.append(result)

        return pd.DataFrame(results)

    def load_dataset(self):
        image_dir = {
            'train': self.config.TRAIN_IMAGE_PATH,
            'valid': self.config.VALID_IMAGE_PATH,
            'test': self.config.TEST_IMAGE_PATH
        }
        shuffle = {
            'train': self.config.TRAIN_SHUFFLE,
            'valid': self.config.VALID_SHUFFLE,
            'test': self.config.TEST_SHUFFLE
        }
        batch_size = {
            'train': self.config.TRAIN_BATCH_SIZE,
            'valid': self.config.VALID_BATCH_SIZE,
            'test': self.config.TEST_BATCH_SIZE
        }
        augment = {
            'train': self.config.TRAIN_AUGMENT,
            'valid': self.config.VALID_AUGMENT,
            'test': None
        }

        self.dataset = {}
        for set_type in ['train', 'valid', 'test']:
            self.dataset[set_type] = ClassifierDirectoryGenerator(
                image_dir=image_dir[set_type],
                image_shape=self.config.IMAGE_SHAPE,
                dataset_type=set_type,
                shuffle=shuffle[set_type],
                batch_size=batch_size[set_type],
                augment=augment[set_type],
                image_type=self.config.IMAGE_TYPE,)

        return self

    def load_model(self):
        self.model = self.config.MODEL(input_shape=self.config.IMAGE_SHAPE,
                                       n_classes=self.config.CLASS_NUMBER)

        self.model.build_model()

        if self.config.MODEL_WEIGHT_PATH is not None:
            self.model.load_weights(self.config.MODEL_WEIGHT_PATH,
                                    skip_mismatch=True,
                                    by_name=True)

        self.model.compile(optimizer=self.config.TRAIN_OPTIMIZER,
                           loss=self.config.TRAIN_LOSS,
                           metrics=self.config.TRAIN_METRIC,
                           loss_weights=self.config.TRAIN_LOSS_WEIGHTS)

        return self
