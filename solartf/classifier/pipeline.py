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
        self.dataset = {}
        for set_type in ['train', 'valid', 'test']:
            self.dataset[set_type] = ClassifierDirectoryGenerator(
                image_dir=self.image_dir[set_type],
                image_shape=self.config.IMAGE_SHAPE,
                dataset_type=set_type,
                shuffle=self.shuffle[set_type],
                batch_size=self.batch_size[set_type],
                augment=self.augment[set_type],
                image_type=self.config.IMAGE_TYPE,)

        return self
