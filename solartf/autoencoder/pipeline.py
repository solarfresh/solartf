import os
import numpy as np
from solartf.core.generator import ImageDirectoryGenerator
from solartf.core.pipeline import TFPipelineBase


class CVAEPipeline(TFPipelineBase):
    def inference(self, dataset_type='test'):
        self.load_model().load_dataset()
        for image_input_list in self.dataset[dataset_type]:
            image_arrays = np.stack([image_input.image_array
                                     for image_input in image_input_list], axis=0)
            predict_results = self.model.predict(image_arrays)
            decoded_images = predict_results['decoded_image']
            logpzs = predict_results['logpz']
            for image_input, decoded_image in zip(image_input_list, decoded_images):
                image_path = image_input.image_path
                fname = os.path.basename(image_path)
                yield fname, (decoded_image * 255.).astype(np.uint8)

    def load_dataset(self):
        if self.model is None:
            raise ValueError(f'model must be load before loading datasets')

        self.dataset = {}
        for set_type in ['train', 'valid', 'test']:
            self.dataset[set_type] = ImageDirectoryGenerator(
                image_dir=self.image_dir[set_type],
                image_shape=self.config.IMAGE_SHAPE,
                shuffle=self.shuffle[set_type],
                batch_size=self.batch_size[set_type],
                augment=self.augment[set_type],
                image_type=self.config.IMAGE_TYPE,
                dataset_type=set_type,
                processor=self.model.data_preprocess)

        return self
