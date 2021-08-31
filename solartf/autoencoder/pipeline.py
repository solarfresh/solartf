from solartf.core.generator import ImageDirectoryGenerator
from solartf.core.pipeline import TFPipelineBase


class CVAEPipeline(TFPipelineBase):
    def inference(self, output_shape=None, dataset_type='test'):
        raise NotImplementedError

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
