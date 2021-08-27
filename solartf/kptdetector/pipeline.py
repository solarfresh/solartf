from solartf.core.pipeline import TFPipelineBase
from .generator import KeypointDirectoryGenerator


class KeypointDetectPipeline(TFPipelineBase):
    def inference(self, dataset_type='test'):
        pass

    def load_dataset(self):
        self.dataset = {}
        for set_type in ['train', 'valid', 'test']:
            self.dataset[set_type] = KeypointDirectoryGenerator(
                image_dir=self.image_dir[set_type],
                label_dir=self.label_dir[set_type],
                image_shape=self.config.IMAGE_SHAPE,
                shuffle=self.shuffle[set_type],
                batch_size=self.batch_size[set_type],
                augment=self.augment[set_type],
                image_type=self.config.IMAGE_TYPE,
                processor=self.model.data_preprocess)

        return self
