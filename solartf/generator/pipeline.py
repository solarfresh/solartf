from solartf.core.pipeline import TFPipelineBase
from .generator import CycleGANImageDirectoryGenerator


class CycleGANPipeline(TFPipelineBase):
    def __init__(self, config):
        super(CycleGANPipeline, self).__init__(config)

        self.image_dir_x = {
            'train': self.config.TRAIN_IMAGE_PATH_X,
            'valid': self.config.VALID_IMAGE_PATH_X,
            'test': self.config.TEST_IMAGE_PATH_X
        }
        self.image_dir_y = {
            'train': self.config.TRAIN_IMAGE_PATH_Y,
            'valid': self.config.VALID_IMAGE_PATH_Y,
            'test': self.config.TEST_IMAGE_PATH_Y
        }

    def inference(self, dataset_type='test'):
        pass

    def load_dataset(self):
        if self.model is None:
            raise ValueError(f'model must be load before loading datasets')

        self.dataset = {}
        for set_type in ['train', 'valid', 'test']:
            self.dataset[set_type] = CycleGANImageDirectoryGenerator(
                image_dir_x=self.image_dir_x[set_type],
                image_shape_x=self.config.IMAGE_SHAPE_X,
                image_type_x=self.config.IMAGE_TYPE_X,
                image_dir_y=self.image_dir_y[set_type],
                image_shape_y=self.config.IMAGE_SHAPE_Y,
                image_type_y=self.config.IMAGE_TYPE_Y,
                shuffle=self.shuffle[set_type],
                batch_size=self.batch_size[set_type],
                augment=self.augment[set_type],
                dataset_type=set_type,
                processor=self.model.data_preprocess)

        return self

