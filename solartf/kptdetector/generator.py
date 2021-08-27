import copy
import numpy as np
from typing import (Any, List,)
from solartf.core.generator import KerasGeneratorBase
from solartf.data.image.processor import ImageInput
from .iterator import KeypointLabelIterator


class KeypointDirectoryGenerator(KerasGeneratorBase):
    def __init__(self,
                 image_dir,
                 label_dir,
                 image_shape,
                 shuffle=True,
                 batch_size=32,
                 image_type='bgr',
                 dataset_type='train',
                 augment=None,
                 processor=None,
                 in_memory=False,):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_shape = image_shape
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.image_type = image_type
        self.dataset_type = dataset_type
        self.in_memory = in_memory
        self.processor = processor

        self.gt_input_list = np.array([
            gt_input for gt_input in KeypointLabelIterator(image_dir,
                                                           label_dir,
                                                           extension=('.json',),
                                                           shuffle=shuffle,)])

        if augment is None:
            self.augment = []
        else:
            self.augment = augment

        self.indexes = np.arange(len(self.gt_input_list))
        self.image_input_list: List[Any] = [None] * self.indexes.size

    def __len__(self):
        return int(np.floor(len(self.gt_input_list) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self._data_generation(indexes)

    def _data_generation(self, indexes):
        image_input_list = []
        kpt_input_list = []
        for index in indexes:
            gt_input = self.gt_input_list[index]
            kpt_input_list.append(copy.deepcopy(gt_input.keypoint_input))
            if self.image_input_list[index] is None:
                source_ref = gt_input.image.source_ref
                image_input = ImageInput(source_ref,
                                         image_type=self.image_type,
                                         image_shape=self.image_shape)
                if self.in_memory:
                    self.image_input_list[index] = image_input

                image_input_list.append(copy.deepcopy(image_input))
            else:
                image_input_list.append(copy.deepcopy(self.image_input_list[index]))

        for augment in self.augment:
            augment.execute(image_input_list, kpt_input_list)
        else:
            for image_input, kpt_input in zip(image_input_list, kpt_input_list):
                kpt_input.resize(scale=image_input.scale)

        if self.dataset_type == 'test':
            return image_input_list, kpt_input_list
        else:
            batch_inputs, batch_outputs = self.processor({'image_input_list': image_input_list,
                                                          'kpt_input_list': kpt_input_list})
            return batch_inputs, batch_outputs
