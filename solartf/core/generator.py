import copy
import os
import numpy as np
from typing import (Any, List)
from tensorflow.keras.utils import Sequence
from solartf.data.image.processor import ImageInput


class KerasGeneratorBase(Sequence):
    def __len__(self):
        """Denotes the number of batches per epoch"""
        raise NotImplementedError

    def __getitem__(self, index):
        """Generate one batch of data"""
        raise NotImplementedError

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        raise NotImplementedError


class ImageDirectoryGenerator(KerasGeneratorBase):
    def __init__(self,
                 image_dir,
                 image_shape,
                 image_type='bgr',
                 dataset_type='train',
                 batch_size=32,
                 shuffle=True,
                 augment=None,
                 image_format=None,
                 processor=None,
                 in_memory=False,):

        if image_format is None:
            image_format = ('.png', '.jpg', '.jpeg')

        self.image_dir = image_dir
        self.image_shape = image_shape
        self.image_type = image_type
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.processor = processor
        self.in_memory = in_memory

        self.shuffle = shuffle
        self.image_path_list = [os.path.join(root, fname) for root, _, fnames in os.walk(self.image_dir)
                                for fname in fnames if fname.endswith(image_format)]

        self.indexes = np.arange(len(self.image_path_list))
        self.image_input_list: List[Any] = [None] * self.indexes.size
        np.random.shuffle(self.indexes)
        if augment is None:
            self.augment = []
        else:
            self.augment = augment

    def __len__(self):
        return int(np.floor(len(self.image_path_list) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self._data_generation(indexes)

    def _data_generation(self, indexes):
        image_input_list = []
        for index in indexes:
            input_image_path = self.image_path_list[index]

            image_input = ImageInput(input_image_path,
                                     image_type=self.image_type,
                                     image_shape=self.image_shape)
            if self.in_memory:
                self.image_input_list[index] = image_input

            image_input_list.append(copy.deepcopy(image_input))

        for augment in self.augment:
            augment.execute(image_input_list)

        if self.dataset_type == 'test':
            return image_input_list
        else:
            batch_inputs, batch_outputs = self.processor({'image_input_list': image_input_list})
            return batch_inputs, batch_outputs
