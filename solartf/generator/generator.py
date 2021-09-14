import copy
import os
import numpy as np
from typing import (Any, List)
from solartf.core.generator import KerasGeneratorBase
from solartf.data.image.processor import ImageInput


class CycleGANImageDirectoryGenerator(KerasGeneratorBase):
    def __init__(self,
                 image_dir_x,
                 image_shape_x,
                 image_dir_y,
                 image_shape_y,
                 image_type_x='bgr',
                 image_type_y='bgr',
                 dataset_type='train',
                 batch_size=32,
                 shuffle=True,
                 augment=None,
                 image_format=None,
                 processor=None,
                 in_memory=False,):

        if image_format is None:
            image_format = ('.png', '.jpg', '.jpeg')

        self.image_dir_x = image_dir_x
        self.image_shape_x = image_shape_x
        self.image_type_x = image_type_x

        self.image_dir_y = image_dir_y
        self.image_shape_y = image_shape_y
        self.image_type_y = image_type_y

        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.processor = processor
        self.in_memory = in_memory

        self.shuffle = shuffle

        self.image_path_list_x = [os.path.join(root, fname) for root, _, fnames in os.walk(self.image_dir_x)
                                  for fname in fnames if fname.endswith(image_format)]
        self.indexes_x = np.arange(len(self.image_path_list_x))
        self.image_input_list_x: List[Any] = [None] * self.indexes_x.size

        self.image_path_list_y = [os.path.join(root, fname) for root, _, fnames in os.walk(self.image_dir_y)
                                  for fname in fnames if fname.endswith(image_format)]
        self.indexes_y = np.arange(len(self.image_path_list_y))
        self.image_input_list_y: List[Any] = [None] * self.indexes_y.size

        np.random.shuffle(self.indexes_x)
        np.random.shuffle(self.indexes_y)

        if augment is None:
            self.augment = []
        else:
            self.augment = augment

    def __len__(self):
        len_x = int(np.floor(len(self.image_input_list_x) / self.batch_size))
        len_y = int(np.floor(len(self.image_input_list_y) / self.batch_size))
        return min(len_x, len_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes_x)
            np.random.shuffle(self.indexes_y)

    def __getitem__(self, index):
        indexes_x = self.indexes_x[index * self.batch_size:(index + 1) * self.batch_size]
        indexes_y = self.indexes_y[index * self.batch_size:(index + 1) * self.batch_size]
        return self._data_generation(indexes_x, indexes_y)

    def _data_generation(self, indexes_x, indexes_y):
        image_input_list_x = []
        image_input_list_y = []
        for index_x, index_y in zip(indexes_x, indexes_y):
            input_image_path_x = self.image_path_list_x[index_x]
            input_image_path_y = self.image_path_list_y[index_y]

            image_input_x = ImageInput(input_image_path_x,
                                       image_type=self.image_type_x,
                                       image_shape=self.image_shape_x)
            image_input_y = ImageInput(input_image_path_y,
                                       image_type=self.image_type_y,
                                       image_shape=self.image_shape_y)
            if self.in_memory:
                self.image_input_list_x[index_x] = image_input_x
                self.image_input_list_y[index_y] = image_input_y

            image_input_list_x.append(copy.deepcopy(image_input_x))
            image_input_list_y.append(copy.deepcopy(image_input_y))

        for augment in self.augment:
            augment.execute(image_input_list_x)
            augment.execute(image_input_list_y)

        if self.dataset_type == 'test':
            return {'real_x': image_input_list_x, 'real_y': image_input_list_y}
        else:
            return self.processor({'real_x': image_input_list_x, 'real_y': image_input_list_y})
