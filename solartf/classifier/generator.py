import os
import numpy as np
from solartf.core.generator import KerasGeneratorBase
from solartf.data.image.processor import ImageInput


class ClassifierDirectoryGenerator(KerasGeneratorBase):
    def __init__(self,
                 image_dir,
                 image_shape,
                 image_type='bgr',
                 dataset_type='train',
                 batch_size=32,
                 shuffle=True,
                 augment=None,
                 image_format=None):

        if image_format is None:
            image_format = ('.png', '.jpg', '.jpeg')

        self.image_dir = image_dir
        self.image_shape = image_shape
        self.image_type = image_type
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        if os.path.exists(self.image_dir):
            self.labels = sorted([dname for dname in os.listdir(self.image_dir)
                                  if os.path.isdir(os.path.join(self.image_dir, dname))])
        else:
            self.labels = []

        self.n_classes = len(self.labels)
        self.class_map = dict(zip(range(self.n_classes), self.labels))
        self.class_map_invert = dict(zip(self.labels, range(self.n_classes)))
        self.shuffle = shuffle
        self.image_path_list = [os.path.join(root, fname) for root, _, fnames in os.walk(self.image_dir)
                                for fname in fnames if fname.endswith(image_format)]

        self.indexes = np.arange(len(self.image_path_list))
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
            image_input_list.append(image_input)

        for augment in self.augment:
            augment.execute(image_input_list)

        if self.dataset_type == 'test':
            batch_image_input = []
        else:
            batch_image_input = np.zeros((self.batch_size,) + self.image_shape)

        batch_label_output = np.zeros((self.batch_size, self.n_classes))
        for idx, image_input in enumerate(image_input_list):
            label = os.path.basename(os.path.dirname(image_input.image_path))
            label_index = self.class_map_invert[label]

            if self.dataset_type == 'test':
                batch_image_input.append(image_input)
            else:
                batch_image_input[idx] = image_input.image_array
            batch_label_output[idx, label_index] = 1.

        return batch_image_input, batch_label_output
