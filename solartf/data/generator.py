import copy
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from .data.label.iterator import ObjectDetectionLabelIterator
from .data.image.type import ImageInput
from .data.set.generator import (object_detection_input_augmentation)


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
        if self.dataset_type == 'test':
            batch_image_input = []
        else:
            batch_image_input = np.zeros((self.batch_size,) + self.image_shape)
        batch_label_output = np.zeros((self.batch_size, self.n_classes))
        # todo: we can apply multiple process or batch process here
        for idx, index in enumerate(indexes):
            input_image_path = self.image_path_list[index]

            image_input = ImageInput(index,
                                       input_image_path,
                                       image_type=self.image_type,
                                       image_shape=self.image_shape)
            for augment in self.augment:
                augment.execute(image_input)

            image_input.image_array = image_input.image_array / 255.
            label = os.path.basename(os.path.dirname(input_image_path))
            label_index = self.class_map_invert[label]

            if self.dataset_type == 'test':
                batch_image_input.append(image_input)
            else:
                batch_image_input[idx] = image_input.image_array
            batch_label_output[idx, label_index] = 1.

        return batch_image_input, batch_label_output


class KerasSSDDirectoryGenerator(KerasGeneratorBase):
    """
    todo: some components must be redesigned
    """

    def __init__(self,
                 image_dir,
                 label_dir,
                 image_shape,
                 n_classes,
                 mask_dir=None,
                 shuffle=True,
                 batch_size=32,
                 image_type='rgb',
                 in_memory=False,
                 augment=False,
                 brightness_ratio=None,
                 flip_orientation=None,
                 scale_ratio=None,
                 degree=None,
                 h_shift=None,
                 v_shift=None,
                 processor=None,
                 bbox_exclude=None,
                 detector=None,
                 image_converter=None):

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_shape = image_shape
        self.n_classes = n_classes
        self.mask_dir = mask_dir
        self.image_type = image_type
        self.shuffle = shuffle
        self.augment = augment
        self.brightness_ratio = brightness_ratio
        self.flip_orientation = flip_orientation
        self.scale_ratio = scale_ratio
        self.degree = degree
        self.h_shift = h_shift
        self.v_shift = v_shift
        self.processor = processor

        self.detector = detector
        self.image_converter = image_converter

        self.gt_input_list = np.array([gt_input
                                       for gt_input in ObjectDetectionLabelIterator(image_dir,
                                                                                    label_dir,
                                                                                    mask_dir=self.mask_dir,
                                                                                    extension=('.json',),
                                                                                    shuffle=shuffle,
                                                                                    bbox_exclude=bbox_exclude)])

        self.in_memory = in_memory
        self.indexes = np.arange(len(self.gt_input_list))
        self.image_input_list = [None] * self.indexes.size
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.gt_input_list) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self._data_generation(indexes)

    def _data_generation(self, indexes):
        # object_detection_input_augmentation
        if self.processor is None:
            batch_image_input = []
        else:
            batch_image_input = np.empty((self.batch_size, *self.image_shape))

        # todo: anchor number must be calculated
        # the shape will be (batch_size, anchor_nb, 12 + n_class)

        mbox_conf = []
        mbox_loc = []
        mbox_priorbox = []
        fpn_output = []

        batch_index = 0
        for index in indexes:
            gt_input = self.gt_input_list[index]
            if self.image_input_list[index] is None:
                source_ref = gt_input.image.source_ref
                image_input = ImageInput(gt_input.image_id,
                                           source_ref,
                                           image_type=self.image_type,
                                           image_shape=self.image_shape)
                if self.in_memory:
                    self.image_input_list[index] = image_input
            else:
                image_input = self.image_input_list[index]

            if self.image_converter is not None and self.detector is not None:
                results = self.detector(image_input)
                if results is not None:
                    detections, _ = results
                    image_input = self.image_converter(copy.deepcopy(image_input), detections)

            image_input, gt_input = object_detection_input_augmentation(copy.deepcopy(image_input),
                                                                        copy.deepcopy(gt_input),
                                                                        brightness_ratio=self.brightness_ratio,
                                                                        flip_orientation=self.flip_orientation,
                                                                        scale_ratio=self.scale_ratio,
                                                                        degree=self.degree,
                                                                        h_shift=self.h_shift,
                                                                        v_shift=self.v_shift,
                                                                        augment=self.augment)

            if self.processor is None:
                outputs = {'mbox_conf': [],
                           'mbox_loc': gt_input.bboxes.to_array(coord='corner'),
                           'mbox_priorbox': []}
                batch_image_input.append(image_input)
            else:
                inputs, outputs = self.processor({'image_input': image_input, 'gt_input': gt_input})
                batch_image_input[batch_index] = inputs['ssd_image_input']

            mbox_conf.append(outputs['mbox_conf'])
            mbox_loc.append(outputs['mbox_loc'])
            mbox_priorbox.append(outputs['mbox_priorbox'])
            if 'fpn_output' in outputs:
                fpn_output.append(outputs['fpn_output'])

            batch_index += 1

        if len(fpn_output) > 0:
            batch_image_output = {'mbox_conf': np.stack(mbox_conf, axis=0),
                                  'mbox_loc': np.stack(mbox_loc, axis=0),
                                  'mbox_priorbox': np.stack(mbox_priorbox, axis=0),
                                  'fpn_output': np.stack(fpn_output, axis=0)}
        else:
            batch_image_output = {'mbox_conf': np.stack(mbox_conf, axis=0),
                                  'mbox_loc': np.stack(mbox_loc, axis=0),
                                  'mbox_priorbox': np.stack(mbox_priorbox, axis=0)}

        return batch_image_input, batch_image_output
