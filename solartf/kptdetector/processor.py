import cv2
import numpy as np
from typing import List
from solartf.data.image.processor import ImageInput
from solartf.data.keypoint.processor import KeypointInput


class RandomOcclusion:
    def __init__(self, shape=None, **kwargs):
        """
        :param shape: circle or rectangle. one of both will be selected randomly if it is None as default
        """
        self.shape = shape

        self.default_attr = {'radius': None, 'width': None, 'height': None}
        self.default_attr.update(kwargs)

    def execute(self,
                image_input_list: List[ImageInput],
                kpt_input_list: List[KeypointInput]):

        for image_input, kpt_input in zip(image_input_list, kpt_input_list):
            image_array = image_input.image_array.copy()
            for idx in kpt_input.indexes:
                center = kpt_input.points_tensor[idx]
                if np.random.randint(0, 2):
                    continue

                self._update_random_attr()
                image_array = self._occluding(image_array, center)
                kpt_input.labels[idx] = 0
                kpt_input.points_tensor[idx] = np.array([0., 0.])

            image_input.image_array = image_array

    def _occluding(self, image_array, center):
        mask = np.zeros_like(image_array, dtype=np.uint8)
        if self.shape == 'circle':
            cv2.circle(mask, center, self.radius, (255, 255, 255), -1)
        elif self.shape == 'rectangle':
            xmin = center[0] - self.width // 2
            xmax = center[0] + self.width // 2
            ymin = center[1] - self.height // 2
            ymax = center[1] + self.height // 2
            cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)
        else:
            raise ValueError(f'It does not support the shape {self.shape}...')

        return np.where(mask, np.random.randint(0, 256, size=image_array.shape).astype(np.uint8), image_array)

    def _update_random_attr(self):
        self.shape = 'circle' if np.random.random_integers(0, 1) else 'rectangle'
        for key, value in self.default_attr.items():
            if key not in ['radius', 'width', 'height']:
                continue

            if value is None:
                self.__setattr__(key, np.random.randint(10, 20))
            else:
                self.__setattr__(key, np.random.randint(value[0], value[1]))


class KeypointAugmentation:
    def __init__(self,
                 brightness_ratio=None,
                 flip_orientation=None,
                 scale_ratio=None,
                 degree=None,
                 h_shift=None,
                 v_shift=None):
        self.brightness_ratio = brightness_ratio
        self.flip_orientation = flip_orientation
        self.scale_ratio = scale_ratio
        self.degree = degree
        self.h_shift = h_shift
        self.v_shift = v_shift

    def execute(self,
                image_input_list: List[ImageInput],
                kpt_input_list: List[KeypointInput]):

        for image_input, kpt_input in zip(image_input_list, kpt_input_list):
            image_type = image_input.image_type

            if self.brightness_ratio is not None:
                image_input.brightness(ratio=self.brightness_ratio,
                                       image_type=image_type)

            if self.scale_ratio is not None:
                transfer_matrix = image_input.rescale(ratio=self.scale_ratio)
                kpt_input.affine(transfer_matrix=transfer_matrix)

            if self.flip_orientation is not None:
                orientation = image_input.flip(orientation=self.flip_orientation)
                kpt_input.flip(image_input.image_shape, orientation=orientation)

            if self.degree is not None:
                transfer_matrix = image_input.rotate(degree=self.degree)
                kpt_input.affine(transfer_matrix=transfer_matrix)

            if self.h_shift is not None or self.v_shift is not None:
                transfer_matrix = image_input.translate(horizontal=self.h_shift, vertical=self.v_shift)
                kpt_input.affine(transfer_matrix=transfer_matrix)
