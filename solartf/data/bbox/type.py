import numpy as np
import tensorflow as tf
from typing import List


class BBoxesMixin:
    bboxes: np.array
    method = 'numpy'

    def to_array(self, coord='corner'):
        if self.method == 'numpy':
            _stack = np.stack
        elif self.method == 'tensorflow':
            _stack = tf.stack
        else:
            ValueError(f'The method {self.method} is not supported...')

        if coord == 'corner':
            return self.bboxes
        elif coord == 'centroid':
            return _stack((self.width_centers, self.height_centers, self.widths, self.heights), axis=-1)
        elif coord == 'minmax':
            return _stack((self.tops, self.lefts, self.bottoms, self.rights), axis=-1)
        else:
            raise ValueError(f'The coordinate {coord} does not support...')

    @property
    def lefts(self):
        return self.bboxes[..., 0]

    @property
    def tops(self):
        return self.bboxes[..., 1]

    @property
    def rights(self):
        return self.bboxes[..., 2]

    @property
    def bottoms(self):
        return self.bboxes[..., 3]

    @property
    def center(self):
        return (self.bboxes[..., 2:] + self.bboxes[..., :2]) / 2

    @property
    def widths(self):
        return self.bboxes[..., 2] - self.bboxes[..., 0]

    @property
    def heights(self):
        return self.bboxes[..., 3] - self.bboxes[..., 1]

    @property
    def width_centers(self):
        return (self.bboxes[..., 2] + self.bboxes[..., 0]) / 2

    @property
    def height_centers(self):
        return (self.bboxes[..., 3] + self.bboxes[..., 1]) / 2


class BBox(object):
    def __init__(self,
                 class_id,
                 bbox,
                 class_map=None,
                 visibility=None,
                 blur_level=None,
                 region_level=None,
                 truncation=None):
        """
        :param class_id: ID of the corresponding class
        :param bbox: (top, left, bottom, right)
        :param visibility
               0 indicates "visible"
               1 indicates "partial-occlusion"
               2 indicates "full-occlusion"
        :param blur_level
               0 indicates no-blur
               1 indicates blur
        :param region_level
               When an object is detected, we are also interested on different part/region of the object
               For example, we can split a person into head, visible region, and full region.
        :param truncation indicates the degree of object parts appears outside a frame
                no truncation = 0 (truncation ratio 0%),
                partial truncation = 1 (truncation ratio 1% ~ 50%))
        """
        # todo: must be corrected from original json file
        self.class_id = class_id
        self.class_map = class_map
        self.bbox = bbox
        self.visibility = visibility
        self.blur_level = blur_level
        self.region_level = region_level
        self.truncation = truncation

    @property
    def top(self):
        return self.bbox[0]

    @property
    def left(self):
        return self.bbox[1]

    @property
    def bottom(self):
        return self.bbox[2]

    @property
    def right(self):
        return self.bbox[3]

    @property
    def width(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def height(self):
        return self.bbox[2] - self.bbox[0]


class BBoxesTensor(BBoxesMixin):
    def __init__(self, bboxes: np.array, coord='corner', method='numpy'):
        self.coord = coord
        self.method = method

        if self.method == 'numpy':
            _concat = np.concatenate
        elif self.method == 'tensorflow':
            _concat = tf.concat
        else:
            ValueError(f'The method {self.method} is not supported...')

        if self.coord == 'corner':
            self.bboxes = bboxes
        elif self.coord == 'centroid':
            bboxes_left_top = bboxes[..., :2] - .5 * bboxes[..., 2:]
            bboxes_right_bottom = bboxes[..., :2] + .5 * bboxes[..., 2:]
            self.bboxes = _concat((bboxes_left_top, bboxes_right_bottom), axis=-1)
        elif self.coord == 'minmax':
            bboxes_left_top = bboxes[..., [1, 0]]
            bboxes_right_bottom = bboxes[..., [3, 2]]
            self.bboxes = _concat((bboxes_left_top, bboxes_right_bottom), axis=-1)
        else:
            raise ValueError(f'The coordinate {self.coord} does not support...')
