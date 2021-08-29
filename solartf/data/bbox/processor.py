import cv2
import numpy as np
from typing import (List, Tuple)
from .type import (BBox, BBoxesTensor)


class BBoxProcessor:
    indexes: np.array
    labels: np.array
    bboxes_tensor: np.array
    scale: Tuple

    def crop(self, xmin: int, ymin: int, xmax: int, ymax: int):
        bboxes_tensor = self.bboxes_tensor.copy()
        labels = self.labels.copy()
        indexes = self.indexes.copy()

        bboxes_tensor[bboxes_tensor[..., 0] < xmin, 0] = xmin
        bboxes_tensor[bboxes_tensor[..., 1] < ymin, 1] = ymin
        bboxes_tensor[bboxes_tensor[..., 2] > xmax, 2] = xmax
        bboxes_tensor[bboxes_tensor[..., 3] > ymax, 3] = ymax
        ignore_indexes = []
        for index in range(labels.size):
            bbox = bboxes_tensor[index]
            if (bbox[..., 0] > bbox[..., 2]) or (bbox[..., 1] > bbox[..., 3]):
                ignore_indexes.append(index)

        self.bboxes_tensor = np.delete(bboxes_tensor, ignore_indexes, axis=0)
        self.labels = np.delete(labels, ignore_indexes)
        self.indexes = np.delete(indexes, ignore_indexes)

        return self

    def flip(self, image_shape, orientation=None):
        height, width, depth = image_shape
        bboxes_tensor = self.bboxes_tensor.copy()

        if orientation == 'horizontal':
            bboxes_tensor[:, 1] = width - bboxes_tensor[:, 1]
            bboxes_tensor[:, 3] = width - bboxes_tensor[:, 3]
            bboxes_tensor[:, [1, 3]] = bboxes_tensor[:, [3, 1]]

        if orientation == 'vertical':
            bboxes_tensor[:, 0] = height - bboxes_tensor[:, 0]
            bboxes_tensor[:, 2] = height - bboxes_tensor[:, 2]
            bboxes_tensor[:, [0, 2]] = bboxes_tensor[:, [2, 0]]

        self.bboxes_tensor = bboxes_tensor

        return self

    def affine(self, transfer_matrix):
        """
        :param transfer_matrix: the transformation matrix
        """
        bboxes_tensor = self.bboxes_tensor.copy()

        left_top = bboxes_tensor[:, [1, 0]]
        right_bottom = bboxes_tensor[:, [3, 2]]
        pts = np.concatenate([left_top, right_bottom], axis=0)
        pts = np.expand_dims(pts, 1)
        affined_pts = np.squeeze(cv2.transform(pts, transfer_matrix))
        left_top, right_bottom = np.split(affined_pts, 2, axis=0)

        self.bboxes_tensor = np.concatenate([left_top[:, [1, 0]], right_bottom[:, [1, 0]]], axis=1)

        return self

    def resize(self, scale: Tuple):
        bboxes_tensor = self.bboxes_tensor.copy()

        bboxes_tensor[:, [0, 2]] = bboxes_tensor[:, [0, 2]] * scale[0] / self.scale[0]
        bboxes_tensor[:, [1, 3]] = bboxes_tensor[:, [1, 3]] * scale[1] / self.scale[1]
        self.scale = scale

        self.bboxes_tensor = bboxes_tensor

        return self


class BBoxeInput(BBoxProcessor):
    def __init__(self, bboxes: List[BBox], bbox_exclude=None):
        if bbox_exclude is None:
            self.bbox_exclude = {}
        else:
            self.bbox_exclude = bbox_exclude

        self._bboxes = np.array(bboxes)
        self.indexes = np.arange(len(self._bboxes))
        self.n_class = 0
        if self.indexes.size > 0:
            self.n_class = len(bboxes[0].class_map)

        self._labels = np.array([bbox.class_id
                                 for key, value in self.bbox_exclude.items()
                                 for bbox in bboxes
                                 if bbox.__getattribute__(key) not in value])
        if self._labels.size > 0:
            bbox_tensor = np.stack([[bbox.left, bbox.top, bbox.right, bbox.bottom]
                                    for bbox in bboxes
                                    for key, value in self.bbox_exclude.items()
                                    if bbox.__getattribute__(key) not in value], axis=0)
        else:
            bbox_tensor = np.empty(shape=(0, 4))

        self.bboxes_tensor = BBoxesTensor(bboxes=bbox_tensor)

    @property
    def labels(self):
        return self._labels[self.indexes]

    @property
    def bboxes(self):
        return self._bboxes[self.indexes]
