import cv2
import numpy as np
from typing import (List, Tuple)
from .type import (BBox, BBoxesTensor)


class BBoxProcessor:
    indexes: np.array
    _bboxes_tensor: np.array
    scale: Tuple

    def crop(self, xmin: int, ymin: int, xmax: int, ymax: int):
        bboxes_tensor = self._bboxes_tensor.copy()
        indexes = self.indexes.copy()
        if bboxes_tensor.size < 1:
            return self

        bboxes_tensor[bboxes_tensor[..., 0] < xmin, 0] = xmin
        bboxes_tensor[bboxes_tensor[..., 1] < ymin, 1] = ymin
        bboxes_tensor[bboxes_tensor[..., 2] > xmax, 2] = xmax
        bboxes_tensor[bboxes_tensor[..., 3] > ymax, 3] = ymax
        ignore_indexes = []
        for index in range(indexes.size):
            bbox = bboxes_tensor[indexes[index]]
            if (bbox[..., 0] > bbox[..., 2]) or (bbox[..., 1] > bbox[..., 3]):
                ignore_indexes.append(index)

        self._bboxes_tensor = bboxes_tensor
        self.indexes = np.delete(indexes, ignore_indexes)

        return self

    def flip(self, image_shape, orientation=None):
        height, width, depth = image_shape
        bboxes_tensor = self._bboxes_tensor.copy()
        if bboxes_tensor.size < 1:
            return self

        if orientation == 'vertical':
            bboxes_tensor[:, 1] = width - bboxes_tensor[:, 1]
            bboxes_tensor[:, 3] = width - bboxes_tensor[:, 3]
            bboxes_tensor[:, [1, 3]] = bboxes_tensor[:, [3, 1]]

        if orientation == 'horizontal':
            bboxes_tensor[:, 0] = height - bboxes_tensor[:, 0]
            bboxes_tensor[:, 2] = height - bboxes_tensor[:, 2]
            bboxes_tensor[:, [0, 2]] = bboxes_tensor[:, [2, 0]]

        self._bboxes_tensor = bboxes_tensor

        return self

    def affine(self, transfer_matrix):
        """
        :param transfer_matrix: the transformation matrix
        """
        bboxes_tensor = self._bboxes_tensor.copy()
        if bboxes_tensor.size < 1:
            return self

        left_top = bboxes_tensor[..., [0, 1]]
        right_bottom = bboxes_tensor[..., [2, 3]]
        pts = np.concatenate([left_top, right_bottom], axis=0)
        pts = np.expand_dims(pts, 1)
        affined_pts = np.squeeze(cv2.transform(pts, transfer_matrix))
        left_top, right_bottom = np.split(affined_pts, 2, axis=0)

        bboxes_tensor = np.concatenate([left_top[:, [0, 1]], right_bottom[:, [0, 1]]], axis=1)
        self._bboxes_tensor = bboxes_tensor

        return self

    def perspective(self, transfer_matrix):
        bboxes_tensor = self._bboxes_tensor.copy()
        if bboxes_tensor.size < 1:
            return self

        left_top = bboxes_tensor[..., [0, 1]]
        right_bottom = bboxes_tensor[..., [2, 3]]
        pts = np.concatenate([left_top, right_bottom], axis=0)
        pts = np.expand_dims(pts.astype(np.float32), 0)
        pts = cv2.perspectiveTransform(pts, transfer_matrix).astype(np.int32)
        left_top, right_bottom = np.split(np.squeeze(pts), 2, axis=0)

        bboxes_tensor = np.concatenate([left_top[:, [0, 1]], right_bottom[:, [0, 1]]], axis=1)
        self._bboxes_tensor = bboxes_tensor
        return self

    def resize(self, scale: Tuple):
        bboxes_tensor = self._bboxes_tensor.copy()
        if bboxes_tensor.size < 1:
            return self

        bboxes_tensor[:, [0, 2]] = bboxes_tensor[:, [0, 2]] * scale[1] / self.scale[1]
        bboxes_tensor[:, [1, 3]] = bboxes_tensor[:, [1, 3]] * scale[0] / self.scale[0]
        self.scale = scale

        self._bboxes_tensor = bboxes_tensor

        return self


class BBoxeInput(BBoxProcessor):
    def __init__(self, bboxes: List[BBox], bbox_exclude=None):
        if bbox_exclude is None:
            self.bbox_exclude = {'class_id': []}
        else:
            self.bbox_exclude = bbox_exclude

        self._bboxes = np.array(bboxes)
        self.n_class = 0
        self._labels = np.array([bbox.class_id for bbox in bboxes])
        self._bboxes_tensor = np.stack([[bbox.left, bbox.top, bbox.right, bbox.bottom] for bbox in bboxes])

        indexes = []
        for idx, bbox in enumerate(bboxes):
            if self.n_class < 1:
                self.n_class = len(bboxes[0].class_map)

            for key, value in self.bbox_exclude.items():
                if bbox.__getattribute__(key) not in value:
                    indexes.append(idx)

        self.indexes = np.array(indexes)
        self.scale = (1., 1.)

    @property
    def labels(self):
        return self._labels[self.indexes]

    @property
    def bboxes(self):
        return self._bboxes[self.indexes]

    @property
    def bboxes_tensor(self):
        return BBoxesTensor(bboxes=self._bboxes_tensor[self.indexes])
