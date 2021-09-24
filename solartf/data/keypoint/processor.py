import cv2
import numpy as np
from typing import (Any, List, Tuple)
from .type import Keypoint


class KeypointProcessor:
    indexes: np.array
    labels: np.array
    points_tensor: np.array
    scale: Tuple

    def crop(self, xmin: int, ymin: int, xmax: int, ymax: int):
        points_tensor = self.points_tensor.copy()
        labels = self.labels.copy()
        indexes = self.indexes.copy()

        ignore_indexes = []
        for index in range(labels.size):
            point = points_tensor[index]
            if (point[..., 0] < xmin) or (point[..., 0] > xmax) or \
                    (point[..., 1] < ymin) or (point[..., 1] > ymax):
                ignore_indexes.append(index)

        self.points_tensor = np.delete(points_tensor, ignore_indexes, axis=0)
        self.labels = np.delete(labels, ignore_indexes)
        self.indexes = np.delete(indexes, ignore_indexes)

        return self

    def flip(self, image_shape, orientation=None):
        height, width = image_shape[:2]
        points_tensor = self.points_tensor.copy()

        if orientation == 'horizontal':
            points_tensor[:, 0] = width - points_tensor[:, 0]

        if orientation == 'vertical':
            points_tensor[:, 1] = height - points_tensor[:, 1]

        self.points_tensor = points_tensor

        return self

    def affine(self, transfer_matrix):
        """
        :param transfer_matrix: the transformation matrix
        """
        points_tensor = self.points_tensor.copy()

        pts = points_tensor[..., [0, 1]]
        pts = np.expand_dims(pts, 1)
        self.points_tensor = np.squeeze(cv2.transform(pts, transfer_matrix))

        return self

    def perspective(self, transfer_matrix):
        points_tensor = self.points_tensor.copy()
        pts = points_tensor[..., [0, 1]].astype(np.float32)
        pts = np.expand_dims(pts, 0)
        points_tensor = cv2.perspectiveTransform(pts, transfer_matrix).astype(np.int32)
        self.points_tensor = np.squeeze(points_tensor)
        return self

    def resize(self, scale: Tuple):
        points_tensor = self.points_tensor.copy()

        points_tensor[..., 0] = points_tensor[..., 0] * scale[1] / self.scale[1]
        points_tensor[..., 1] = points_tensor[..., 1] * scale[0] / self.scale[0]
        self.scale = scale

        self.points_tensor = points_tensor

        return self


class KeypointInput(KeypointProcessor):
    def __init__(self, keypoints: List[Keypoint]):
        self._keypoints = np.array(keypoints)
        self.indexes = np.arange(len(self._keypoints))
        self.n_class = 0
        if self.indexes.size > 0:
            self.n_class = len(keypoints[0].class_map)

        self.labels = np.zeros(shape=(self.n_class,))
        for kpts in keypoints:
            self.labels[kpts.class_id] = kpts.visible

        self.points_tensor = np.array([[kp.point.x, kp.point.y] for kp in keypoints])
        self.scale = (1., 1.)

    @property
    def keypoints(self):
        return self._keypoints[self.indexes]
