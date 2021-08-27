import numpy as np
from typing import List


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Keypoint:
    def __init__(self, class_id, x, y, visible=1, class_map=None):
        self.class_id = class_id
        self.point = Point(x, y)
        self.visible = visible
        self.class_map = class_map


class Keypoints:
    def __init__(self, keypoints: List[Keypoint]):
        self._keypoints = keypoints

        self._labels = np.array([kpt.class_id for kpt in self._keypoints])
        if self._labels.size > 0:
            kpt_tensor = np.stack([[kpt.x, kpt.y] for kpt in self._keypoints], axis=0)
        else:
            kpt_tensor = np.empty(shape=(0, 2))

        self.kpt_tensor = kpt_tensor
