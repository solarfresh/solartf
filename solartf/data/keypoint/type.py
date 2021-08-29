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
