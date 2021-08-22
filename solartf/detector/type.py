from typing import List
from solartf.data.bbox.type import (BBox, BBoxes)
from solartf.data.image.type import Image


class DetectionInput:
    """
    A class collecting information of ground truth for evaluating object detection results
    """
    def __init__(self, image_id, image: Image, bboxes: List[BBox], bbox_exclude=None):
        if bbox_exclude is None:
            self.bbox_exclude = {}

        self.image_id = image_id
        self.image = image
        self.bboxes = BBoxes(bboxes, bbox_exclude=bbox_exclude)
