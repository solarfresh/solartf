from typing import List
from solartf.data.bbox.type import BBox
from solartf.data.bbox.processor import BBoxeInput
from solartf.data.image.type import Image


class DetectionInput:
    """
    A class collecting information of ground truth for evaluating object detection results
    """
    def __init__(self, image: Image, bboxes: List[BBox], bbox_exclude=None):
        if bbox_exclude is None:
            self.bbox_exclude = {}

        # todo: image_id must be contained by ImageInput
        self.image = image
        self.bboxes_input = BBoxeInput(bboxes, bbox_exclude=bbox_exclude)
