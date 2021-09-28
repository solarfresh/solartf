from typing import List
from solartf.data.image.type import Image
from solartf.data.keypoint.type import Keypoint
from solartf.data.keypoint.processor import KeypointInput


class KeypointDetectInput:
    def __init__(self, image: Image, keypoints: List[Keypoint], cls_prop: List):
        self.image = image
        self.keypoint_input = KeypointInput(keypoints=keypoints)
        self.classes = cls_prop
