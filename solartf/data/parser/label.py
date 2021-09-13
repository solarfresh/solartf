import json
from solartf.data.image.type import Image
from solartf.data.bbox.type import BBox
from solartf.data.keypoint.type import Keypoint


class LabelReader:
    def __init__(self, filepath):
        self.filepath = filepath

        with open(self.filepath) as json_file:
            self.annotation = json.load(json_file)

        image_size = self.annotation['annotations']['image_size']
        image_prop = self.annotation['metadata']
        self.image_prop = Image(source_ref=self.annotation['source-ref'],
                                image_shape=(image_size['height'], image_size['width'], image_size['depth']))
        for key, value in image_prop.items():
            key = key.replace('-', '_')
            if hasattr(self.image_prop, key):
                setattr(self.image_prop, key, value)

        self.class_map = None
        if 'metadata' in self.annotation:
            if 'class-map' in self.annotation['metadata']:
                self.class_map = self.annotation['metadata']['class-map']

        if 'bboxes' in self.annotation['annotations']:
            self.bboxes = self._get_bboxes()
        else:
            self.bboxes = []

        if 'keypoints' in self.annotation['annotations']:
            self.keypoints = self._get_keypoints()
        else:
            self.keypoints = []

    def _get_keypoints(self):
        keypoints = self.annotation['annotations']['keypoints']
        return [Keypoint(class_id=kpt['class_id'],
                         x=kpt['x'],
                         y=kpt['y'],
                         visible=kpt['visible'],
                         class_map=self.class_map) for kpt in keypoints]

    def _get_bboxes(self):
        bboxes = self.annotation['annotations']['bboxes']
        boxes = []
        for bbox in bboxes:
            bbox_obj = BBox(class_id=bbox['class_id'],
                            bbox=(bbox['left'], bbox['top'], bbox['right'], bbox['bottom']),
                            class_map=self.class_map)

            for key in bbox.keys():
                if hasattr(bbox_obj, key):
                    setattr(bbox_obj, key, bbox[key])

            boxes.append(bbox_obj)

        return boxes
