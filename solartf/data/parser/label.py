import json
from solartf.data.image.type import Image
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
            if self.image_prop.__getattribute__(key):
                self.image_prop.__setattr__(key, value)

        if 'keypoints' in self.annotation['annotations']:
            self.keypoints = self._get_keypoints()
        else:
            self.keypoints = []

    def _get_keypoints(self):
        keypoints = self.annotation['annotations']['keypoints']
        return [Keypoint(class_id=kpt['class_id'], x=kpt['x'], y=kpt['y']) for kpt in keypoints]
