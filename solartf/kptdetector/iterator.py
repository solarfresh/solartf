import os
from typing import Tuple
from solartf.core.iterator import DirectoryIterator
from solartf.data.parser.label import LabelReader
from .type import KeypointDetectInput


class KeypointLabelIterator(DirectoryIterator):

    def __init__(self,
                 image_dir,
                 label_dir,
                 extension: Tuple = ('.json',),
                 shuffle=True):
        super().__init__(label_dir, extension, shuffle)
        self.image_dir = image_dir
        self.label_dir = label_dir

    def __next__(self):
        self.item_index += 1
        if self.item_index < self.item_count:
            image_id = self.index_array[self.item_index]
            file_path = self.file_path_list[image_id]

            label_reader = LabelReader(file_path)
            image_prop = label_reader.image_prop
            kpts_prop = label_reader.keypoints
            cls_prop = label_reader.classes
            image_prop.source_ref = os.path.join(self.image_dir, image_prop.source_ref)

            return KeypointDetectInput(image_prop, kpts_prop, cls_prop)

        raise StopIteration
