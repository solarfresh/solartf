import os
from typing import Tuple
from solartf.core.iterator import DirectoryIterator
from solartf.data.parser.label import LabelReader
from .type import DetectionInput


class DetectionLabelIterator(DirectoryIterator):

    def __init__(self,
                 image_dir,
                 label_dir,
                 extension: Tuple = ('.json',),
                 shuffle=True,
                 bbox_exclude=None):
        super().__init__(label_dir, extension, shuffle)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.bbox_exclude = bbox_exclude

    def __next__(self):
        self.item_index += 1
        if self.item_index < self.item_count:
            image_id = self.index_array[self.item_index]
            file_path = self.file_path_list[image_id]

            label_reader = LabelReader(file_path)
            image_prop = label_reader.image_prop
            bboxes_prop = label_reader.bboxes

            image_prop.source_ref = os.path.join(self.image_dir, image_prop.source_ref)

            return DetectionInput(image_prop,
                                  bboxes_prop,
                                  bbox_exclude=self.bbox_exclude)

        raise StopIteration
