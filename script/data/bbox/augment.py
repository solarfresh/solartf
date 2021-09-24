import cv2
import numpy as np
from solartf.objdetector.processor import (BBoxesAugmentation, MosaicImageAugmentation)
from solartf.objdetector.generator import DetectDirectoryGenerator


class Config:
    IMAGE_DIR = '/Users/huangshangyu/Downloads/processed_head_count_dataset/mask_faces_images'
    LABEL_DIR = '/Users/huangshangyu/Downloads/processed_head_count_dataset/label/train'
    IMAGE_TYPE = 'bgr'
    IMAGE_SHAPE = (500, 500, 3)

    BRIGHTNESS_RATIO = (.5, 2.)
    FLIP_ORIENTATION = 'horizontal_random'
    SCALE_RATIO = (.8, 1.2)
    DEGREE = (-5., 5.)
    H_SHIFT = (-10, 10)
    V_SHIFT = (-10, 10)
    ANGLE_SCALE = 0.1
    IRREGULARITY = 0.01
    SPIKEYNESS = 0.01

    AUGMENT = [
        MosaicImageAugmentation(n_slice=2),
        BBoxesAugmentation(brightness_ratio=BRIGHTNESS_RATIO,
                           flip_orientation=FLIP_ORIENTATION,
                           scale_ratio=SCALE_RATIO,
                           degree=DEGREE,
                           h_shift=H_SHIFT,
                           v_shift=V_SHIFT,
                           angle_scale=ANGLE_SCALE,
                           irregularity=IRREGULARITY,
                           spikeyness=SPIKEYNESS)
    ]


if __name__ == '__main__':
    config = Config()
    for image_input_list, bbox_input_list in DetectDirectoryGenerator(image_dir=config.IMAGE_DIR,
                                                                      label_dir=config.LABEL_DIR,
                                                                      image_shape=config.IMAGE_SHAPE,
                                                                      image_type=config.IMAGE_TYPE,
                                                                      dataset_type='test',
                                                                      augment=config.AUGMENT,
                                                                      bbox_exclude={'region_level': ['visualable', 'full']}):

        for image_input, bbox_input in zip(image_input_list, bbox_input_list):
            image_array = image_input.image_array.copy()
            bboxes_tensor = bbox_input.bboxes_tensor.to_array(coord='corner').astype(np.int32)
            for bbox in bboxes_tensor:
                cv2.rectangle(image_array,
                              (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]),
                              (0, 0, 255), 3)

            cv2.imshow('Augment', image_array)
            key = cv2.waitKey(5000)

        if key == ord('q') or key == 27:
            break
