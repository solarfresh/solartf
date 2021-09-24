import cv2
from solartf.kptdetector.processor import (KeypointAugmentation, RandomOcclusion)
from solartf.kptdetector.generator import KeypointDirectoryGenerator


class Config:
    IMAGE_DIR = '/Users/huangshangyu/Downloads/experiment/facekpt/image'
    LABEL_DIR = '/Users/huangshangyu/Downloads/experiment/facekpt/annotation'
    IMAGE_TYPE = 'bgr'
    IMAGE_SHAPE = (640, 640, 3)

    BRIGHTNESS_RATIO = (.5, 2.)
    FLIP_ORIENTATION = None
    SCALE_RATIO = None
    DEGREE = (-1., 1.)
    H_SHIFT = (-1, 1)
    V_SHIFT = (-1, 1)
    ANGLE_SCALE = 0.1
    IRREGULARITY = 0.01
    SPIKEYNESS = 0.01

    AUGMENT = [
        KeypointAugmentation(brightness_ratio=BRIGHTNESS_RATIO,
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
    for image_input_list, kpt_input_list in KeypointDirectoryGenerator(image_dir=config.IMAGE_DIR,
                                                                       label_dir=config.LABEL_DIR,
                                                                       image_shape=config.IMAGE_SHAPE,
                                                                       image_type=config.IMAGE_TYPE,
                                                                       dataset_type='test',
                                                                       augment=config.AUGMENT):

        for image_input, kpt_input in zip(image_input_list, kpt_input_list):
            image_array = image_input.image_array.copy()
            for idx in kpt_input.indexes:
                kpt = kpt_input.points_tensor[idx]
                if kpt_input.labels[idx]:
                    image_array = cv2.circle(image_array, tuple(kpt), radius=0, color=(0, 0, 255), thickness=10)

            cv2.imshow('Augment', image_array)

        key = cv2.waitKey(5000)
        if key == ord('q') or key == 27:
            break
