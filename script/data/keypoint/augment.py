import cv2
from solartf.kptdetector.processor import KeypointAugmentation
from solartf.kptdetector.generator import KeypointDirectoryGenerator


class Config:
    IMAGE_DIR = '/Users/huangshangyu/Downloads/experiment/facekpt/image'
    LABEL_DIR = '/Users/huangshangyu/Downloads/experiment/facekpt/annotation'
    IMAGE_TYPE = 'bgr'
    IMAGE_SHAPE = (500, 500, 3)

    BRIGHTNESS_RATIO = (.5, 2.)
    FLIP_ORIENTATION = 'horizontal_random'
    SCALE_RATIO = (.8, 1.2)
    DEGREE = (-5., 5.)
    H_SHIFT = (-10, 10)
    V_SHIFT = (-10, 10)


if __name__ == '__main__':
    config = Config()
    augment = KeypointAugmentation(brightness_ratio=config.BRIGHTNESS_RATIO,
                                   flip_orientation=config.FLIP_ORIENTATION,
                                   scale_ratio=config.SCALE_RATIO,
                                   degree=config.DEGREE,
                                   h_shift=config.H_SHIFT,
                                   v_shift=config.V_SHIFT)

    for image_input_list, kpt_input_list in KeypointDirectoryGenerator(image_dir=config.IMAGE_DIR,
                                                                       label_dir=config.LABEL_DIR,
                                                                       image_shape=config.IMAGE_SHAPE,
                                                                       image_type=config.IMAGE_TYPE,):

        augment.execute(image_input_list=image_input_list,
                        kpt_input_list=kpt_input_list)
        for image_input, kpt_input in zip(image_input_list, kpt_input_list):
            image_array = image_input.image_array.copy()
            for kpt in kpt_input.points_tensor:
                image_array = cv2.circle(image_array, tuple(kpt), radius=0, color=(0, 0, 255), thickness=10)

            cv2.imshow('Augment', image_array)

        key = cv2.waitKey(5000)
        if key == ord('q') or key == 27:
            break
