import cv2
from solartf.classifier.generator import ClassifierDirectoryGenerator
from solartf.data.image.processor import ImageAugmentation


class Config:
    IMAGE_DIR = '/Users/huangshangyu/Downloads/experiment/maskimage/tmp/train'
    IMAGE_TYPE = 'bgr'
    IMAGE_SHAPE = (512, 512, 3)

    BRIGHTNESS_RATIO = (.5, 2.)
    FLIP_ORIENTATION = 'horizontal_random'
    SCALE_RATIO = (.8, 1.2)
    DEGREE = (-5., 5.)
    H_SHIFT = (-10, 10)
    V_SHIFT = (-10, 10)


if __name__ == '__main__':
    config = Config()
    augment = ImageAugmentation(brightness_ratio=config.BRIGHTNESS_RATIO,
                                flip_orientation=config.FLIP_ORIENTATION,
                                scale_ratio=config.SCALE_RATIO,
                                degree=config.DEGREE,
                                h_shift=config.H_SHIFT,
                                v_shift=config.V_SHIFT)

    for image_input_list, _ in ClassifierDirectoryGenerator(image_dir=config.IMAGE_DIR,
                                                            image_type=config.IMAGE_TYPE,
                                                            image_shape=config.IMAGE_SHAPE,
                                                            dataset_type='test'):

        augment.execute(image_input_list=image_input_list)
        for image_input in image_input_list:
            cv2.imshow('Augment', image_input.image_array)

        key = cv2.waitKey(5000)
        if key == ord('q') or key == 27:
            break
