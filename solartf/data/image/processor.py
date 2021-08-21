import cv2
import numpy as np
from .type import ImageInput


class ImageProcessor:

    @staticmethod
    def affine(image: np.array, origin_pts=None, horizontal=None, vertical=None):

        height, width, depth = image.shape

        if origin_pts is None:
            origin_pts = tuple([np.random.randint(0, width), np.random.randint(0, height)] for _ in range(3))

        if isinstance(horizontal, tuple) or isinstance(horizontal, list):
            dw = tuple(np.random.randint(horizontal[0], horizontal[1]+1) for _ in range(3))
        elif horizontal is None:
            dw = (0, 0, 0)
        else:
            dw = horizontal

        if isinstance(vertical, tuple) or isinstance(vertical, list):
            dh = tuple(np.random.randint(vertical[0], vertical[1]+1) for _ in range(3))
        elif horizontal is None:
            dh = (0, 0, 0)
        else:
            dh = vertical

        if (not len(origin_pts) == 3) or (not len(horizontal) == 3) or (not len(vertical) == 3):
            raise ValueError('There must be 3 points provided...')

        target_pts = tuple()
        for origin_pt, dx, dy in zip(origin_pts, dw, dh):
            _x = origin_pts[0] + dx
            _x = width - 1 if _x >= width else _x
            _x = 0 if _x < 0 else _x

            _y = origin_pts[1] + dy
            _y = height - 1 if _y >= height else _y
            _y = 0 if _y < 0 else _y

            target_pts += ([_x, _y],)

        M = cv2.getAffineTransform(np.float32(origin_pts), np.float32(target_pts))
        return cv2.warpAffine(image, M, (width, height))

    @staticmethod
    def brightness(image: np.array, ratio=None, image_type='rgb'):
        if image_type == 'rgb':
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif image_type == 'bgr':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError(f'{image_type} is not implemented...')

        if ratio is None:
            ratio = (.5, 2.)

        if isinstance(ratio, tuple) or isinstance(ratio, list):
            factor = np.random.uniform(ratio[0], ratio[1])
        else:
            factor = ratio

        mask = hsv[..., 2] * factor > 255
        v_channel = np.where(mask, 255, hsv[:, :, 2] * factor)
        hsv[..., 2] = v_channel

        if image_type == 'rgb':
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if image_type == 'bgr':
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def equalizer(self,
                  image: np.array,
                  method='hist',
                  image_type='rgb',
                  clip_limit=None,
                  tile_grid_size_width=None,
                  tile_grid_size_height=None):
        if method not in ['hist', 'clahe']:
            raise ValueError(f'{method} is not implemented...')

        image_copied = np.copy(image)
        if method == 'hist':
            return self._hist_equalizer(image_copied, image_type=image_type)
        if method == 'clahe':
            if clip_limit is None:
                clip_limit = (1. , 2.)

            if tile_grid_size_width is None:
                tile_grid_size_width = (6, 10)

            if tile_grid_size_height is None:
                tile_grid_size_height = (6, 10)

            if isinstance(clip_limit, tuple) or isinstance(clip_limit, list):
                clip_factor = np.random.uniform(clip_limit[0], clip_limit[1])
            else:
                clip_factor = clip_limit

            if isinstance(tile_grid_size_width, tuple) or isinstance(tile_grid_size_width, list):
                width_size = np.random.randint(tile_grid_size_width[0], tile_grid_size_width[1])
            else:
                width_size = tile_grid_size_width

            if isinstance(tile_grid_size_height, tuple) or isinstance(tile_grid_size_height, list):
                height_size = np.random.randint(tile_grid_size_height[0], tile_grid_size_height[1])
            else:
                height_size = tile_grid_size_height

            return self._clahe(image_copied, clip_limit=clip_factor, tile_grid_size=(height_size, width_size))

    @staticmethod
    def flip(image: np.array, orientation=None):
        """
        :param orientation: horizontal, vertical, random, horizontal_random, vertical_random
        """
        if orientation == 'vertical':
            return cv2.flip(image, 0), orientation

        if orientation == 'horizontal':
            return cv2.flip(image, 1), orientation

        if orientation == 'horizontal_random':
            if np.random.randint(0, 2):
                return cv2.flip(image, 1), 'horizontal'
            else:
                return image, None

        if orientation == 'vertical_random':
            if np.random.randint(0, 2):
                return cv2.flip(image, 0), 'vertical'
            else:
                return image, None

        return image, None

    @staticmethod
    def resize_image(image: np.array, size, keep_aspect_ratio=False):
        h, w = image.shape[:2]

        if not keep_aspect_ratio:
            resized_frame = cv2.resize(image, size)
            scale = (size[0] / w, size[1] / h)
        else:
            scale = min(size[1] / h, size[0] / w)
            resized_frame = cv2.resize(image, None, fx=scale, fy=scale)
            scale = (scale, scale)
        return resized_frame, scale

    @staticmethod
    def rotate(image: np.array, degree=None):
        height, width, depth = image.shape
        if isinstance(degree, tuple) or isinstance(degree, list):
            factor = np.random.uniform(degree[0], degree[1])
        else:
            factor = degree

        M = cv2.getRotationMatrix2D((width / 2, height / 2), factor, 1)
        return cv2.warpAffine(image, M, (width, height)), M

    @staticmethod
    def scale(image: np.array, ratio=None):
        height, width, depth = image.shape

        if isinstance(ratio, tuple) or isinstance(ratio, list):
            factor = np.random.uniform(ratio[0], ratio[1])
        else:
            factor = ratio

        M = cv2.getRotationMatrix2D((width / 2, height / 2), 0, factor)
        # M will be adopted when processing bboxes
        return cv2.warpAffine(image, M, (width, height)), M

    @staticmethod
    def translate(image: np.array, horizontal=None, vertical=None):
        height, width, depth = image.shape

        if isinstance(horizontal, tuple) or isinstance(horizontal, list):
            dw = np.random.randint(horizontal[0], horizontal[1]+1)
        elif horizontal is None:
            dw = 0
        else:
            dw = horizontal

        if isinstance(vertical, tuple) or isinstance(vertical, list):
            dh = np.random.randint(vertical[0], vertical[1]+1)
        elif vertical is None:
            dh = 0
        else:
            dh = vertical

        # create the transformation matrix
        M = np.float32([[1, 0, dw], [0, 1, dh]])

        # todo: there will be the empty black space, and we can crop that

        return cv2.warpAffine(image, M, (width, height)), M

    @staticmethod
    def _hist_equalizer(image: np.array, image_type='rgb'):
        if image_type == 'rgb':
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif image_type == 'bgr':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError(f'{image_type} is not implemented...')

        hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])

        if image_type == 'rgb':
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if image_type == 'bgr':
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def _clahe(image: np.array, clip_limit=2.0, tile_grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        if len(image.shape) == 2:
            return clahe.apply(image)
        elif len(image.shape) == 3:
            for channel in range(3):
                image[..., channel] = clahe.apply(image[..., channel])

            return image
        else:
            return image


class ImageAugmentation:
    def __init__(self,
                 brightness_ratio=None,
                 flip_orientation=None,
                 scale_ratio=None,
                 degree=None,
                 h_shift=None,
                 v_shift=None):
        self.brightness_ratio = brightness_ratio
        self.flip_orientation = flip_orientation
        self.scale_ratio = scale_ratio
        self.degree = degree
        self.h_shift = h_shift
        self.v_shift = v_shift
        self.image_processor = ImageProcessor()

    def execute(self, image_input: ImageInput):
        image_array = image_input.image_array
        image_type = image_input.image_type

        if self.brightness_ratio is not None:
            image_array = self.image_processor.brightness(image_array,
                                                          ratio=self.brightness_ratio,
                                                          image_type=image_type)
        if self.flip_orientation is not None:
            image_array, orientation = self.image_processor.flip(image_array, orientation=self.flip_orientation)

        if self.scale_ratio is not None:
            image_array, M = self.image_processor.scale(image_array, ratio=self.scale_ratio)

        if self.degree is not None:
            image_array, M = self.image_processor.rotate(image_array, degree=self.degree)

        if self.h_shift is not None or self.v_shift is not None:
            image_array, M = self.image_processor.translate(image_array, horizontal=self.h_shift, vertical=self.v_shift)

        image_input.image_array = image_array
