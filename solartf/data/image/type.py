import cv2


class ImageInput:
    def __init__(self,
                 image_id,
                 image_path,
                 image_type='rgb',
                 image_shape=None,
                 mode=None):
        self.image_id = image_id
        self.image_path = image_path
        self.image_type = image_type.lower()
        self.image_shape = image_shape
        self.scale = (1, 1)

        # The shape of image_array returns (height, width, depth)
        if mode is not None:
            self.image_array = cv2.imread(self.image_path, mode)
        else:
            self.image_array = cv2.imread(self.image_path)

        if self.image_type == 'rgb':
            self.image_array = cv2.cvtColor(self.image_array, cv2.COLOR_BGR2RGB)

        if self.image_type == 'gray':
            self.image_array = cv2.cvtColor(self.image_array, cv2.COLOR_BGR2GRAY)

        if self.image_type == 'hsv':
            self.image_array = cv2.cvtColor(self.image_array, cv2.COLOR_BGR2HSV)

        if self.image_shape is not None:
            self.scale = tuple(self.image_array.shape[index] / image_shape[index] for index in range(2))
            self.image_array = cv2.resize(self.image_array, self.image_shape[:2])

    def convert(self, image_type):
        if self.image_type == 'bgr':
            if image_type == 'bgr':
                return self.image_array
            elif image_type == 'rgb':
                return cv2.cvtColor(self.image_array, cv2.COLOR_BGR2RGB)
            elif image_type == 'gray':
                return cv2.cvtColor(self.image_array, cv2.COLOR_BGR2GRAY)
            elif image_type == 'hsv':
                return cv2.cvtColor(self.image_array, cv2.COLOR_BGR2HSV)
        elif self.image_type == 'rgb':
            if image_type == 'bgr':
                return cv2.cvtColor(self.image_array, cv2.COLOR_RGB2BGR)
            elif image_type == 'rgb':
                return self.image_array
            elif image_type == 'gray':
                return cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
            elif image_type == 'hsv':
                return cv2.cvtColor(self.image_array, cv2.COLOR_RGB2HSV)
        elif self.image_type == 'hsv':
            if image_type == 'bgr':
                return cv2.cvtColor(self.image_array, cv2.COLOR_HSV2BGR)
            elif image_type == 'rgb':
                return cv2.cvtColor(self.image_array, cv2.COLOR_HSV2RGB)
            elif image_type == 'gray':
                image_array = cv2.cvtColor(self.image_array, cv2.COLOR_HSV2BGR)
                return cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            elif image_type == 'hsv':
                return self.image_array
        else:
            raise ValueError(f'The type {image_type} can be converted from {self.image_type}')

    def resize(self, dsize):
        return cv2.resize(self.image_array, dsize)
