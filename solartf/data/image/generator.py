import os
from solartf.core.generator import KerasGeneratorBase
from solartf.data.image.processor import ImageInput


class ImageDirectoryGenerator(KerasGeneratorBase):
    def __init__(self,
                 image_dir,
                 image_type,
                 image_shape,
                 image_format=None):
        if image_format is None:
            image_format = ('.png', '.jpg', '.jpeg')

        self.image_dir = image_dir
        self.image_type = image_type
        self.image_shape = image_shape
        self.image_format = image_format

        self.image_path_list = [os.path.join(root, fname) for root, _, fnames in os.walk(self.image_dir)
                                for fname in fnames if fname.endswith(image_format)]

    def __len__(self):
        return len(self.image_path_list)

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        input_image_path = self.image_path_list[index]
        image_input = ImageInput(input_image_path,
                                 image_type=self.image_type,
                                 image_shape=self.image_shape)
        return image_input
