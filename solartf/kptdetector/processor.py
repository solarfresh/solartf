from typing import List
from solartf.data.image.processor import ImageInput
from solartf.data.keypoint.processor import KeypointInput


class KeypointAugmentation:
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

    def execute(self,
                image_input_list: List[ImageInput],
                kpt_input_list: List[KeypointInput]):

        for image_input, kpt_input in zip(image_input_list, kpt_input_list):
            image_type = image_input.image_type

            if self.brightness_ratio is not None:
                image_input.brightness(ratio=self.brightness_ratio,
                                       image_type=image_type)

            # if self.scale_ratio is not None:
            #     image_input.rescale(ratio=self.scale_ratio)
            #     kpt_input.resize(scale=image_input.scale)

            if self.flip_orientation is not None:
                orientation = image_input.flip(orientation=self.flip_orientation)
                kpt_input.flip(image_input.image_shape, orientation=orientation)

            if self.degree is not None:
                transfer_matrix = image_input.rotate(degree=self.degree)
                kpt_input.affine(transfer_matrix=transfer_matrix)

            if self.h_shift is not None or self.v_shift is not None:
                transfer_matrix = image_input.translate(horizontal=self.h_shift, vertical=self.v_shift)
                kpt_input.affine(transfer_matrix=transfer_matrix)
