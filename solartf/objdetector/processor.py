import copy
import numpy as np
from typing import List
from solartf.data.image.processor import ImageInput
from solartf.data.bbox.processor import BBoxeInput
from solartf.data.bbox.type import BBox
from .type import DetectionInput


class MosaicComponent:
    def __init__(self, section_index, image_array, bboxes, labels, xmin, ymin, xmax, ymax):
        self.section_index = section_index
        self.image_array = image_array
        self.bboxes = bboxes
        self.labels = labels
        # Coordinates of the position on a mosaic image
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


class MosaicImageAugmentation:
    def __init__(self, n_slice, image_shape):
        """
        :param n_slice: the number of images will be sliced along an axis,
        and the total sliced images will be n_slice^2
        :param image_shape: (height, width), a shape of mixed images
        """
        self.n_slice = n_slice
        if len(image_shape) >= 3:
            self.height, self.width, self.depth = image_shape[:3]
        else:
            raise ValueError(f'The size of image_shape must be equal or larger than 2...')

    def execute(self,
                image_input_list: List[ImageInput],
                detection_input_list: List[DetectionInput]):

        n_batch = len(image_input_list)
        if not n_batch == len(detection_input_list):
            raise ValueError(f'The sizes of image_input_list and detection_gt_input must be equal...')

        mosaic_components = self.gen_mosaic_components(image_input_list, detection_input_list)
        mosaic_images, mosaic_bboxes, mosaic_labels = self.gen_mosaic_images(
            shape=(n_batch, self.height, self.width, self.depth),
            components=mosaic_components)

        for image_input, mosaic_image in zip(image_input_list, mosaic_images):
            image_input.image_array = mosaic_image

        bbox_list = []
        for detection_input, mosaic_bbox, mosaic_label in zip(detection_input_list, mosaic_bboxes, mosaic_labels):
            for label, bbox in zip(mosaic_label, mosaic_bbox):
                bbox_list.append(BBox(class_id=label, bbox=bbox))

            detection_input.bboxes_input = BBoxeInput(bbox_list)

    def gen_mosaic_components(self,
                              image_input_list: List[ImageInput],
                              detection_input_list: List[DetectionInput]):

        x_intersect = np.random.randint(0, self.width - 1, self.n_slice - 1)
        y_intersect = np.random.randint(0, self.height - 1, self.n_slice - 1)
        x_intersect = np.concatenate([[0], x_intersect, [self.width]])
        y_intersect = np.concatenate([[0], y_intersect, [self.height]])

        mosaic_components = []
        for image_input, detection_input in zip(image_input_list, detection_input_list):
            for xindex in range(self.n_slice):
                for yindex in range(self.n_slice):
                    xmin = x_intersect[xindex]
                    xmax = x_intersect[xindex + 1]
                    ymin = y_intersect[yindex]
                    ymax = y_intersect[yindex + 1]

                    if (self.width - xmax + xmin - 1) > 0:
                        x_anchor = np.random.randint(0, self.width - xmax + xmin - 1)
                    else:
                        x_anchor = 0

                    if (self.height - ymax + ymin - 1) > 0:
                        y_anchor = np.random.randint(0, self.height - ymax + ymin - 1)
                    else:
                        y_anchor = 0

                    section_index = yindex + xindex * self.n_slice

                    image_input = copy.deepcopy(image_input)
                    detection_input = copy.deepcopy(detection_input)

                    image_array = image_input.crop(xmin=x_anchor,
                                                   ymin=y_anchor,
                                                   xmax=x_anchor + xmax - xmin,
                                                   ymax=y_anchor + ymax - ymin)

                    detection_input.bboxes_input.crop(xmin=x_anchor,
                                                      ymin=y_anchor,
                                                      xmax=x_anchor + xmax - xmin,
                                                      ymax=y_anchor + ymax - ymin)
                    for bbox in detection_input.bboxes_input.bboxes_tensor:
                        bbox[..., 0] = bbox[..., 0] - x_anchor + xmin
                        bbox[..., 1] = bbox[..., 1] - y_anchor + ymin
                        bbox[..., 2] = bbox[..., 2] - x_anchor + xmin
                        bbox[..., 3] = bbox[..., 3] - y_anchor + ymin

                    mosaic_components.append(MosaicComponent(section_index=section_index,
                                                             image_array=image_array,
                                                             bboxes=detection_input.bboxes_input.bboxes_tensor.copy(),
                                                             labels=detection_input.bboxes_input.labels.copy(),
                                                             xmin=xmin,
                                                             ymin=ymin,
                                                             xmax=xmax,
                                                             ymax=ymax))
        np.random.shuffle(mosaic_components)
        return mosaic_components

    def gen_mosaic_images(self, shape, components):
        section_map = {index: 0 for index in range(self.n_slice ** 2)}
        mosaic_images = np.zeros(shape=shape)
        mosaic_bboxes = [[] for _ in range(shape[0])]
        mosaic_labels = [[] for _ in range(shape[0])]
        for component in components:
            batch_index = section_map[component.section_index]
            mosaic_images[batch_index, component.ymin:component.ymax, component.xmin:component.xmax, :] \
                = component.image_array
            if component.bboxes.any():
                mosaic_bboxes[batch_index].append(component.bboxes)
                mosaic_labels[batch_index].append(component.labels)

            section_map[component.section_index] += 1

        mosaic_bboxes = [np.concatenate(bbox, axis=0) if len(bbox) else np.zeros(shape=(1, 4))
                         for bbox in mosaic_bboxes]
        mosaic_labels = [np.concatenate(labels, axis=0) if len(labels) else np.zeros(shape=(1,))
                         for labels in mosaic_labels]
        return mosaic_images, mosaic_bboxes, mosaic_labels


class BBoxesAugmentation:
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
                bbox_input_list: List[BBoxeInput]):

        for image_input, bbox_input in zip(image_input_list, bbox_input_list):
            image_type = image_input.image_type

            if self.brightness_ratio is not None:
                image_input.brightness(ratio=self.brightness_ratio,
                                       image_type=image_type)

            if self.scale_ratio is not None:
                transfer_matrix = image_input.rescale(ratio=self.scale_ratio)
                bbox_input.affine(transfer_matrix=transfer_matrix)

            if self.flip_orientation is not None:
                orientation = image_input.flip(orientation=self.flip_orientation)
                bbox_input.flip(image_input.image_shape, orientation=orientation)

            if self.degree is not None:
                transfer_matrix = image_input.rotate(degree=self.degree)
                bbox_input.affine(transfer_matrix=transfer_matrix)

            if self.h_shift is not None or self.v_shift is not None:
                transfer_matrix = image_input.translate(horizontal=self.h_shift, vertical=self.v_shift)
                bbox_input.affine(transfer_matrix=transfer_matrix)
