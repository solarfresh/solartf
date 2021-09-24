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
    def __init__(self, n_slice):
        """
        :param n_slice: the number of images will be sliced along an axis,
        and the total sliced images will be n_slice^2
        :param image_shape: (height, width), a shape of mixed images
        """
        self.n_slice = n_slice
        self.width = 0
        self.height = 0
        self.class_map = {}

    def execute(self,
                image_input_list: List[ImageInput],
                bboxes_input_list: List[BBoxeInput]):

        n_batch = len(image_input_list)
        if not n_batch == len(bboxes_input_list):
            raise ValueError(f'The sizes of image_input_list and detection_gt_input must be equal...')

        if n_batch > 0:
            image_shape = image_input_list[0].image_shape
            self.height, self.width, _ = image_shape
        else:
            raise ValueError(f'The batch size must be equal or larger than 1...')

        mosaic_components = self.gen_mosaic_components(image_input_list, bboxes_input_list)
        mosaic_images, mosaic_bboxes, mosaic_labels = self.gen_mosaic_images(
            shape=(n_batch,) + image_shape,
            components=mosaic_components)

        for image_input, mosaic_image in zip(image_input_list, mosaic_images):
            image_input.image_array = mosaic_image

        for bboxes_input, mosaic_bbox, mosaic_label in zip(bboxes_input_list, mosaic_bboxes, mosaic_labels):
            bbox_list = []
            for label, bbox in zip(mosaic_label, mosaic_bbox):
                bbox_list.append(BBox(class_id=label, bbox=bbox, class_map=self.class_map))

            bboxes_input.__init__(bbox_list, bbox_exclude=bboxes_input.bbox_exclude)

    def gen_mosaic_components(self,
                              image_input_list: List[ImageInput],
                              bboxes_input_list: List[BBoxeInput]):

        x_intersect = np.sort(np.random.randint(0, self.width - 1, self.n_slice - 1))
        y_intersect = np.sort(np.random.randint(0, self.height - 1, self.n_slice - 1))
        x_intersect = np.concatenate([[0], x_intersect, [self.width]])
        y_intersect = np.concatenate([[0], y_intersect, [self.height]])

        mosaic_components = []
        for image_input, bboxes_input in zip(image_input_list, bboxes_input_list):
            if len(self.class_map) < 1 and len(bboxes_input.bboxes) > 0:
                self.class_map = bboxes_input.bboxes[0].class_map

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

                    cropped_image_input = copy.deepcopy(image_input)
                    cropped_bboxes_input = copy.deepcopy(bboxes_input)

                    cropped_image_input.crop(xmin=x_anchor,
                                             ymin=y_anchor,
                                             xmax=x_anchor + xmax - xmin,
                                             ymax=y_anchor + ymax - ymin)

                    cropped_bboxes_input.crop(xmin=x_anchor,
                                              ymin=y_anchor,
                                              xmax=x_anchor + xmax - xmin,
                                              ymax=y_anchor + ymax - ymin)

                    bboxes = cropped_bboxes_input.bboxes_tensor.to_array(coord='corner').copy()
                    for bbox in bboxes:
                        bbox[..., 0] = bbox[..., 0] - x_anchor + xmin
                        bbox[..., 1] = bbox[..., 1] - y_anchor + ymin
                        bbox[..., 2] = bbox[..., 2] - x_anchor + xmin
                        bbox[..., 3] = bbox[..., 3] - y_anchor + ymin

                    mosaic_components.append(MosaicComponent(section_index=section_index,
                                                             image_array=cropped_image_input.image_array.copy(),
                                                             bboxes=bboxes,
                                                             labels=cropped_bboxes_input.labels.copy(),
                                                             xmin=xmin,
                                                             ymin=ymin,
                                                             xmax=xmax,
                                                             ymax=ymax))
        np.random.shuffle(mosaic_components)
        return mosaic_components

    def gen_mosaic_images(self, shape, components):
        section_map = {index: 0 for index in range(self.n_slice ** 2)}
        mosaic_images = np.zeros(shape=shape).astype(np.uint8)
        mosaic_bboxes = [[] for _ in range(shape[0])]
        mosaic_labels = [[] for _ in range(shape[0])]
        for component in components:
            batch_index = section_map[component.section_index]
            mosaic_images[batch_index, component.ymin:component.ymax, component.xmin:component.xmax] \
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
                 v_shift=None,
                 angle_scale=None,
                 irregularity=None,
                 spikeyness=None):
        self.brightness_ratio = brightness_ratio
        self.flip_orientation = flip_orientation
        self.scale_ratio = scale_ratio
        self.degree = degree
        self.h_shift = h_shift
        self.v_shift = v_shift
        self.angle_scale = angle_scale
        self.irregularity = irregularity
        self.spikeyness = spikeyness

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

            if self.angle_scale is not None or self.irregularity is not None or self.spikeyness is not None:
                transfer_matrix = image_input.perspective(
                    angle_scale=self.angle_scale,
                    irregularity=self.irregularity,
                    spikeyness=self.spikeyness)
                bbox_input.perspective(transfer_matrix=transfer_matrix)
