import numpy as np
import tensorflow as tf
from solartf.core.output import OutputBase
from solartf.data.bbox.type import BBoxesTensor


class BBoxOutput(OutputBase):
    def __init__(self, method='iou', normalize=False, package='numpy'):
        # todo: AnchorBox layer now is fixed in corner coordinates now, and we will modify it to be more flexible
        self.method = method
        self.normalize = normalize
        self.package = package

        if self.package == 'numpy':
            self._concat = np.concatenate

        if self.package == 'tensorflow':
            self._concat = tf.concat

    def encoder(self, outputs: np.array):
        bboxes = BBoxesTensor(outputs[..., :4], method=self.package)
        anchors = BBoxesTensor(outputs[..., 4:8], method=self.package)
        variance = outputs[..., 8:]

        if self.method == 'iou':
            return self._iou_encoder(bboxes=bboxes, anchors=anchors, variance=variance)
        elif self.method == 'centroid':
            return self._centroid_encoder(bboxes=bboxes, anchors=anchors, variance=variance)
        elif self.method == 'corner':
            return self._corner_encoder(bboxes=bboxes, anchors=anchors, variance=variance)
        else:
            raise ValueError(f'The method {self.method} does not support yet...')

    def decoder(self, outputs: np.array):
        bboxes = outputs[..., :4]
        anchors = outputs[..., 4:8]
        variance = outputs[..., 8:]

        if self.method == 'iou':
            return self._iou_decoder(bboxes, anchors, variance)
        if self.method == 'centroid':
            return self._centroid_decoder(bboxes, anchors, variance)
        elif self.method == 'corner':
            return self._corner_decoder(bboxes, anchors, variance)
        else:
            raise ValueError(f'The method {self.method} does not support yet...')

    def _centroid_encoder(self, bboxes, anchors, variance):
        bboxes_array = bboxes.to_array(coord='centroid')
        anchors_array = anchors.to_array(coord='centroid')
        center = bboxes_array[..., :2] - anchors_array[..., :2]
        center /= anchors_array[..., 2:]
        # The value in log is forced to be positive, and it is supposed that
        # data is clean enough and there are only a few errors coming from augmentation
        size = np.log(np.abs(bboxes_array[..., 2:] / (anchors_array[..., 2:])))
        center /= variance[..., :2]
        size /= variance[..., 2:]
        return self._concat((center, size, anchors.to_array(coord='corner'), variance), axis=-1)

    def _centroid_decoder(self, bboxes, anchors, variance):
        anchors = BBoxesTensor(anchors, coord='corner').to_array(coord='centroid')
        bboxes_decoded = np.zeros_like(bboxes)
        bboxes_decoded[..., :2] = bboxes[..., :2] * anchors[..., 2:] * variance[..., :2] + anchors[..., :2]
        bboxes_decoded[..., 2:] = np.exp(bboxes[..., 2:] * variance[..., 2:]) * anchors[..., 2:]
        return BBoxesTensor(bboxes_decoded, coord='centroid', method=self.package)

    def _corner_encoder(self, bboxes, anchors, variance):
        bboxes_array = bboxes.to_array(coord='corner')
        anchors_array = anchors.to_array(coord='corner')

        if self.normalize:
            anchor_wh = anchors_array[..., 2:] - anchors_array[..., :2]
            anchor_wh = self._concat((anchor_wh, anchor_wh), axis=-1)
            bboxes_encoded = (bboxes_array - anchors_array) / anchor_wh / variance
        else:
            bboxes_encoded = (bboxes_array - anchors_array) / variance
        output_encoded = self._concat((bboxes_encoded, anchors.to_array(coord='corner'), variance), axis=-1)

        return output_encoded

    def _corner_decoder(self, bboxes, anchors, variance):
        if self.normalize:
            anchor_wh = anchors[..., 2:] - anchors[..., :2]
            anchor_wh = self._concat((anchor_wh, anchor_wh), axis=-1)

            bboxes_decoded = \
                bboxes * anchor_wh * variance + anchors
        else:
            bboxes_decoded = \
                bboxes * variance + anchors

        return BBoxesTensor(bboxes_decoded, coord='corner', method=self.package)

    def _iou_encoder(self, bboxes, anchors, variance):
        bboxes_array = bboxes.to_array(coord='corner')
        anchors_array = anchors.to_array(coord='corner')

        bboxes_encoded = bboxes_array - self._concat((anchors_array[..., :2], anchors_array[..., :2]), axis=-1)
        if self.normalize:
            anchor_wh = anchors_array[..., 2:] - anchors_array[..., :2]
            bboxes_encoded = bboxes_encoded / self._concat((anchor_wh, anchor_wh), axis=-1)

        bboxes_encoded = bboxes_encoded / variance
        return self._concat((bboxes_encoded, anchors.to_array(coord='corner'), variance), axis=-1)

    def _iou_decoder(self, bboxes, anchors, variance):
        bboxes_decoded = bboxes * variance
        if self.normalize:
            anchor_wh = anchors[..., 2:] - anchors[..., :2]
            bboxes_decoded = bboxes_decoded * self._concat((anchor_wh, anchor_wh), axis=-1)

        bboxes_decoded = bboxes_decoded + self._concat((anchors[..., :2], anchors[..., :2]), axis=-1)
        return BBoxesTensor(bboxes_decoded, coord='corner', method=self.package)


class GridAnchor:
    image_shape = None
    feature_map_shape = None

    def generate_anchors(self,
                         image_shape,
                         feature_map_shape,
                         aspect_ratios,
                         scale: float,
                         step_shape=None,
                         offset_shape=None):

        self.image_shape = image_shape
        self.feature_map_shape = feature_map_shape

        boxes_per_cell = self._generate_boxes_per_cell(aspect_ratios=aspect_ratios,
                                                       scale=scale)

        # the last will contain `(cx, cy, w, h)`
        anchor_tensor = self._generate_center_points(boxes_per_cell=boxes_per_cell,
                                                     step_shape=step_shape,
                                                     offset_shape=offset_shape)

        return anchor_tensor

    def _generate_boxes_per_cell(self,
                                 scale: float,
                                 aspect_ratios: List[float]):
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        base_size = min(self.image_shape[:2]) * scale

        # Compute the box widths and and heights for all aspect ratios
        boxes_per_cell = []
        for ar in aspect_ratios:
            if isinstance(ar, float):
                sqrt_ar = np.sqrt(ar)
                ar_w = sqrt_ar
                ar_h = 1. / sqrt_ar
            else:
                ar_w, ar_h = ar

            boxes_per_cell.append((ar_w * base_size, ar_h * base_size))

        return np.array(boxes_per_cell)

    def _generate_center_points(self,
                                boxes_per_cell,
                                step_shape,
                                offset_shape=None):

        n_boxes = len(boxes_per_cell)

        image_height, image_width = self.image_shape[:2]
        feature_map_height, feature_map_width = self.feature_map_shape

        if step_shape is None:
            step_height = image_height / feature_map_height
            step_width = image_width / feature_map_width
        elif isinstance(step_shape, float) or isinstance(step_shape, int):
            step_height = step_width = step_shape
        else:
            step_height, step_width = step_shape

        if offset_shape is None:
            offset_height = .5
            offset_width = .5
        elif isinstance(offset_shape, float) or isinstance(offset_shape, int):
            offset_height = offset_width = offset_shape
        else:
            offset_height, offset_width = offset_shape

        cy = np.arange(feature_map_height) + offset_height
        cy = cy * step_height
        cx = np.arange(feature_map_width) + offset_width
        cx = cx * step_width

        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, n_boxes, 4))

        # Set cx
        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))
        # Set cy
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))
        # Set w
        boxes_tensor[:, :, :, 2] = boxes_per_cell[:, 0]
        # Set h
        boxes_tensor[:, :, :, 3] = boxes_per_cell[:, 1]

        return boxes_tensor
