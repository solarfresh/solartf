import logging
import numpy as np
import tensorflow as tf


class BBoxesMixin:
    bboxes: np.array
    method = 'numpy'

    def to_array(self, coord='corner'):
        if self.method == 'numpy':
            _stack = np.stack
        elif self.method == 'tensorflow':
            _stack = tf.stack
        else:
            ValueError(f'The method {self.method} is not supported...')

        if coord == 'corner':
            return self.bboxes
        elif coord == 'centroid':
            return _stack((self.width_centers, self.height_centers, self.widths, self.heights), axis=-1)
        elif coord == 'minmax':
            return _stack((self.tops, self.lefts, self.bottoms, self.rights), axis=-1)
        else:
            raise ValueError(f'The coordinate {coord} does not support...')

    def compute_iou_mask(self, boxes, overlap_threshold):
        iou = self.jaccard(boxes)
        logging.debug(f'Obtained IoU is {iou}')
        # make a mask of all "matched" predictions vs gt
        logging.debug(f'overlap_threshold is {overlap_threshold}...')
        return iou >= overlap_threshold

    def intersect_area(self, boxes):
        """
        Compute the area of intersection between two rectangular bounding box
        Bounding boxes use corner notation : [x1, y1, x2, y2]
        Args:
          box_a: (np.array) bounding boxes, Shape: [A,4].
          box_b: (np.array) bounding boxes, Shape: [B,4].
        Return:
          np.array intersection area, Shape: [A,B].
        """
        box_a = self.bboxes.copy()
        box_b = boxes.copy()
        logging.debug(f'shape of box_a is {box_a.shape}')
        logging.debug(f'shape of box_b is {box_b.shape}')
        resized_a = box_a[:, np.newaxis, :]
        resized_b = box_b[np.newaxis, :, :]
        max_xy = np.minimum(resized_a[:, :, 2:], resized_b[:, :, 2:])
        min_xy = np.maximum(resized_a[:, :, :2], resized_b[:, :, :2])

        diff_xy = (max_xy - min_xy)
        # although clip is faster, it will be wrong when diff_xy are negative
        inter = np.maximum(diff_xy, 0.)
        # inter = np.clip(diff_xy, a_min=0, a_max=np.max(diff_xy))
        return inter[:, :, 0] * inter[:, :, 1]

    def union_area(self, boxes, inter):
        box_a = self.bboxes.copy()
        box_b = boxes.copy()
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))
        area_a = area_a[:, np.newaxis]
        area_b = area_b[np.newaxis, :]
        return area_a + area_b - inter

    def jaccard(self, boxes):
        """
        Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (np.array) Predicted bounding boxes,    Shape: [n_pred, 4]
            box_b: (np.array) Ground Truth bounding boxes, Shape: [n_gt, 4]
        Return:
            jaccard overlap: (np.array) Shape: [n_pred, n_gt]
        """
        inter = self.intersect_area(boxes)
        logging.debug(f'Obtained intersection of bboxes is {inter}')
        union = self.union_area(boxes, inter)

        logging.debug(f'Obtained union of bboxes is {union}')
        return inter / union

    @property
    def lefts(self):
        return self.bboxes[..., 0]

    @property
    def tops(self):
        return self.bboxes[..., 1]

    @property
    def rights(self):
        return self.bboxes[..., 2]

    @property
    def bottoms(self):
        return self.bboxes[..., 3]

    @property
    def center(self):
        return (self.bboxes[..., 2:] + self.bboxes[..., :2]) / 2

    @property
    def widths(self):
        return self.bboxes[..., 2] - self.bboxes[..., 0]

    @property
    def heights(self):
        return self.bboxes[..., 3] - self.bboxes[..., 1]

    @property
    def width_centers(self):
        return (self.bboxes[..., 2] + self.bboxes[..., 0]) / 2

    @property
    def height_centers(self):
        return (self.bboxes[..., 3] + self.bboxes[..., 1]) / 2

    @property
    def shape(self):
        return self.bboxes.shape


class BBox(object):
    def __init__(self,
                 class_id,
                 bbox,
                 class_map=None,
                 visibility=None,
                 blur_level=None,
                 region_level=None,
                 truncation=None):
        """
        :param class_id: ID of the corresponding class
        :param bbox: (top, left, bottom, right)
        :param visibility
               0 indicates "visible"
               1 indicates "partial-occlusion"
               2 indicates "full-occlusion"
        :param blur_level
               0 indicates no-blur
               1 indicates blur
        :param region_level
               When an object is detected, we are also interested on different part/region of the object
               For example, we can split a person into head, visible region, and full region.
        :param truncation indicates the degree of object parts appears outside a frame
                no truncation = 0 (truncation ratio 0%),
                partial truncation = 1 (truncation ratio 1% ~ 50%))
        """
        # todo: must be corrected from original json file
        self.class_id = class_id
        self.class_map = class_map
        self.bbox = bbox
        self.visibility = visibility
        self.blur_level = blur_level
        self.region_level = region_level
        self.truncation = truncation

    @property
    def top(self):
        return self.bbox[0]

    @property
    def left(self):
        return self.bbox[1]

    @property
    def bottom(self):
        return self.bbox[2]

    @property
    def right(self):
        return self.bbox[3]

    @property
    def width(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def height(self):
        return self.bbox[2] - self.bbox[0]


class BBoxesTensor(BBoxesMixin):
    def __init__(self, bboxes: np.array, coord='corner', method='numpy'):
        self.coord = coord
        self.method = method

        if self.method == 'numpy':
            _concat = np.concatenate
        elif self.method == 'tensorflow':
            _concat = tf.concat
        else:
            ValueError(f'The method {self.method} is not supported...')

        if self.coord == 'corner':
            self.bboxes = bboxes
        elif self.coord == 'centroid':
            bboxes_left_top = bboxes[..., :2] - .5 * bboxes[..., 2:]
            bboxes_right_bottom = bboxes[..., :2] + .5 * bboxes[..., 2:]
            self.bboxes = _concat((bboxes_left_top, bboxes_right_bottom), axis=-1)
        elif self.coord == 'minmax':
            bboxes_left_top = bboxes[..., [1, 0]]
            bboxes_right_bottom = bboxes[..., [3, 2]]
            self.bboxes = _concat((bboxes_left_top, bboxes_right_bottom), axis=-1)
        else:
            raise ValueError(f'The coordinate {self.coord} does not support...')
