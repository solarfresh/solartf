import tensorflow as tf
from .data.label.bbox import NumpyBBoxes
from .util import tf_bbox_intersection


class IoUFamilyMetric:
    def __init__(self, coord='corner'):
        if coord == 'iou':
            self.coord = 'corner'
        else:
            self.coord = coord

    def tf_iou(self, y_true, y_pred):
        """
        inputs here are based on centroid coordinate
        """
        y_true_corner = NumpyBBoxes(y_true, coord=self.coord, method='tensorflow').to_array(coord='corner')
        y_pred_corner = NumpyBBoxes(y_pred, coord=self.coord, method='tensorflow').to_array(coord='corner')

        boxes1_area = (y_true_corner[..., 2] - y_true_corner[..., 0]) * (y_true_corner[..., 3] - y_true_corner[..., 1])
        boxes2_area = (y_pred_corner[..., 2] - y_pred_corner[..., 0]) * (y_pred_corner[..., 3] - y_pred_corner[..., 1])
        inter_area = tf_bbox_intersection(y_true_corner, y_pred_corner)
        union_area = boxes1_area + boxes2_area - inter_area

        return tf.clip_by_value(inter_area / (union_area + tf.keras.backend.epsilon()), 0.0, 1.0)


class SSDMultipleMetric:

    def __init__(self,
                 mbox_conf_metric,
                 mbox_loc_metric):

        self.mbox_conf_metric = mbox_conf_metric
        self.mbox_loc_metric = mbox_loc_metric

    def compute_classification_metric(self, y_true, y_pred):
        sample_mask = tf.cast(tf.reduce_sum(y_true, axis=-1), tf.float32)

        y_true_sample = tf.boolean_mask(y_true, sample_mask)
        y_pred_sample = tf.boolean_mask(y_pred, sample_mask)

        return self.mbox_conf_metric(y_true_sample, y_pred_sample)

    def compute_localization_metric(self, y_true, y_pred):
        sample_mask = tf.cast(tf.reduce_sum(y_true[..., :2], axis=-1), tf.float32)

        y_true_sample = tf.boolean_mask(y_true, sample_mask)
        y_pred_sample = tf.boolean_mask(y_pred, sample_mask)

        return self.mbox_loc_metric(y_true_sample, y_pred_sample)
