import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from solartf.data.bbox.type import BBoxesTensor
from .util import tf_bbox_intersection


def tf_category_crossentropy(y_true, y_pred):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_true, logits=y_pred
    )
    probs = tf.nn.sigmoid(y_pred)
    pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
    loss = pt * cross_entropy
    return tf.reduce_sum(loss, axis=-1)


def tf_convert_bbox_coordinates(tensor, conversion='centroids2corners'):
    if conversion == 'centroids2corners':
        tensor_prime = tf.concat([tensor[..., :2] - .5 * tensor[..., 2:],
                                  tensor[..., :2] + .5 * tensor[..., 2:]], axis=-1)
        # It will induce negative prediction after allowing the rule below
        # tensor_prime = tf.concat([tf.minimum(tensor_prime[..., :2], tensor_prime[..., 2:]),
        #                           tf.maximum(tensor_prime[..., :2], tensor_prime[..., 2:])], axis=-1)

    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids', 'centroids2minmax', "
                         "'corners2centroids', 'centroids2corners', 'minmax2corners', and 'corners2minmax'.")

    return tensor_prime


def log_loss(y_true, y_pred):
    """
    Compute the softmax log loss.

    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape (batch_size, #boxes, #classes)
            and contains the ground truth bounding box categories.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box categories.

    Returns:
        The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).
    """
    # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
    y_pred = tf.maximum(y_pred, 1e-15)
    # Compute the log loss
    log_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    return log_loss


def smooth_L1_loss(y_true, y_pred):
    """
    Compute smooth L1 loss, see references.

    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
            contains the ground truth bounding box coordinates, where the last dimension
            contains `(xmin, xmax, ymin, ymax)`.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box coordinates.

    Returns:
        The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).

    References:
        https://arxiv.org/abs/1504.08083
    """
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred) ** 2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)


class IoUFamilyLoss:
    def __init__(self, coord='corner'):
        if coord == 'iou':
            self.coord = 'corner'
        else:
            self.coord = coord

    def tf_diou_loss(self, y_true, y_pred):
        """
        inputs here are based on centroid coordinate
        """
        y_true_bbox = BBoxesTensor(y_true, coord=self.coord, method='tensorflow')
        y_pred_bbox = BBoxesTensor(y_pred, coord=self.coord, method='tensorflow')

        y_true_corner = y_true_bbox.to_array(coord='corner')
        y_pred_corner = y_pred_bbox.to_array(coord='corner')

        # Although the coordinate converted twice, it is convenient to read.
        iou = self.tf_iou_loss(y_true, y_pred)

        enclose_left_up = tf.minimum(y_true_corner[..., :2], y_pred_corner[..., :2])
        enclose_right_down = tf.maximum(y_true_corner[..., 2:], y_pred_corner[..., 2:])
        enclose_wh = enclose_right_down - enclose_left_up
        enclose_c2 = tf.math.pow(enclose_wh[..., 0], 2) + tf.math.pow(enclose_wh[..., 1], 2)

        y_true_center = y_true_bbox.center
        y_pred_center = y_pred_bbox.center

        p2 = (tf.math.pow(y_true_center[..., 0] - y_pred_center[..., 0], 2)
              + tf.math.pow(y_true_center[..., 1] - y_pred_center[..., 1], 2))

        return iou + 1.0 * p2 / (enclose_c2 + tf.keras.backend.epsilon())

    def tf_iou_loss(self, y_true, y_pred):
        """
        inputs here are based on centroid coordinate
        """
        y_true_corner = BBoxesTensor(y_true, coord=self.coord, method='tensorflow').to_array(coord='corner')
        y_pred_corner = BBoxesTensor(y_pred, coord=self.coord, method='tensorflow').to_array(coord='corner')

        boxes1_area = (y_true_corner[..., 2] - y_true_corner[..., 0]) * (y_true_corner[..., 3] - y_true_corner[..., 1])
        boxes2_area = (y_pred_corner[..., 2] - y_pred_corner[..., 0]) * (y_pred_corner[..., 3] - y_pred_corner[..., 1])
        inter_area = tf_bbox_intersection(y_true_corner, y_pred_corner)
        union_area = boxes1_area + boxes2_area - inter_area

        return 1.0 - tf.clip_by_value(inter_area / (union_area + tf.keras.backend.epsilon()), 0.0, 1.0)


class MonteCarloEstimateLoss:

    def logpx_loss(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
        return -tf.reduce_mean(logpx_z)

    def logpz_loss(self, y_true, y_pred):
        z, mean, logvar = tf.split(y_pred, num_or_size_splits=3, axis=-1)
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpz - logqz_x)

    @staticmethod
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)


class WeightedCategoricalCrossEntropy:
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    def __init__(self, weights):
        self.weights = K.variable(weights)

    def compute_loss(self, y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * self.weights
        loss = -K.sum(loss, -1)
        return loss


class SSDMultipleLoss:

    def __init__(self,
                 mbox_conf_loss,
                 mbox_loc_loss,
                 pos_weights=None,
                 neg_pos_ratio=3,
                 n_neg_min=0):

        self.mbox_conf_loss = mbox_conf_loss
        self.mbox_loc_loss = mbox_loc_loss

        if pos_weights is None:
            self.pos_weights = tf.constant(1, dtype=tf.float32)
        else:
            self.pos_weights = tf.constant(pos_weights, dtype=tf.float32)

        self.neg_pos_ratio = tf.constant(neg_pos_ratio, dtype=tf.float32)
        self.n_neg_min = tf.constant(n_neg_min, dtype=tf.int32)

    def compute_classification_loss(self, y_true, y_pred):
        batch_size, n_boxes = self._get_shape(y_pred)

        # generate masks to index negatives and positives
        negatives = y_true[..., 0]
        positives = tf.cast(tf.reduce_sum(y_true[..., 1:], axis=-1), tf.float32)
        weighted_positives = tf.cast(tf.reduce_sum(y_true[..., 1:] * self.pos_weights, axis=-1), tf.float32)

        # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch
        n_positive = tf.cast(tf.reduce_sum(positives), dtype=tf.int32)

        classification_loss = tf.cast(self.mbox_conf_loss(y_true, y_pred), dtype=tf.float32)
        pos_class_loss = tf.reduce_sum(classification_loss * weighted_positives, axis=-1)
        neg_class_loss_all = classification_loss * negatives

        n_neg_losses = tf.math.count_nonzero(neg_class_loss_all, dtype=tf.int32)

        n_negative_keep = tf.minimum(tf.maximum(tf.cast(self.neg_pos_ratio * tf.cast(n_positive, dtype=tf.float32), dtype=tf.int32),
                                                self.n_neg_min),
                                     n_neg_losses)
        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)),
                                 lambda: self._all_negative_loss(batch_size),
                                 lambda: self._top_k_negative_loss(classification_loss,
                                                                   batch_size,
                                                                   n_boxes,
                                                                   neg_class_loss_all,
                                                                   n_negative_keep))

        return (pos_class_loss + neg_class_loss) / tf.cast(n_positive, dtype=tf.float32) * tf.cast(batch_size, dtype=tf.float32)

    def compute_localization_loss(self, y_true, y_pred):
        batch_size, n_boxes = self._get_shape(y_pred)
        localization_loss = tf.cast(self.mbox_loc_loss(y_true, y_pred), dtype=tf.float32)
        # the last two items in centroid representation are not zeros
        positives = tf.cast(tf.reduce_sum(y_true[..., :2], axis=-1), tf.float32)
        positives = tf.where(positives == 0, tf.zeros_like(positives), tf.ones_like(positives))
        n_positive = tf.cast(tf.reduce_sum(positives), dtype=tf.int32)
        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)

        return loc_loss / tf.cast(n_positive, dtype=tf.float32) * tf.cast(batch_size, dtype=tf.float32)

    @staticmethod
    def _all_negative_loss(batch_size):
        return tf.zeros([batch_size])

    @staticmethod
    def _top_k_negative_loss(loss, batch_size, n_boxes, neg_loss_all, n_negative_keep):
        neg_class_loss_all_1d = tf.reshape(neg_loss_all, [-1])
        values, indices = tf.nn.top_k(neg_class_loss_all_1d, n_negative_keep, False)

        # Tensor of shape (batch_size * n_boxes,)
        negatives_keep = tf.scatter_nd(tf.expand_dims(indices, axis=1),
                                       updates=tf.ones_like(indices, dtype=tf.int32),
                                       shape=tf.shape(neg_class_loss_all_1d))
        negatives_keep = tf.cast(
            tf.reshape(negatives_keep, [batch_size, n_boxes]), dtype=tf.float32)

        return tf.reduce_sum(loss * negatives_keep, axis=-1)

    @staticmethod
    def _get_shape(y):
        batch_size = tf.shape(y)[0]
        n_boxes = tf.shape(y)[1]
        return batch_size, n_boxes


class FocalLoss(object):
    '''
    The SSD loss, see https://arxiv.org/abs/1512.02325.
    '''

    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        '''
        Arguments:
            neg_pos_ratio (int, optional): The maximum ratio of negative (i.e. background)
                to positive ground truth boxes to include in the loss computation.
                There are no actual background ground truth boxes of course, but `y_true`
                contains anchor boxes labeled with the background class. Since
                the number of background boxes in `y_true` will usually exceed
                the number of positive boxes by far, it is necessary to balance
                their influence on the loss. Defaults to 3 following the paper.
            n_neg_min (int, optional): The minimum number of negative ground truth boxes to
                enter the loss computation *per batch*. This argument can be used to make
                sure that the model learns from a minimum number of negatives in batches
                in which there are very few, or even none at all, positive ground truth
                boxes. It defaults to 0 and if used, it should be set to a value that
                stands in reasonable proportion to the batch size used for training.
            alpha (float, optional): A factor to weight the localization loss in the
                computation of the total loss. Defaults to 1.0 following the paper.
        '''
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        Compute smooth L1 loss, see references.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
                contains the ground truth bounding box coordinates, where the last dimension
                contains `(xmin, xmax, ymin, ymax)`.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box coordinates.

        Returns:
            The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).

        References:
            https://arxiv.org/abs/1504.08083
        '''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred, gamma=2, alpha=0.5):
        '''
        Compute the softmax log loss.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape (batch_size, #boxes, #classes)
                and contains the ground truth bounding box categories.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box categories.

        Returns:
            The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).
        '''
        # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
        # sigmoid_p = tf.nn.sigmoid(y_pred)
        # zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)

        # # For poitive prediction, only need consider front part loss, back part is 0;
        # # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        # pos_p_sub = array_ops.where(y_true > zeros, y_true - y_pred, zeros)

        # # For negative prediction, only need consider back part loss, front part is 0;
        # # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        # neg_p_sub = array_ops.where(y_true > zeros, zeros, y_pred)
        # per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0)) \
        #                   - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - y_pred, 1e-8, 1.0))
        # return tf.reduce_sum(per_entry_cross_ent)
        y_pred = tf.maximum(y_pred, 1e-15)
        log_y_pred = tf.math.log(y_pred)
        focal_scale = tf.multiply(tf.pow(tf.subtract(1.0, y_pred), gamma), alpha)
        focal_loss = tf.multiply(y_true, tf.multiply(focal_scale, log_y_pred))
        return -tf.reduce_sum(focal_loss, axis=-1)

    def compute_loss(self, y_true, y_pred):
        '''
        Compute the loss of the SSD model prediction against the ground truth.

        Arguments:
            y_true (array): A Numpy array of shape `(batch_size, #boxes, #classes + 12)`,
                where `#boxes` is the total number of boxes that the model predicts
                per image. Be careful to make sure that the index of each given
                box in `y_true` is the same as the index for the corresponding
                box in `y_pred`. The last axis must have length `#classes + 12` and contain
                `[classes one-hot encoded, 4 ground truth box coordinate offsets, 8 arbitrary entries]`
                in this order, including the background class. The last eight entries of the
                last axis are not used by this function and therefore their contents are
                irrelevant, they only exist so that `y_true` has the same shape as `y_pred`,
                where the last four entries of the last axis contain the anchor box
                coordinates, which are needed during inference. Important: Boxes that
                you want the cost function to ignore need to have a one-hot
                class vector of all zeros.
            y_pred (Keras tensor): The model prediction. The shape is identical
                to that of `y_true`, i.e. `(batch_size, #boxes, #classes + 12)`.
                The last axis must contain entries in the format
                `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.

        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)

        batch_size = tf.shape(y_pred)[0]  # Output dtype: tf.int32
        n_boxes = tf.shape(y_pred)[
            1]  # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell

        # 1: Compute the losses for class and box predictions for every box

        classification_loss = tf.cast(
            self.log_loss(y_true[:, :, :-12], y_pred[:, :, :-12]), dtype=tf.float32)  # Output shape: (batch_size, n_boxes)
        localization_loss = tf.cast(
            self.smooth_L1_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8]), dtype=tf.float32)  # Output shape: (batch_size, n_boxes)

        # 2: Compute the classification losses for the positive and negative targets

        # Create masks for the positive and negative ground truth classes
        negatives = y_true[:, :, 0]  # Tensor of shape (batch_size, n_boxes)
        positives = tf.cast(tf.reduce_max(y_true[:, :, 1:-12], axis=-1), dtype=tf.float32)  # Tensor of shape (batch_size, n_boxes)

        # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch
        n_positive = tf.reduce_sum(positives)

        # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
        # (Keras loss functions must output one scalar loss value PER batch item, rather than just
        # one scalar for the entire batch, that's why we're not summing across all axes)
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)  # Tensor of shape (batch_size,)

        # Compute the classification loss for the negative default boxes (if there are any)

        # First, compute the classification loss for all negative boxes
        neg_class_loss_all = classification_loss * negatives  # Tensor of shape (batch_size, n_boxes)
        # The number of non-zero loss entries in `neg_class_loss_all`
        n_neg_losses = tf.math.count_nonzero(neg_class_loss_all, dtype=tf.int32)
        # What's the point of `n_neg_losses`? For the next step, which will be to compute which negative boxes enter the classification
        # loss, we don't just want to know how many negative ground truth boxes there are, but for how many of those there actually is
        # a positive (i.e. non-zero) loss. This is necessary because `tf.nn.top-k()` in the function below will pick the top k boxes with
        # the highest losses no matter what, even if it receives a vector where all losses are zero. In the unlikely event that all negative
        # classification losses ARE actually zero though, this behavior might lead to `tf.nn.top-k()` returning the indices of positive
        # boxes, leading to an incorrect negative classification loss computation, and hence an incorrect overall loss computation.
        # We therefore need to make sure that `n_negative_keep`, which assumes the role of the `k` argument in `tf.nn.top-k()`,
        # is at most the number of negative boxes for which there is a positive classification loss.

        # Compute the number of negative examples we want to account for in the loss
        # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller)
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.cast(n_positive, tf.int32), self.n_neg_min),
                                     n_neg_losses)

        # In the unlikely case when either (1) there are no negative ground truth boxes at all
        # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`
        def f1():
            return tf.zeros([batch_size])

        # Otherwise compute the negative loss
        def f2():
            # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
            # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
            # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

            # To do this, we reshape `neg_class_loss_all` to 1D...
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])  # Tensor of shape (batch_size * n_boxes,)
            # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
            values, indices = tf.nn.top_k(neg_class_loss_all_1D, n_negative_keep, False)  # We don't need sorting
            # ...and with these indices we'll create a mask...
            negatives_keep = tf.scatter_nd(tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32), shape=tf.shape(
                    neg_class_loss_all_1D))  # Tensor of shape (batch_size * n_boxes,)
            negatives_keep = tf.cast(
                tf.reshape(negatives_keep, [batch_size, n_boxes]), dtype=tf.float32)  # Tensor of shape (batch_size, n_boxes)
            # ...and use it to keep only those boxes and mask all other classification losses
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep,
                                           axis=-1)  # Tensor of shape (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss  # Tensor of shape (batch_size,)

        # 3: Compute the localization loss for the positive targets
        #    We don't penalize localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to)

        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)  # Tensor of shape (batch_size,)

        # 4: Compute the total loss

        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive)  # In case `n_positive == 0`
        # Keras has the annoying habit of dividing the loss by the batch size, which sucks in our case
        # because the relevant criterion to average our loss over is the number of positive boxes in the batch
        # (by which we're dividing in the line above), not the batch size. So in order to revert Keras' averaging
        # over the batch size, we'll have to multiply by it.
        total_loss *= tf.cast(batch_size, dtype=tf.float32)

        return total_loss


class WeightedFocalLoss(FocalLoss):
    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0,
                 weights=None):
        super(WeightedFocalLoss, self).__init__(neg_pos_ratio,
                                                n_neg_min,
                                                alpha)
        self.weights = weights

    def log_loss(self, y_true, y_pred, gamma=2, alpha=0.5):
        weighted = tf.multiply(y_true, self.weights)
        y_pred = tf.maximum(y_pred, 1e-15)
        log_y_pred = tf.math.log(y_pred)
        focal_scale = tf.multiply(tf.pow(tf.subtract(1.0, y_pred), gamma), alpha)
        focal_loss = tf.multiply(weighted, tf.multiply(focal_scale, log_y_pred))
        return -tf.reduce_sum(focal_loss, axis=-1)


def dice_loss(y_true, y_pred):
    """
    Loss function based on the Dice coefficient, which measures the size of overlap between
    a segmentation map predicted by the model and the ground truth segmentation map.
    It can better cope with unbalanced data than the binary crossentropy.
    The Dice coefficient is calculated as

    ```
         2*y_true*y_pred
    Dc = ---------------
         y_true + y_pred
    ```
    Because the Dice coefficient by itself gets *higher*, the more y_true and y_pred
    overlap, we have to substract it from 1 to yield a proper loss function

    Notes:

    * The +1 in the numerator and denominator is there to prevent a catastrophic
      division by zero. And because it's in both numerator and denominator it cancels
      each other out mathematically.
    * Beware when using this loss function, it might produce nasty gradients (To see why,
      compare its derivative with the one of the binary crossentropy). When in doubt,
      use a lower learning rate.
    * The original publication multiplies the numerator with 2 but that leads to a
      minimal value of ~0.5 - probably because it originally was designed for *volumes* instead
      of 2D segmentation maps. Changing the factor 2 to 4 yields a loss very close to zero
      when the maps overlap highly but it might make the gradient problem worse.

    Originally appeared in:
    [1] "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
        Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi (https://arxiv.org/abs/1606.04797)

    See also:
    [2] Carole H Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, and
    M Jorge Cardoso. "Generalised dice overlap as a deep learning loss
    function for highly unbalanced segmentations". In Deep learning in
    medical image analysis and multimodal learning for clinical decision
    support, pages 240–248. Springer, 2017.

    Arguments:
        y_true: A (usually 4D) tensor containing ground truth heatmaps for each element
            in the batch.
        y_pred: A (usually 4D) tensor containing the model's predicted heatmaps

    Returns:
        A scalar in the interval [0, 1] signifying the Dice loss.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coeff = (2.0 * intersection + 1) / ((K.sum(y_true_f) + K.sum(y_pred_f) + 1))
    return 1.0 - dice_coeff
