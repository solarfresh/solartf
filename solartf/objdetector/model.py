import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import (categorical_crossentropy, mse)
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from solartf.core import graph
from solartf.core.block import InvertedResBlock
from solartf.core.model import TFModelBase
from solartf.data.bbox.type import BBoxesTensor
from .bbox import (BBoxLabeler, BBoxOutput, GridAnchor)
from .graph import AnchorDetectHead
from .layer import AnchorBoxes


class AnchorBaseDetectTFModel(TFModelBase):
    def __init__(self,
                 input_shape,
                 n_classes,
                 aspect_ratios,
                 feature_map_shapes,
                 scales=None,
                 step_shapes=None,
                 offset_shapes=None,
                 variances=None,
                 bbox_encoding='iou',
                 bbox_normalize=False,
                 pos_iou_threshold=.5,
                 neg_iou_threshold=.3,
                 bbox_labeler_method='threshold'):

        self.input_shape = input_shape
        self.n_classes = n_classes

        self.aspect_ratios = aspect_ratios
        self.n_boxes = []
        for ar in self.aspect_ratios:
            self.n_boxes.append(len(ar))

        self.n_predictor_layers = len(aspect_ratios)
        self.scales = scales
        self.step_shapes = step_shapes
        self.offset_shapes = offset_shapes
        self.variances = variances
        self.bbox_encoding = bbox_encoding
        self.bbox_normalize = bbox_normalize
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.bbox_encoding = bbox_encoding
        self.bbox_normalize = bbox_normalize
        self.bbox_output = BBoxOutput(method=self.bbox_encoding, normalize=self.bbox_normalize)
        self.bbox_labeler_method = bbox_labeler_method
        self.anchor_generator = GridAnchor()
        # It is in corner representation
        self.feature_map_shapes = feature_map_shapes
        self.detect_head = AnchorDetectHead(self.n_classes,
                                            image_shape=self.input_shape[:2],
                                            n_boxes=self.n_boxes,
                                            scales=self.scales,
                                            aspect_ratios=self.aspect_ratios,
                                            step_shapes=self.step_shapes,
                                            offset_shapes=self.offset_shapes,
                                            variances=self.variances,
                                            l2_regularization=0.0005)

        self.anchor_tensor = self._get_anchor_tensor(image_shape=input_shape[:2])
        self.bbox_labeler = BBoxLabeler(self.anchor_tensor, )

        if variances is None:
            variances = [0.1, 0.1, 0.2, 0.2]

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))

        self.variances = np.array(variances)
        if np.any(self.variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        self.classes_tensor = np.zeros((self.anchor_tensor.shape[0], self.n_classes))
        self.class_vector = np.eye(self.n_classes)
        self.variances_tensor = np.zeros(shape=self.anchor_tensor.shape)
        self.variances_tensor += self.variances

    def data_preprocess(self, inputs):
        return self._train_encoder(inputs)

    def data_postprocess(self, outputs, meta):
        return outputs

    def build_model(self):
        raise NotImplementedError

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None):
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04) \
            if optimizer is None else optimizer

        loss = {'mbox_conf': categorical_crossentropy,
                'mbox_loc': mse,
                'mbox_priorbox': mse} if loss is None else loss

        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)

        return self

    def load_model(self, filepath, custom_objects=None):
        if custom_objects is not None:
            custom_objects.update({'AnchorBoxes': AnchorBoxes})
        else:
            custom_objects = {'AnchorBoxes': AnchorBoxes}

        self.model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
        return self

    def _train_encoder(self, inputs):
        image_input_list = inputs['image_input_list']
        bboxes_input_list = inputs['bboxes_input_list']

        batch_image_input = np.stack([image_input.image_array
                                      for image_input in image_input_list], axis=0)

        batch_mbox_conf = []
        batch_mbox_loc = []
        batch_mbox_priorbox = []
        for bboxes_input in bboxes_input_list:
            labels = bboxes_input.labels

            mbox_conf = self.classes_tensor.copy()
            mbox_loc = self.anchor_tensor.to_array(coord='corner').copy()

            gt_boxes = bboxes_input.to_array(coord='corner').copy()
            if gt_boxes.size > 0:
                bbox_label_info = self.bbox_labeler.get_bbox_label_mask(gt_boxes,
                                                                        method=self.bbox_labeler_method,
                                                                        th_neg=self.neg_iou_threshold,
                                                                        th_pos=self.pos_iou_threshold)
                neg_anchor_mask, pos_anchor_mask, pos_bbox_index = bbox_label_info
                for pos_anchor_index in np.argwhere(pos_anchor_mask).reshape(-1):
                    mbox_conf[pos_anchor_index] = self.class_vector[labels[pos_bbox_index[pos_anchor_index]]]
                    mbox_loc[pos_anchor_index] = gt_boxes[pos_bbox_index[pos_anchor_index]]

                mbox_conf[neg_anchor_mask, 0] = 1

            bboxes_encoded_outputs = self.bbox_output.encoder(BBoxesTensor(mbox_loc,
                                                                           coord='corner',
                                                                           method='numpy'),
                                                              self.anchor_tensor,
                                                              self.variances_tensor)

            batch_mbox_conf.append(mbox_conf)
            batch_mbox_loc.append(bboxes_encoded_outputs[..., :4])
            batch_mbox_priorbox.append(bboxes_encoded_outputs[..., 4:])

        outputs = {'mbox_conf': np.stack(batch_mbox_conf, axis=0),
                   'mbox_loc': np.stack(batch_mbox_loc, axis=0),
                   'mbox_priorbox': np.stack(batch_mbox_priorbox, axis=0)}

        return batch_image_input, outputs

    def _get_anchor_tensor(self, image_shape):
        anchor_tensor_list = []
        for index in range(self.n_predictor_layers):
            anchor_tensor = self.anchor_generator.generate_anchors(image_shape=image_shape,
                                                                   feature_map_shape=self.feature_map_shapes[index],
                                                                   aspect_ratios=self.aspect_ratios[index],
                                                                   scale=self.scales[index],
                                                                   step_shape=self.step_shapes[index],
                                                                   offset_shape=self.offset_shapes[index])
            anchor_tensor_list.append(anchor_tensor.reshape((-1, 4)))

        return BBoxesTensor(np.concatenate(anchor_tensor_list),
                            coord='centroid',
                            method='numpy')


class MobileNetV3SmallTFModel(AnchorBaseDetectTFModel):
    def __init__(self,
                 fpn_n_filter=40,
                 *args, **kwargs):
        super(MobileNetV3SmallTFModel, self).__init__(*args, **kwargs)

        self.backbone = graph.MobileNetV3Small(
            last_point_ch=1024,
            alpha=1.0
        )
        self.fpn = graph.FeaturePyramidNetwork(
            n_features=5,
            n_filters=fpn_n_filter,
        )

        self.inverted_res_blocks = []
        for idx in range(3):
            self.inverted_res_blocks.append([
                InvertedResBlock(
                    expansion=6,
                    infilters=fpn_n_filter,
                    filters=fpn_n_filter,
                    kernel_size=3,
                    strides=1,
                    se_ratio=.25,
                ),
                InvertedResBlock(
                    expansion=6,
                    infilters=fpn_n_filter,
                    filters=fpn_n_filter,
                    kernel_size=3,
                    strides=2,
                    se_ratio=.25,
                )
            ])
            self.detect_conv_blocks = [
                InvertedResBlock(
                    expansion=6,
                    infilters=fpn_n_filter,
                    filters=fpn_n_filter,
                    kernel_size=3,
                    strides=1,
                    se_ratio=.25,
                ),
                InvertedResBlock(
                    expansion=6,
                    infilters=fpn_n_filter,
                    filters=fpn_n_filter,
                    kernel_size=3,
                    strides=1,
                    se_ratio=.25,
                )
            ]

    def build_model(self):
        image_input = Input(shape=self.input_shape, name='detect_image_input')
        backbone_outputs = self.backbone(image_input)
        detect_neck = self.fpn(backbone_outputs[:-1])
        detect_neck.reverse()
        x = detect_neck[-1]
        for inverted_res_block_group in self.inverted_res_blocks:
            for inverted_res_block in inverted_res_block_group:
                x = inverted_res_block(x)
            detect_neck.append(x)

        predictions = self.detect_head.call(detect_neck[-self.n_predictor_layers:])

        self.model = Model(image_input, predictions)
        return self

    def detect_conv_block(self, x):
        for conv in self.detect_conv_blocks:
            x = conv(x)
        return x
