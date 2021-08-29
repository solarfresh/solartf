import os
import numpy as np
from solartf.core.pipeline import TFPipelineBase
from .generator import DetectDirectoryGenerator


class AnchorDetectPipeline(TFPipelineBase):

    def inference(self, dataset_type='test'):
        self.load_model().load_dataset()
        result_list = []
        for image_input_list, bboxes_input_list in self.dataset[dataset_type]:
            image_arrays, batch_gts = self.model.data_preprocess({'image_input_list': image_input_list,
                                                                  'bboxes_input_list': bboxes_input_list})

            height, width, _ = self.model.input_shape
            predict_results = self.model.predict(image_arrays)
            batch_mbox_conf = batch_gts['mbox_conf']
            batch_mbox_loc = batch_gts['mbox_loc']
            batch_mbox_priorbox = batch_gts['mbox_priorbox']
            mbox_conf_outputs = predict_results['mbox_conf']
            mbox_loc_outputs = predict_results['mbox_loc']
            for info in zip(image_input_list,
                            batch_mbox_conf,
                            batch_mbox_loc,
                            mbox_conf_outputs,
                            mbox_loc_outputs,
                            batch_mbox_priorbox):
                image_input, mbox_conf_gt, mbox_loc_gt, mbox_conf_pred, mbox_loc_pred, mbox_priorbox = info

                result = {
                    'fname': os.path.basename(image_input.image_path),
                }

                bboxes_gt = self.model.bbox_output.decoder(np.concatenate([mbox_loc_gt, mbox_priorbox], axis=-1))
                bboxes_pred = self.model.bbox_output.decoder(np.concatenate([mbox_loc_pred, mbox_priorbox], axis=-1))

                for idx in range(self.model.n_classes):
                    result.update({f'mbox_conf_{idx}_gt': mbox_conf_gt[..., idx]})
                    result.update({f'mbox_conf_{idx}_pred': mbox_conf_pred[..., idx]})

                bboxes_tensor_gt = bboxes_gt.to_array(coord='corner').astype(np.int32)
                bboxes_tensor_pred = bboxes_pred.to_array(coord='corner').astype(np.int32)
                for idx, name in enumerate(['left', 'top', 'right', 'bottom']):
                    result.update({f'{name}_gt': bboxes_tensor_gt[..., idx]})
                    result.update({f'{name}_pred': bboxes_tensor_pred[..., idx]})

                result_list.append(result)

        return result_list

    def load_dataset(self, training=True):
        if self.model is None:
            raise ValueError(f'model must be load before loading datasets')

        self.dataset = {}
        for set_type in ['train', 'valid', 'test']:
            self.dataset[set_type] = DetectDirectoryGenerator(
                image_dir=self.image_dir[set_type],
                label_dir=self.label_dir[set_type],
                image_shape=self.config.IMAGE_SHAPE,
                shuffle=self.shuffle[set_type],
                batch_size=self.batch_size[set_type],
                image_type=self.config.IMAGE_TYPE,
                dataset_type=set_type,
                in_memory=self.config.IN_MEMORY,
                augment=self.augment[set_type],
                processor=self.model.data_preprocess,
                bbox_exclude=self.config.BBOX_EXCLUDE
            )

        return self
