import os
import numpy as np
from solartf.core.pipeline import TFPipelineBase
from .generator import KeypointDirectoryGenerator


class KeypointDetectPipeline(TFPipelineBase):
    def inference(self, output_shape=None, dataset_type='test'):
        if output_shape is None:
            output_shape = self.model.input_shape

        self.load_model().load_dataset()

        result_list = []
        for image_input_list, kpt_input_list in self.dataset[dataset_type]:
            image_arrays, batch_gts = self.model.data_preprocess({'image_input_list': image_input_list,
                                                                  'kpt_input_list': kpt_input_list})
            height, width, _ = self.model.input_shape
            predict_results = self.model.predict(image_arrays)
            batch_cls_gts = batch_gts['cls']
            batch_kpt_gts = batch_gts['kpt']
            cls_outputs = predict_results['cls']
            kpt_outputs = predict_results['kpt']
            for image_input, cls_gt, kpt_gt, cls_output, kpt_output in zip(image_input_list,
                                                                           batch_cls_gts,
                                                                           batch_kpt_gts,
                                                                           cls_outputs,
                                                                           kpt_outputs):

                result = {
                    'fname': os.path.basename(image_input.image_path),
                    'width': output_shape[1],
                    'height': output_shape[0]
                }

                kpt_gt[..., 0] = kpt_gt[..., 0] * output_shape[1]
                kpt_gt[..., 1] = kpt_gt[..., 1] * output_shape[0]
                kpt_output[..., 0] = kpt_output[..., 0] * output_shape[1]
                kpt_output[..., 1] = kpt_output[..., 1] * output_shape[0]
                kpt_gt = kpt_gt.astype(np.int32)
                kpt_output = kpt_output.astype(np.int32)
                for idx in range(self.model.n_classes):
                    result.update({f'cls_{idx}_gt': cls_gt[idx]})
                    result.update({f'cls_{idx}_pred': cls_output[idx]})

                    result.update({f'cls_{idx}_kptx_gt': kpt_gt[idx, 0]})
                    result.update({f'cls_{idx}_kpty_gt': kpt_gt[idx, 1]})
                    result.update({f'cls_{idx}_kptx_pred': kpt_output[idx, 0]})
                    result.update({f'cls_{idx}_kpty_pred': kpt_output[idx, 1]})

                result_list.append(result)

        return result_list

    def load_dataset(self):
        if self.model is None:
            raise ValueError(f'model must be load before loading datasets')

        self.dataset = {}
        for set_type in ['train', 'valid', 'test']:
            self.dataset[set_type] = KeypointDirectoryGenerator(
                image_dir=self.image_dir[set_type],
                label_dir=self.label_dir[set_type],
                image_shape=self.config.IMAGE_SHAPE,
                shuffle=self.shuffle[set_type],
                batch_size=self.batch_size[set_type],
                augment=self.augment[set_type],
                image_type=self.config.IMAGE_TYPE,
                dataset_type=set_type,
                processor=self.model.data_preprocess)

        return self
