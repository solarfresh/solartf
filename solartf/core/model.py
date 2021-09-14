import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


class TFModelBase:
    model = None
    tf_freeze_model = None

    def data_preprocess(self, inputs):
        raise NotImplementedError

    def data_postprocess(self, outputs, meta):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def fit(self,
            train_generator,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            valid_generator=None,
            shuffle=True,
            class_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):
        return self.model.fit(x=train_generator,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=verbose,
                              callbacks=callbacks,
                              validation_data=valid_generator,
                              shuffle=shuffle,
                              class_weight=class_weight,
                              initial_epoch=initial_epoch,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              validation_batch_size=validation_batch_size,
                              validation_freq=validation_freq,
                              max_queue_size=max_queue_size,
                              workers=workers,
                              use_multiprocessing=use_multiprocessing)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def load_weights(self, filepath, skip_mismatch=True, by_name=True):
        self.model.load_weights(filepath, skip_mismatch=skip_mismatch, by_name=by_name)
        return self

    def load_model(self, filepath, custom_objects=None):
        self.model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
        return self

    def save_model(self, filepath):
        self.model.save(filepath, save_format='tf')
        return self

    def freeze_graph(self, save_dir=None, model_name=None, as_text=False):
        selftf_freeze_model = tf.function(lambda x: self.model(x))
        self.tf_freeze_model = selftf_freeze_model.get_concrete_function(
            tf.TensorSpec(self.model.inputs[0].shape,
                          self.model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(self.tf_freeze_model)
        # frozen_func.graph.as_graph_def()

        if model_name is not None:
            logdir = save_dir if save_dir is not None else './'
            tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                              logdir=logdir,
                              name=model_name,
                              as_text=as_text)

        return self
