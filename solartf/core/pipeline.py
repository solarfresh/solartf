class TFPipelineBase:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.dataset = {}
        self.history = None

        if hasattr(self.config, 'TRAIN_IMAGE_PATH') \
                and hasattr(self.config, 'VALID_IMAGE_PATH') \
                and hasattr(self.config, 'TEST_IMAGE_PATH'):
            self.image_dir = {
                'train': self.config.TRAIN_IMAGE_PATH,
                'valid': self.config.VALID_IMAGE_PATH,
                'test': self.config.TEST_IMAGE_PATH
            }

        if hasattr(self.config, 'TRAIN_LABEL_PATH') \
                and hasattr(self.config, 'VALID_LABEL_PATH') \
                and hasattr(self.config, 'TEST_LABEL_PATH'):
            self.label_dir = {
                'train': self.config.TRAIN_LABEL_PATH,
                'valid': self.config.VALID_LABEL_PATH,
                'test': self.config.TEST_LABEL_PATH
            }

        self.shuffle = {
            'train': self.config.TRAIN_SHUFFLE,
            'valid': self.config.VALID_SHUFFLE,
            'test': self.config.TEST_SHUFFLE
        }
        self.batch_size = {
            'train': self.config.TRAIN_BATCH_SIZE,
            'valid': self.config.VALID_BATCH_SIZE,
            'test': self.config.TEST_BATCH_SIZE
        }
        self.augment = {
            'train': self.config.TRAIN_AUGMENT,
            'valid': self.config.VALID_AUGMENT,
            'test': None
        }

    def inference(self, *args, **kwargs):
        raise NotImplementedError

    def train(self):
        self.load_model().load_dataset()
        self.history = self.model.fit(self.dataset['train'],
                                      batch_size=self.config.TRAIN_BATCH_SIZE,
                                      epochs=self.config.TRAIN_EPOCH,
                                      verbose=self.config.TRAIN_VERBOSE,
                                      callbacks=self.config.TRAIN_CALLBACKS,
                                      valid_generator=self.dataset['valid'],
                                      shuffle=self.config.TRAIN_SHUFFLE,
                                      class_weight=self.config.TRAIN_CLASS_WEIGHT,
                                      initial_epoch=self.config.TRAIN_INITIAL_EPOCH,
                                      steps_per_epoch=self.config.TRAIN_STEP_PER_EPOCH,
                                      validation_steps=self.config.VALID_STEP_PER_EPOCH,
                                      validation_batch_size=self.config.VALID_BATCH_SIZE,
                                      validation_freq=self.config.VALID_FREQUENCY,
                                      max_queue_size=self.config.MAX_QUEUE_SIZE,
                                      workers=self.config.WORKER_NUMBER,
                                      use_multiprocessing=self.config.USE_MULTIPROCESSING)
        return self

    def load_dataset(self):
        raise NotImplementedError

    def load_model(self):
        self.model = self.config.MODEL
        self.model.build_model()

        if self.config.MODEL_WEIGHT_PATH is not None:
            self.model.load_model(self.config.MODEL_WEIGHT_PATH,
                                  custom_objects=self.config.CUSTOM_OBJECTS)

        self.model.compile(optimizer=self.config.TRAIN_OPTIMIZER,
                           loss=self.config.TRAIN_LOSS,
                           metrics=self.config.TRAIN_METRIC,
                           loss_weights=self.config.TRAIN_LOSS_WEIGHTS)

        return self

    def save_model(self):
        train_config = self.config.train
        self.model.save_model(train_config['save_path'])
        return self
