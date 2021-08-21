class TFPipelineBase:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.dataset = {}
        self.history = None

    def inference(self):
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
        raise NotImplementedError

    def save_model(self):
        train_config = self.config.train
        self.model.save_model(train_config['save_path'])
        return self
