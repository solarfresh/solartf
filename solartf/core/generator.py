from tensorflow.keras.utils import Sequence


class KerasGeneratorBase(Sequence):
    def __len__(self):
        """Denotes the number of batches per epoch"""
        raise NotImplementedError

    def __getitem__(self, index):
        """Generate one batch of data"""
        raise NotImplementedError

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        raise NotImplementedError
