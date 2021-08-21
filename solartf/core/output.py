import numpy as np


class OutputBase:
    def encoder(self, outputs: np.array):
        raise NotImplementedError

    def decoder(self, outputs: np.array):
        raise NotImplementedError
