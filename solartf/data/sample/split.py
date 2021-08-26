import os
import shutil
import numpy as np


def random_select_samples(number, dirname, target, method='copy'):
    filenames = os.listdir(dirname)
    choice = np.unique(np.random.choice(filenames, number))
    for filename in choice:
        if method == 'copy':
            shutil.copy(os.path.join(dirname, filename),
                        os.path.join(target, filename))
        elif method == 'move':
            shutil.move(os.path.join(dirname, filename),
                        os.path.join(target, filename))
        else:
            raise ValueError(f'{method} does not support...')
