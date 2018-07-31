import h5py
import numpy as np
from skimage import transform
from dataset import *

class USPSDataset(Dataset):
    def __init__(self, batch_size, sess):
        self.batch_size = batch_size
        with h5py.File("USPS_data/usps.h5", "r") as hf:
            train = hf.get('train')
            self.X_tr = train.get('data')[:]
            self.y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
        self.sess = sess
        self._initialize()
    def _reshape_images(self, imgs):
        return imgs.reshape((-1, 16, 16, 1))