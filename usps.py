import h5py
import numpy as np
from skimage import transform


class USPSDataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        with h5py.File("USPS_data/usps.h5", "r") as hf:
            train = hf.get('train')
            self.X_tr = train.get('data')[:]
            self.y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
        self.current_train_batch = 0
        self.num_train_batch = len(self.X_tr) // batch_size
        
    def _resize_batch(self, imgs):
        imgs = imgs.reshape((-1, 16, 16, 1))
        resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
        for i in range(imgs.shape[0]):
            resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
        return resized_imgs
        
    def next_batch_train(self):
        current_batch = self.current_train_batch
        start = current_batch * self.batch_size
        end = (current_batch + 1) * self.batch_size 
        if end > len(self.X_tr):
            end = len(self.X_tr) - 1
            self.current_train_batch = 0
            
        data = self.X_tr[start: end]
        data = self._resize_batch(data)
        
        label = self.y_tr[start: end]
        self.current_train_batch += 1
        return data, label
        
