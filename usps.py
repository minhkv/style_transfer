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
        
        
        self.current_index = 0
        
        self.num_train_batch = len(self.X_tr) // batch_size
        
    def _resize_batch(self, imgs):
        imgs = imgs.reshape((-1, 16, 16, 1))
        resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
        for i in range(imgs.shape[0]):
            resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
        return resized_imgs
        
    def next_batch_train(self):
        
        addition_from_start = 0
        start = self.current_index
        end = start + self.batch_size
        if end > len(self.X_tr):
            addition_from_start = end - len(self.X_tr) + 1
            end = len(self.X_tr) - 1
            self.current_index = addition_from_start
        else: 
            self.current_index += self.batch_size
        
        data = self.X_tr[start:end]
        label = self.y_tr[start: end]
        if addition_from_start > 0:
            data = np.concatenate((data, self.X_tr[0:addition_from_start]), axis=0)
            label = np.concatenate((label, self.y_tr[0:addition_from_start]), axis=0)
        data = self._resize_batch(data)
        return data, label
        
