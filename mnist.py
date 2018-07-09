import h5py
import numpy as np
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data

class MNISTDataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.mnist = input_data.read_data_sets("MNIST_data")
        self.num_train_batch = self.mnist.train.num_examples // batch_size
        
    def _resize_batch(self, imgs):
        imgs = imgs.reshape((-1, 28, 28, 1))
        resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
        for i in range(imgs.shape[0]):
            resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
        return resized_imgs
        
    def next_batch_train(self):
        batch_img, batch_label = self.mnist.train.next_batch(self.batch_size)  # read a batch
        batch_img = batch_img.reshape((-1, 28, 28, 1))               # reshape each sample to an (28, 28) image
        batch_img = self._resize_batch(batch_img)                          # reshape the images to (32, 32)
        return batch_img, batch_label
