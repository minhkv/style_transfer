import h5py
import numpy as np
from skimage import transform
from dataset import *
from tensorflow.examples.tutorials.mnist import input_data

class MNISTDataset(Dataset):
    def __init__(self, batch_size, sess):
        self.batch_size = batch_size
        self.mnist = input_data.read_data_sets("MNIST_data")#, one_hot=True)
        self.sess = sess
        self.X_tr, self.y_tr = self.mnist.train.next_batch(len(self.mnist.train.images))
        self.y_tr = self.sess.run(tf.one_hot(self.y_tr, depth=10))
        self._initialize()
    
    def _reshape_images(self, imgs):
        return imgs.reshape((-1, 28, 28, 1))
     