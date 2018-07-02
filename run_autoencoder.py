from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data
from autoencoder import *

def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    # Args:
    #   imgs: a numpy array of size [batch_size, 28 X 28].
    # Returns:
    #   a numpy array of size [batch_size, 32, 32].
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs

mnist = input_data.read_data_sets("MNIST_data")

batch_size = 500  # Number of samples in each batch
epoch_num = 5     # Number of epochs to train the network
lr = 0.0001        # Learning rate
# calculate the number of batches per epoch
batch_per_ep = mnist.train.num_examples // batch_size

autoencoder = Autoencoder()

# for ep in range(epoch_num):  # epochs loop
#     for batch_n in range(batch_per_ep):  # batches loop

for step in range(1000):
    batch_img, batch_label = mnist.train.next_batch(batch_size)  # read a batch
    batch_img = batch_img.reshape((-1, 28, 28, 1))               # reshape each sample to an (28, 28) image
    batch_img = resize_batch(batch_img)                          # reshape the images to (32, 32)
    
    # step = (epoch_num) * batch_per_ep + batch_n
    autoencoder.fit(batch_img, step)

