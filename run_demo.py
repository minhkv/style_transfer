from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt

from autoencoder import *
from mnist import *
import tensorflow as tf


restore_source = Autoencoder("source", 
        meta_graph="/tmp/model/ae_source.meta", 
        checkpoint_dir="/tmp/model/ae_source")

batch_size = 100  # Number of samples in each batch

mnist_data = MNISTDataset(batch_size=batch_size)
batch_img, batch_label = mnist_data.next_batch_train() 

restore_source.merge_all()
restore_source.fit(batch_img, 1)

# output_img = restore_source.forward(batch_img)
# plt.imshow(np.reshape(output_img[0], (32, 32)), cmap='gray')
# plt.show()
