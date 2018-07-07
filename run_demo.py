from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt

from autoencoder import *

from utils import *

from mnist import *
from tqdm import tqdm
import tensorflow as tf


sess = tf.Session()
saver = tf.train.import_meta_graph("/tmp/model/ae_source.meta")
saver.restore(sess, "/tmp/model/ae_source")

graph = tf.get_default_graph()
ae_inputs = graph.get_tensor_by_name("Placeholder:0")
output = graph.get_tensor_by_name("decoder_source/output_source/BiasAdd:0")

batch_size = 100  # Number of samples in each batch
epoch_num = 5     # Number of epochs to train the network
lr = 0.0001        # Learning rate

mnist_data = MNISTDataset(batch_size=batch_size)
batch_img, batch_label = mnist_data.next_batch_train() 

output_img = sess.run(output, feed_dict = {ae_inputs: batch_img})

plt.imshow(np.reshape(output_img[0], (32, 32)), cmap='gray')
