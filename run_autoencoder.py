import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
from autoencoder import Autoencoder
from usps_dataset import *

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

mnist = input_data.read_data_sets('/home/minhkv/Documents/MachineLearning/tensorflow_tutorial/MNIST_data')

tf.reset_default_graph()

batch_size = 64

source_autoencoder = Autoencoder()
# source_autoencoder.construct()

for i in tqdm(range(30000)):
    batch = [np.pad(np.reshape(b, [28, 28]), 2, pad_with) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
    # batch = [np.kron(np.reshape(b, [16, 16]), np.ones((2, 2))) for b in X_tr[i * batch_size: (i + 1) * batch_size]]
    source_autoencoder.fit(batch, i)    
    
