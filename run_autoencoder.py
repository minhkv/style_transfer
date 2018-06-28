import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
from autoencoder import Autoencoder

mnist = input_data.read_data_sets('/home/minhkv/Documents/MachineLearning/tensorflow_tutorial/MNIST_data')

tf.reset_default_graph()

batch_size = 64

source_autoencoder = Autoencoder()
# source_autoencoder.construct()



for i in tqdm(range(30000)):
    batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
    source_autoencoder.fit(batch, i)    
    
