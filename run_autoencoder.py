from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt

from autoencoder import *
from domain_adaptation import *
from utils import *
from usps import *
from mnist import *


# mnist = input_data.read_data_sets("MNIST_data")

batch_size = 100  # Number of samples in each batch
epoch_num = 5     # Number of epochs to train the network
lr = 0.0001        # Learning rate

# mnist_autoencoder = Autoencoder()
# # usps_data = USPSDataset(batch_size=batch_size)
# usps_data = MNISTDataset(batch_size=batch_size)
# for step in range(1000):
#     batch_img, batch_label = usps_data.next_batch_train() 
#     mnist_autoencoder.fit(batch_img, step)

mnist_autoencoder = Autoencoder()
usps_autoencoder = Autoencoder()
domain_adaptation = DomainAdaptation(mnist_autoencoder, usps_autoencoder)

