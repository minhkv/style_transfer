from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt

from autoencoder import *
from domain_adaptation import *
from utils import *
from usps import *
from mnist import *
from tqdm import tqdm

tf.reset_default_graph()

batch_size = 100  # Number of samples in each batch

usps_data = USPSDataset(batch_size=batch_size)
mnist_data = MNISTDataset(batch_size=batch_size)
mnist_autoencoder = Autoencoder(name="source")
usps_autoencoder = Autoencoder(name="target")
domain_adaptation = DomainAdaptation(mnist_autoencoder, usps_autoencoder)
domain_adaptation.merge_all()

# mnist_autoencoder.init_variable()
# mnist_autoencoder.merge_all() 

for step in (range(100)):
    batch_img, batch_label = mnist_data.next_batch_train() 
    batch_target, label_target = usps_data.next_batch_train()
    domain_adaptation.fit(batch_img, batch_target, step)
