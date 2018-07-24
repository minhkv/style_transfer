from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt

from autoencoder import *
from feature_discriminator import *
from domain_adaptation import *
from utils import *
from usps import *
from mnist import *
from tqdm import tqdm

tf.reset_default_graph()

batch_size = 100  # Number of samples in each batch

usps_data = USPSDataset(batch_size=batch_size)
mnist_data = MNISTDataset(batch_size=batch_size)
mnist_autoencoder = Autoencoder(name="source", lr = 0.00001)
usps_autoencoder = Autoencoder(name="target")
# feat_dis = FeatureDiscriminator(name="feature_discriminator")

domain_adaptation = DomainAdaptation(mnist_autoencoder, usps_autoencoder)
domain_adaptation.merge_all()

r_cls = 100
r_c = 300
r_f = 100
current_step = 0

for i in range(1):
    batch_img, batch_label = mnist_data.next_batch_train() 
    batch_target, label_target = usps_data.next_batch_train()
    # domain_adaptation.run_optimize_image_discriminator_class(batch_img, batch_target, batch_label, i + current_step)
    domain_adaptation.minimize_autoencoder(batch_img, batch_target, i + current_step, batch_label)

# for i in range(50):
#     batch_img, batch_label = mnist_data.next_batch_train() 
#     batch_target, label_target = usps_data.next_batch_train()
#     domain_adaptation.run_optimize_feature_discriminator_type_g(batch_img, batch_target, batch_label,  i*2 + current_step)
#     domain_adaptation.run_optimize_feature_discriminator_type_d(batch_img, batch_target, batch_label,  i*2 + 1 + current_step)


# for step in (range(r_cls)):
#     batch_img, batch_label = mnist_data.next_batch_train() 
#     batch_target, label_target = usps_data.next_batch_train()
#     domain_adaptation.run_optimize_feature_classifier(batch_img, batch_target, batch_label,  step + current_step)
# current_step = r_cls

# for step in (range(r_c)):
#     batch_img, batch_label = mnist_data.next_batch_train() 
#     batch_target, label_target = usps_data.next_batch_train()
#     domain_adaptation.minimize_autoencoder(batch_img, batch_target, step + current_step, batch_label)


# for i in range(9):
    
#     for step in (range(r_c)):
#         batch_img, batch_label = mnist_data.next_batch_train() 
#         batch_target, label_target = usps_data.next_batch_train()
#         domain_adaptation.minimize_autoencoder(batch_img, batch_target, step + current_step, batch_label)
    
#     current_step += r_c
    
#     for step in range(r_f):
#         batch_img, batch_label = mnist_data.next_batch_train() 
#         batch_target, label_target = usps_data.next_batch_train()
#         domain_adaptation.minimize_feedback(batch_img, batch_target, step + current_step, batch_label)
        
#     current_step += r_f
        
#     for step in (range(r_c)):
#         batch_img, batch_label = mnist_data.next_batch_train() 
#         batch_target, label_target = usps_data.next_batch_train()
#         domain_adaptation.minimize_autoencoder(batch_img, batch_target, step + current_step, batch_label)
        
#     current_step += r_c
