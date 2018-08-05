from __future__ import division, print_function, absolute_import
import numpy as np
import os, sys
import matplotlib.pyplot as plt

from autoencoder import *
from feature_discriminator import *
from domain_adaptation import *
# from utils import *
from usps import *
from mnist import *

tf.reset_default_graph()

save_iter = 1000
name = sys.argv[1]
model_folder = "./model/{}".format(name)
logdir = os.path.join("log", name)
step1_log = os.path.join(logdir, 'step1')
step2_log = os.path.join(logdir, 'step2')
step3_log = os.path.join(logdir, 'step3')
step4_log = os.path.join(logdir, 'step4')

step1_model = os.path.join(model_folder, 'step1')
step2_model = os.path.join(model_folder, 'step2')
step3_model = os.path.join(model_folder, 'step3')
step4_model = os.path.join(model_folder, 'step4')

if not os.path.exists(step1_model):
    os.makedirs(step1_model)
if not os.path.exists(step2_model):
    os.makedirs(step2_model)
if not os.path.exists(step3_model):
    os.makedirs(step3_model)
if not os.path.exists(step4_model):
    os.makedirs(step4_model)

mnist_autoencoder = Autoencoder(name="source")
usps_autoencoder = Autoencoder(name="target")
domain_adaptation = DomainAdaptation(mnist_autoencoder, usps_autoencoder, gpu_fraction=0.2)
domain_adaptation.merge_all()
saver = tf.train.Saver()

batch_size = 100  # Number of samples in each batch
usps_data = USPSDataset(batch_size=batch_size, sess=domain_adaptation.sess)
mnist_data = MNISTDataset(batch_size=batch_size, sess=domain_adaptation.sess)

mnist_data.sample_dataset(2000)
usps_data.sample_dataset(1800)

domain_adaptation.set_logdir(step1_log)
saver.restore(domain_adaptation.sess, os.path.join(model_folder, "/home/acm528/Minh/style_transfer/model/test_acc/step2/model_step3_16999.ckpt"))
# domain_adaptation.duplicate_source_ae_to_target_ae()
for i in range(20):
    batch_img, batch_label = mnist_data.next_batch()
    batch_target, label_target = usps_data.next_batch()
    domain_adaptation.collect_feature(batch_img, batch_target, batch_label, i)
domain_adaptation.visualize_feature()
# domain_adaptation.tsne_sklearn()