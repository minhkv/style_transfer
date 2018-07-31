from __future__ import division, print_function, absolute_import
import numpy as np
import os
import matplotlib.pyplot as plt

from autoencoder import *
from feature_discriminator import *
from domain_adaptation import *
from utils import *
from usps import *
from mnist import *

tf.reset_default_graph()

save_iter = 2000
name = "adagrad_step1_10K"
model_folder = "./model/{}".format(name)
logdir = os.path.join("log", name)
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
domain_adaptation = DomainAdaptation(mnist_autoencoder, usps_autoencoder, logdir=logdir)
domain_adaptation.merge_all()
saver = tf.train.Saver()

batch_size = 100  # Number of samples in each batch
usps_data = USPSDataset(batch_size=batch_size, sess=domain_adaptation.sess)
mnist_data = MNISTDataset(batch_size=batch_size, sess=domain_adaptation.sess)

mnist_data.sample_dataset(2000)
usps_data.sample_dataset(1800)

r_1_fc = 10000
r_2_rec = 10000
r_3_df = 1000
r_4_di = 10
current_step = 0

 
 
# for step in (range(r_1_fc)):
#     if (step + 1) % save_iter == 0:
#         save_path = saver.save(domain_adaptation.sess, os.path.join(step1_model, "model_step1_{}.ckpt".format(step)))
#     batch_img, batch_label = mnist_data.next_batch()
#     batch_target, label_target = usps_data.next_batch()
#     domain_adaptation.run_step1(batch_img, batch_target, batch_label,  step + current_step)
current_step += r_1_fc

# save_path = saver.save(domain_adaptation.sess, os.path.join(step1_model, "model_step1_{}.ckpt".format(current_step)))

saver.restore(domain_adaptation.sess, os.path.join(step1_model, "model_step1_{}.ckpt".format(r_1_fc)))

for step in (range(r_2_rec)):

    if (step + 1) % save_iter == 0:
        save_path = saver.save(domain_adaptation.sess, os.path.join(step2_model, "model_step2_{}.ckpt".format(step)))

    batch_img, batch_label = mnist_data.next_batch()
    batch_target, label_target = usps_data.next_batch()
    domain_adaptation.run_step2(batch_img, batch_target, batch_label,  step + current_step)
current_step += r_2_rec

save_path = saver.save(domain_adaptation.sess, os.path.join(step2_model, "model_step2_{}.ckpt".format(current_step)))

# current_step = r_1_fc + r_2_rec
# saver.restore(domain_adaptation.sess, os.path.join(model_folder, "step2/model_{}.ckpt".format(r_1_fc + r_2_rec)))
# domain_adaptation.duplicate_source_ae_to_target_ae()

# for step in (range(r_3_df)):
#     batch_img, batch_label = mnist_data.next_batch_train()
#     batch_target, label_target = usps_data.next_batch_train()
#     domain_adaptation.run_step3(batch_img, batch_target, batch_label,  step + current_step)
# current_step += r_3_df

# for step in (range(r_4_di)):
#     batch_img, batch_label = mnist_data.next_batch_train()
#     batch_target, label_target = usps_data.next_batch_train()
#     domain_adaptation.run_step4(batch_img, batch_target, batch_label,  step + current_step)
