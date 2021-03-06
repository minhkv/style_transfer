from __future__ import division, print_function, absolute_import
import numpy as np
import os, sys
import matplotlib.pyplot as plt

from autoencoder import *
from feature_discriminator import *
from domain_adaptation import *
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

# if os.path.exists(step1_log):
#     os.system("rm {}/*".format(step1_log))
# if os.path.exists(step2_log):
#     os.system("rm {}/*".format(step2_log))
# if os.path.exists(step3_log):
#     os.system("rm {}/*".format(step3_log))
# if os.path.exists(step4_log):
#     os.system("rm {}/*".format(step4_log))


mnist_autoencoder = Autoencoder(name="source")
usps_autoencoder = Autoencoder(name="target")
domain_adaptation = DomainAdaptation(mnist_autoencoder, usps_autoencoder)
domain_adaptation.merge_all()


# variable_not_restored = [var for var in tf.trainable_variables() 
#     if var.name.startswith('discriminator_{}'.format(domain_adaptation.feature_discriminator.name)) 
#     and 'BatchNorm' in str(var.name)
#     ]
# variable_not_restored += [var for var in tf.trainable_variables() 
#     if var.name.startswith('feature_classifier_{}'.format(domain_adaptation.name)) 
#     and 'BatchNorm' in str(var.name)
#     ]
# # variable_not_restored = []
# variable_to_restore = [var for var in tf.trainable_variables()
#     if var not in variable_not_restored]

# saver = tf.train.Saver(var_list=variable_to_restore)
saver = tf.train.Saver(max_to_keep=100)

batch_size = 100  # Number of samples in each batch
usps_data = USPSDataset(batch_size=batch_size, sess=domain_adaptation.sess)
mnist_data = MNISTDataset(batch_size=batch_size, sess=domain_adaptation.sess)

mnist_data.sample_dataset(2000)
usps_data.sample_dataset(1800)

# mnist_data.one_hot_encoding_label()

r_1_fc = 5000
r_2_rec = 5000
r_3_df = 5000
r_4_di = 10
current_step = 10000

# # saver.restore(domain_adaptation.sess, os.path.join(step1_model, "model_step1_{}.ckpt".format(999)))

# domain_adaptation.set_logdir(step1_log)
# for step in (range(r_1_fc)):
#     if (step + 1) % save_iter == 0:
#         save_path = saver.save(domain_adaptation.sess, os.path.join(step1_model, "model_step1_{}.ckpt".format(step)))
#     batch_img, batch_label = mnist_data.next_batch()
#     batch_target, label_target = usps_data.next_batch()
#     domain_adaptation.run_step1(batch_img, batch_target, batch_label,  step + current_step)
    
# current_step += r_1_fc
# save_path = saver.save(domain_adaptation.sess, os.path.join(step1_model, "model_step1_{}.ckpt".format(current_step)))

# # saver.restore(domain_adaptation.sess, os.path.join(step1_model, "model_step1_{}.ckpt".format(4999)))

# domain_adaptation.set_logdir(step2_log)
# for step in (range(r_2_rec)):
#     if (step + 1) % save_iter == 0:
#         save_path = saver.save(domain_adaptation.sess, os.path.join(step2_model, "model_step2_{}.ckpt".format(step + current_step)))
#     batch_img, batch_label = mnist_data.next_batch()
#     batch_target, label_target = usps_data.next_batch()
#     domain_adaptation.run_step2(batch_img, batch_target, batch_label,  step + current_step)
# current_step += r_2_rec

# save_path = saver.save(domain_adaptation.sess, os.path.join(step2_model, "model_step2_{}.ckpt".format(current_step)))

saver.restore(domain_adaptation.sess, "/home/acm528/Minh/style_transfer/model/rerun_all_test_step3/step3/model_step3_29999.ckpt")
# domain_adaptation.duplicate_source_ae_to_target_ae()
domain_adaptation.set_logdir(step3_log)
for step in (range(r_3_df)):
    if (step + 1) % save_iter == 0:
        save_path = saver.save(domain_adaptation.sess, os.path.join(step3_model, "model_step3_{}.ckpt".format(step + current_step)))
    batch_img, batch_label = mnist_data.next_batch()
    batch_target, label_target = usps_data.next_batch()
    domain_adaptation.run_step3(batch_img, batch_target, batch_label,  step + current_step)
current_step += r_3_df

# domain_adaptation.set_logdir(step4_log)
# for step in (range(r_4_di)):
#     batch_img, batch_label = mnist_data.next_batch_train()
#     batch_target, label_target = usps_data.next_batch_train()
#     domain_adaptation.run_step4(batch_img, batch_target, batch_label,  step + current_step)