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
step5_log = os.path.join(logdir, 'step5')
step6_log = os.path.join(logdir, 'step6')
step7_log = os.path.join(logdir, 'step7')

step1_model = os.path.join(model_folder, 'step1')
step2_model = os.path.join(model_folder, 'step2')
step3_model = os.path.join(model_folder, 'step3')
step4_model = os.path.join(model_folder, 'step4')
step5_model = os.path.join(model_folder, 'step5')
step6_model = os.path.join(model_folder, 'step6')
step7_model = os.path.join(model_folder, 'step7')

if not os.path.exists(step1_model):
    os.makedirs(step1_model)
if not os.path.exists(step2_model):
    os.makedirs(step2_model)
if not os.path.exists(step3_model):
    os.makedirs(step3_model)
if not os.path.exists(step4_model):
    os.makedirs(step4_model)
if not os.path.exists(step5_model):
    os.makedirs(step5_model)
if not os.path.exists(step6_model):
    os.makedirs(step6_model)
if not os.path.exists(step7_model):
    os.makedirs(step7_model)


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
variable_not_restored = domain_adaptation.image_discriminator_source.vars_d + domain_adaptation.image_discriminator_target.vars_d
variable_to_restore = [var for var in tf.trainable_variables()
    if var not in variable_not_restored]

saver = tf.train.Saver(max_to_keep=100, var_list=variable_to_restore)

# saver = tf.train.Saver(max_to_keep=100)

batch_size = 100  # Number of samples in each batch
usps_data = USPSDataset(batch_size=batch_size, sess=domain_adaptation.sess)
mnist_data = MNISTDataset(batch_size=batch_size, sess=domain_adaptation.sess)

mnist_data.sample_dataset(2000)
usps_data.sample_dataset(1800)

# mnist_data.one_hot_encoding_label()

r_1_fc = 15
r_2_rec = 5
r_3_df = 15
r_4_di = 5
r_5_ex_entropy = 8000
current_step = 57000

# saver.restore(domain_adaptation.sess, os.path.join(step1_model, "model_step1_{}.ckpt".format(999)))

# domain_adaptation.set_logdir(step1_log)
# for step in (range(r_1_fc)):
#     if (step + 1) % save_iter == 0:
#         save_path = saver.save(domain_adaptation.sess, os.path.join(step1_model, "model_step1_{}.ckpt".format(step)))
#     batch_img, batch_label = mnist_data.next_batch()
#     batch_target, label_target = usps_data.next_batch()
#     domain_adaptation.run_step1(batch_img, batch_target, batch_label, label_target,  step + current_step)
    
# current_step += r_1_fc
# save_path = saver.save(domain_adaptation.sess, os.path.join(step1_model, "model_step1_{}.ckpt".format(current_step)))

# domain_adaptation.set_logdir(step2_log)
# for step in (range(r_2_rec)):
#     if (step + 1) % save_iter == 0:
#         save_path = saver.save(domain_adaptation.sess, os.path.join(step2_model, "model_step2_{}.ckpt".format(step + current_step)))
#     batch_img, batch_label = mnist_data.next_batch()
#     batch_target, label_target = usps_data.next_batch()
#     domain_adaptation.run_step2(batch_img, batch_target, batch_label, label_target,  step + current_step)
# current_step += r_2_rec

# domain_adaptation.duplicate_source_ae_to_target_ae()

# domain_adaptation.set_logdir(step3_log)
# for step in (range(r_3_df)):
#     if (step + 1) % save_iter == 0:
#         save_path = saver.save(domain_adaptation.sess, os.path.join(step3_model, "model_step3_{}.ckpt".format(step + current_step)))
#     batch_img, batch_label = mnist_data.next_batch()
#     batch_target, label_target = usps_data.next_batch()
#     domain_adaptation.run_step3(batch_img, batch_target, batch_label, label_target,  step + current_step)
# current_step += r_3_df

# domain_adaptation.set_logdir(step4_log)
# for step in (range(r_4_di)):
#     if (step + 1) % save_iter == 0:
#         save_path = saver.save(domain_adaptation.sess, os.path.join(step4_model, "model_step4_{}.ckpt".format(step + current_step)))
#     batch_img, batch_label = mnist_data.next_batch()
#     batch_target, label_target = usps_data.next_batch()
#     domain_adaptation.run_step4(batch_img, batch_target, batch_label, label_target,  step + current_step)
# current_step += r_4_di

saver.restore(domain_adaptation.sess, "/content/drive/DaiHoc/ThucTap/Coding/style_transfer/model/change_summary/step5/model_step5_56999.ckpt")
saver = tf.train.Saver(max_to_keep=100)

domain_adaptation.set_logdir(step5_log)
for step in (range(r_5_ex_entropy)):
    if (step + 1) % save_iter == 0:
        save_path = saver.save(domain_adaptation.sess, os.path.join(step5_model, "model_step5_{}.ckpt".format(step + current_step)))
    batch_img, batch_label = mnist_data.next_batch()
    batch_target, label_target = usps_data.next_batch()
    domain_adaptation.run_step5(batch_img, batch_target, batch_label, label_target,  step + current_step)
    
    
    