from mnist import *
from usps import *
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

mnist = MNISTDataset(100, sess)
usps = USPSDataset(100, sess)
mnist.sample_dataset(2000)
usps.sample_dataset(1800)

# for i in range(2000):
#     mnist.next_batch()
# img, label = mnist.next_batch()

mnist_label = (usps.get_all_label()[:1800])
with open("usps_meta.txt", "w") as f:
    for l in mnist_label:
        f.write("{}\n".format(l))