import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# %matplotlib inline
from tensorflow.examples.tutorials.mnist import input_data

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))
def relu(x):
    return tf.nn.relu(x)
class Autoencoder:
    def __init__(self):
        self.dec_in_channels = 1
        self.n_latent = 8

        self.reshaped_dim = [-1, 7, 7, self.dec_in_channels]
        self.inputs_decoder = 49 * self.dec_in_channels / 2
        self.construct()

    def encoder(self, X_in, keep_prob):
        
        with tf.variable_scope("encoder", reuse=None):
            X = tf.reshape(X_in, shape=[-1, 32, 32, 1])

            x = tf.layers.conv2d(X, filters=64, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu, name="C1")
            x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2, name="S1")

            x = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu, name="C2")
            x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2, name="S2")

            x = tf.layers.conv2d(x, filters=256, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu, name="C3")
            x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2, name="S3")

            x = tf.contrib.layers.flatten(x)
            print(x)
            mn          = tf.layers.dense(x, units=self.n_latent)
            sd          = 0.5 * tf.layers.dense(x, units=self.n_latent)            
            epsilon     = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent])) 
            z           = mn + tf.multiply(epsilon, tf.exp(sd))
            return x, mn, sd

    def decoder(self, sampled_z, keep_prob):
        with tf.variable_scope("decoder", reuse=None):
            x = tf.layers.dense(sampled_z, units=256, activation=relu)
            x = tf.reshape(x, shape=[-1, 1, 1, 256])
            # x = tf.layers.dense(x, units=self.inputs_decoder * 2 + 1, activation=relu)
            print(x)
            x = tf.image.resize_images(images=x, size=[2, 2]) #U3
            # tf.keras.layers.UpSampling2D(x, )
            print(x)
            
            x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
            x = tf.image.resize_images(images=x, size=[12, 12])

            x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
            x = tf.image.resize_images(images=x, size=[32, 32])

            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
            
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=32*32, activation=tf.nn.sigmoid)
            img = tf.reshape(x, shape=[32, 32])
            return img

    def construct(self):
        tf.reset_default_graph()
        self.X_in = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32], name='X')
        h = self.X_in
        self.Y    = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32], name='Y')
        Y_flat = tf.reshape(self.Y, shape=[-1, 32 * 32])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

        sampled, mn, sd = self.encoder(h, self.keep_prob)
        dec = self.decoder(sampled, self.keep_prob)

        unreshaped = tf.reshape(dec, [-1, 32*32])
        img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
        latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
        self.loss = tf.reduce_mean(img_loss + latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.0005).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        gen_reshape = tf.reshape(dec, [-1, 32, 32, 1])
        source_reshape = tf.reshape(self.X_in, [-1, 32, 32, 1])

        tf.summary.scalar('total_loss', self.loss)
        tf.summary.image('gen_image', gen_reshape, 1)
        tf.summary.image('source_image', source_reshape, 1)
        self.merged = tf.summary.merge_all()
        
        self.train_writer = tf.summary.FileWriter('./log', self.sess.graph)

    def fit(self, batch, step):
        summary, _ = self.sess.run([self.merged, self.optimizer], feed_dict = {self.X_in: batch, self.Y: batch, self.keep_prob: 0.8})
        self.train_writer.add_summary(summary, step)
