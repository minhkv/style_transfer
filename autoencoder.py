import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# %matplotlib inline

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/home/minhkv/Documents/MachineLearning/tensorflow_tutorial/MNIST_data')

tf.reset_default_graph()



def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

class Autoencoder:
    def __init__(self):
        self.dec_in_channels = 1
        self.n_latent = 8

        self.reshaped_dim = [-1, 7, 7, self.dec_in_channels]
        self.inputs_decoder = 49 * self.dec_in_channels / 2
        self.construct()
        

    def lrelu(self, x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))

    def encoder(self, X_in, keep_prob):
        activation = lrelu
        with tf.variable_scope("encoder", reuse=None):
            X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
            x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.contrib.layers.flatten(x)
            mn = tf.layers.dense(x, units=self.n_latent)
            sd       = 0.5 * tf.layers.dense(x, units=self.n_latent)            
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent])) 
            z  = mn + tf.multiply(epsilon, tf.exp(sd))
            
            return z, mn, sd
    def decoder(self, sampled_z, keep_prob):
        with tf.variable_scope("decoder", reuse=None):
            x = tf.layers.dense(sampled_z, units=self.inputs_decoder, activation=lrelu)
            x = tf.layers.dense(x, units=self.inputs_decoder * 2 + 1, activation=lrelu)
            x = tf.reshape(x, self.reshaped_dim)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
            
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=28*28, activation=tf.nn.sigmoid)
            img = tf.reshape(x, shape=[-1, 28, 28])
            return img
    def construct(self):
        tf.reset_default_graph()
        self.X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
        h = self.X_in
        self.Y    = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
        Y_flat = tf.reshape(self.Y, shape=[-1, 28 * 28])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

        sampled, mn, sd = self.encoder(h, self.keep_prob)
        dec = self.decoder(sampled, self.keep_prob)

        unreshaped = tf.reshape(dec, [-1, 28*28])
        img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
        latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
        self.loss = tf.reduce_mean(img_loss + latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.0005).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        # gen_reshape = tf.reshape(dec, [-1, 28, 28, 1])
        # source_reshape = tf.reshape(self.X_in, [-1, 28, 28, 1])

        tf.summary.scalar('total_loss', self.loss)
        # tf.summary.image('gen_image', gen_reshape, 3)
        # tf.summary.image('source_image', X_in[0])
        self.merged = tf.summary.merge_all()
        
        self.train_writer = tf.summary.FileWriter('./log', self.sess.graph)

    def fit(self, batch, step):
        summary, _ = self.sess.run([self.merged, self.optimizer], feed_dict = {self.X_in: batch, self.Y: batch, self.keep_prob: 0.8})
        self.train_writer.add_summary(summary, step)
