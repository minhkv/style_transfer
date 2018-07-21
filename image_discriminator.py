import tensorflow as tf
import numpy as np
from discriminator import *

lays = tf.layers

class ImageDiscriminator(Discriminator):
    def model(self, inputs, labels):
        with tf.variable_scope("discriminator_{}".format(self.name), reuse=tf.AUTO_REUSE):
            with tf.variable_scope("feature_extract_{}".format(self.name), reuse=tf.AUTO_REUSE):
                net = lays.conv2d(inputs, 64, [5, 5], strides=1, padding='SAME', activation=tf.nn.relu, name="C1")
                net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S1")
                
                net = lays.conv2d(net, 128, [5, 5], strides=1, padding='VALID', activation=tf.nn.relu, name="C2")
                net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S2")
                
                net = lays.conv2d(net, 256, [5, 5], strides=1, padding='VALID', activation=tf.nn.relu, name="C3")
                # net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S3")
                net = lays.flatten(net, name="C3_flat")

            with tf.variable_scope("fully_connected_{}".format(self.name), reuse=tf.AUTO_REUSE):
                net = lays.dense(net, 256, activation=tf.nn.relu)
                net = lays.dense(net, 128, activation=tf.nn.relu)
                net = lays.dense(net, 10, activation=tf.nn.relu)
        return net
    
