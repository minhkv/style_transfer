import tensorflow as tf
import numpy as np

lays = tf.layers

def feature_classifier(inputs):
    net = lays.dense(inputs, 128, activation=tf.nn.relu)
    net = lays.dense(net, 128, activation=tf.nn.relu)
    net = lays.dense(net, 10, activation=tf.nn.relu)
    return net
