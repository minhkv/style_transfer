import tensorflow as tf
import tensorflow.layers as lays
import numpy as np
import matplotlib.pyplot as plt

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))
def relu(x):
    return tf.nn.relu(x)
class Autoencoder:
    def __init__(self):
        self._construct_graph()
        self._construct_summary()

    def encoder(self, inputs):
        
        # encoder
        # C1: 1 x 32 x 32   ->  64 x 32 x 32
        # S1: 64 x 32 x 32  ->  64 x 16 x 16
        
        # C2: 64 x 16 x 16  -> 128 x 12 x 12 
        # S2: 128 x 12 x 12 -> 128 x 6 x 6
        
        # C3: 128 x 6 x 6   -> 256 x 2 x 2 
        # S3: 256 x 2 x 2   -> 256 x 1 x 1
        with tf.variable_scope("encoder", reuse=None):
            net = lays.conv2d(inputs, 64, [5, 5], strides=1, padding='SAME', name="C1")
            net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S1")
            
            net = lays.conv2d(net, 128, [5, 5], strides=1, padding='VALID', name="C2")
            net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S2")
            
            net = lays.conv2d(net, 256, [5, 5], strides=1, padding='VALID', name="C3")
            net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S3")
            return net

    def decoder(self, latent):
        # decoder
        # U3: 256 x 1 x 1   -> 256 x 2 x 2
        # D3: 256 x 2 x 2   -> 512 x 6 x 6
        
        # U2: 512 x 6 x 6   -> 512 x 12 x 12
        # D2: 512 x 12 x 12 -> 256 x 16 x 16
        
        # U1: 256 x 16 x 16 -> 256 x 32 x 32 
        # D1: 256 x 32 x 32 -> 128 x 32 x 32
        
        # output: 128 x 32 x 32 -> 1 x 32 x 32
        with tf.variable_scope("decoder", reuse=None):
            net = tf.image.resize_images(images=latent, size=[2, 2]) 
            net = lays.conv2d_transpose(net, 512, [5, 5], strides=1, padding='VALID', name="D3")
            
            net = tf.image.resize_images(images=net, size=[12, 12]) 
            net = lays.conv2d_transpose(net, 256, [5, 5], strides=1, padding='VALID', name="D2")
            
            net = tf.image.resize_images(images=net, size=[32, 32]) 
            net = lays.conv2d_transpose(net, 128, [5, 5], strides=1, padding='SAME', name="D1")
            
            net = lays.conv2d_transpose(net, 1, [5, 5], strides=1, padding='SAME', name="output")
            return net

    def _construct_graph(self):
        lr = 0.0001        # Learning rate
        
        tf.reset_default_graph()
        self.ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 1))  # input to the network (MNIST images)
        self.latent = self.encoder(self.ae_inputs)
        self.specific, self.common = tf.split(self.latent, num_or_size_splits=2, axis=3)
        self.ae_outputs = self.decoder(self.latent)  # create the Autoencoder network
        
        # calculate the loss and optimize the network
        self.loss = tf.reduce_mean(tf.square(self.ae_outputs - self.ae_inputs))  # claculate the mean square error loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        
        # initialize the network
        init = tf.global_variables_initializer()
        
        self.sess = tf.Session()
        self.sess.run(init)
        
    def _construct_summary(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.image('reconstructed', self.ae_outputs, 1)
        tf.summary.image('source', self.ae_inputs, 1)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('/tmp/log', self.sess.graph)

    def fit(self, batch, step):
        summary, total_loss, _ = self.sess.run([self.merged, self.loss, self.optimizer], feed_dict = {self.ae_inputs: batch})
        print("Iter {}: loss={}".format(step, total_loss))
        self.train_writer.add_summary(summary, step)
