import tensorflow as tf
# import tensorflow.layers as lays
import numpy as np
import matplotlib.pyplot as plt
lays = tf.layers
def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))
def relu(x):
    return tf.nn.relu(x)
class Autoencoder:
    def __init__(self, name="", meta_graph=None, checkpoint_dir=None, lr = 0.0001):
        self.name = name
        self.lr = lr
        self.endpoints = {}
        if meta_graph == None:
            self._construct_graph()
            self._construct_loss()
            self._construct_summary()
        else:
            self._create_from_graph(meta_graph, checkpoint_dir)
            # self._construct_loss()
            # self._construct_summary()


    def encoder(self, inputs):
        # encoder
        # C1: 1 x 32 x 32   ->  64 x 32 x 32
        # S1: 64 x 32 x 32  ->  64 x 16 x 16
        
        # C2: 64 x 16 x 16  -> 128 x 12 x 12 
        # S2: 128 x 12 x 12 -> 128 x 6 x 6
        
        # C3: 128 x 6 x 6   -> 256 x 2 x 2 
        # S3: 256 x 2 x 2   -> 256 x 1 x 1
        # with tf.variable_scope("encoder_{}".format(self.name), reuse=tf.AUTO_REUSE):
        with tf.name_scope("C1"):
            net = lays.conv2d(inputs, 64, [5, 5], strides=1, padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            # net = tf.contrib.layers.batch_norm(inputs= net, center=True, scale=True, is_training=True)
            net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S1")
        
        with tf.name_scope("C2"):
            net = lays.conv2d(net, 128, [5, 5], strides=1, padding='VALID', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S2")
        
        with tf.name_scope("C3"):
            net = lays.conv2d(net, 256, [5, 5], strides=1, padding='VALID', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S3")
        # net = lays.dense(net, 256)
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
        # with tf.variable_scope("decoder_{}".format(self.name), reuse=tf.AUTO_REUSE):
        net = tf.image.resize_images(images=latent, size=[2, 2]) 
        with tf.name_scope("D3"):
            net = lays.conv2d_transpose(net, 512, [5, 5], strides=1, padding='VALID', 
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                bias_initializer=tf.constant_initializer(0.1))
            net = tf.contrib.layers.batch_norm(inputs= net, center=True, scale=True, is_training=True)
            net = tf.nn.relu(net)
        net = tf.image.resize_images(images=net, size=[12, 12]) 
        
        with tf.name_scope("D2"):
            net = lays.conv2d_transpose(net, 256, [5, 5], strides=1, padding='VALID', 
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                bias_initializer=tf.constant_initializer(0.1))
            # net = tf.contrib.layers.batch_norm(inputs= net, center=True, scale=True, is_training=True)
            net = tf.nn.relu(net)
        net = tf.image.resize_images(images=net, size=[32, 32]) 
        
        with tf.name_scope("D1"):
            net = lays.conv2d_transpose(net, 128, [5, 5], strides=1, padding='SAME', 
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                bias_initializer=tf.constant_initializer(0.1))
            net = tf.contrib.layers.batch_norm(inputs= net, center=True, scale=True, is_training=True)
            net = tf.nn.relu(net)
        self.endpoints['D1'] = net
        with tf.name_scope("output"):
            net = lays.conv2d_transpose(net, 1, [5, 5], strides=1, padding='SAME',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                bias_initializer=tf.constant_initializer(0.1))
            net = tf.contrib.layers.batch_norm(inputs= net, center=True, scale=True, is_training=True)
            net = tf.nn.sigmoid(net)
        return net
    
    def _create_from_graph(self, meta_graph, checkpoint_dir):
        print("[LOG] Construct from graph")
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(meta_graph)
        saver.restore(self.sess, checkpoint_dir)
        graph = tf.get_default_graph()

        self.ae_inputs = graph.get_tensor_by_name("input_{}:0".format(self.name))
        self.latent = graph.get_tensor_by_name("encoder_{0}/S3_{0}/MaxPool:0".format(self.name))
        self.ae_outputs = graph.get_tensor_by_name("decoder_{0}/output_{0}/BiasAdd:0".format(self.name))

        self.loss = tf.losses.mean_squared_error(self.ae_inputs, self.ae_outputs, scope="loss_{}".format(self.name))
        self.optimizer = tf.get_collection("optimizer_{}".format(self.name))[0]

    def _construct_graph(self):
        with tf.variable_scope("data_{}".format(self.name)):
            self.ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 1), name="input_{}".format(self.name))  # input to the network (MNIST images)
        with tf.variable_scope("encoder_{}".format(self.name)) as encoder_scope:
            self.latent = self.encoder(self.ae_inputs)
            self.encoder_scope = encoder_scope
        with tf.variable_scope("decoder_{}".format(self.name)) as decoder_scope:
            self.ae_outputs = self.decoder(self.latent)  # create the Autoencoder network
            self.decoder_scope = decoder_scope
        with tf.variable_scope("latent_{}".format(self.name)):
            self.specific, self.common = tf.split(self.latent, num_or_size_splits=2, axis=3, name="split_{}".format(self.name))
        self.endpoints['latent'] = self.latent
        self.endpoints['specific'] = self.specific
        self.endpoints['common'] = self.common

    def _construct_loss(self):
        # calculate the loss and optimize the network
        with tf.variable_scope("loss_autoencoder_{}".format(self.name)):
            self.loss = tf.losses.mean_squared_error(self.ae_inputs, self.ae_outputs)
    def _construct_summary(self):
        tf.summary.scalar('loss_reconstruct_{}'.format(self.name), self.loss)
        tf.summary.image('reconstructed_{}'.format(self.name), self.ae_outputs, 3)
        tf.summary.image('inputs_{}'.format(self.name), self.ae_inputs, 3)

    def get_split_feature(self, batch, sess):
        return sess.run([self.specific, self.common], feed_dict = {self.ae_inputs: batch})
