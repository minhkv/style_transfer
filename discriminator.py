import tensorflow as tf
import numpy as np

lays = tf.layers

class Discriminator:
    def __init__(self, name="", lr = 0.0001):
        self.name = name
        self.lr = lr
        self._construct_graph()
        self._construct_loss()
        self._construct_summary()

    def model(self, inputs, labels):
        pass
    def _construct_graph(self):
        self.inputs = tf.placeholder(tf.float32, (None, 32, 32, 1), name="input_{}".format(self.name))  
        self.labels = tf.placeholder(tf.float32, (None, 10), name="label_{}".format(self.name))  
        self.logits = self.model(self.inputs, self.labels)

    def _construct_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        tf.add_to_collection("optimizer_{}".format(self.name), self.optimizer)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def _construct_summary(self):
        tf.summary.scalar("loss_d_{}".format(self.name), self.loss)

    def init_variable(self):
        # initialize the network
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def merge_all(self):
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('/tmp/log', self.sess.graph)
    
    def fit(self, inputs, labels, step):#, sess):
        labels = self.sess.run(tf.one_hot(labels, depth=10))
        summary, loss, _ = self.sess.run([self.merged, self.loss, self.optimizer], feed_dict={self.inputs: inputs, self.labels: labels})
        print("Iter {}: loss={}".format(step, loss))
        self.writer.add_summary(summary, step)
