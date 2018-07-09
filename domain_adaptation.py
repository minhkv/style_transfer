import tensorflow as tf
# import tensorflow.layers as lays
import numpy as np
lays = tf.layers

class DomainAdaptation:
    def __init__(self, source_autoencoder, target_autoencoder):
        self.source_autoencoder = source_autoencoder
        self.target_autoencoder = target_autoencoder
        self._construct_graph()
        self._construct_summary()
        
    def _construct_graph(self):
        source_specific_latent = self.source_autoencoder.specific
        source_common_latent = self.source_autoencoder.common
        target_specific_latent = self.target_autoencoder.specific
        target_common_latent = self.target_autoencoder.common

        latent_transfer_to_source = tf.concat([source_specific_latent, target_common_latent], 3)
        latent_transfer_to_target = tf.concat([target_specific_latent, source_common_latent], 3)
        
        self.gen_transfer_to_source = self.source_autoencoder.decoder(latent_transfer_to_source)

    def _construct_summary(self):
        tf.summary.image("gen_transfer_to_source", self.gen_transfer_to_source)
    def merge_all(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('/tmp/log', self.sess.graph)
    def fit(self, batch_source, batch_target, step):
        source_inputs = self.source_autoencoder.ae_inputs
        source_loss = self.source_autoencoder.loss
        source_optimizer = self.source_autoencoder.optimizer
        
        target_inputs = self.target_autoencoder.ae_inputs
        target_loss = self.target_autoencoder.loss
        target_optimizer = self.target_autoencoder.optimizer
        
        summary, loss_s, loss_t, a, b = self.sess.run(
            [self.merged, source_loss, target_loss, source_optimizer, target_optimizer], 
            feed_dict={source_inputs: batch_source, target_inputs: batch_target})
        print("Iter {}: loss source: {:.4f}, loss target: {:.4f}".format(step, loss_s, loss_t))
        self.train_writer.add_summary(summary, step)        
