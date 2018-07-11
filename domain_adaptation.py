import tensorflow as tf
# import tensorflow.layers as lays
import numpy as np
lays = tf.layers

class DomainAdaptation:
    def __init__(self, source_autoencoder, target_autoencoder, lr = 0.0001, name="domain_adaptation"):
        self.source_autoencoder = source_autoencoder
        self.target_autoencoder = target_autoencoder
        self.lr = lr
        self.name = name
        self._construct_graph()
        self._construct_summary()
        
    def _construct_graph(self):
        source_specific_latent = self.source_autoencoder.specific
        source_common_latent = self.source_autoencoder.common
        target_specific_latent = self.target_autoencoder.specific
        target_common_latent = self.target_autoencoder.common
        # Exchange feature
        spe_source_com_target = tf.concat([source_specific_latent, target_common_latent], 3)
        spe_target_com_source = tf.concat([target_specific_latent, source_common_latent], 3)
        
        self.img_spe_source_com_target = self.source_autoencoder.decoder(spe_source_com_target)
        self.img_spe_target_com_source = self.source_autoencoder.decoder(spe_target_com_source)
        
        # Feedback source
        feature_spe_source_com_target = self.source_autoencoder.encoder(self.img_spe_source_com_target)
        self.feedback_loss_source = tf.losses.mean_squared_error(spe_source_com_target, feature_spe_source_com_target)

        self.optimize_feedback_source = tf.train.AdamOptimizer(learning_rate=self.lr, name="feedback_source_optimize").minimize(self.feedback_loss_source)

        # feature_spe_target_com_source = self.source_autoencoder.encoder(self.img_spe_target_com_source)

    def _construct_summary(self):
        tf.summary.image("spe_source_com_target", self.img_spe_source_com_target)
        tf.summary.image("spe_target_com_source", self.img_spe_target_com_source)
        
    def merge_all(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('/tmp/log', self.sess.graph)
    def minimize_autoencoder(self, batch_source, batch_target, step):
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
    def minimize_feedback(self, batch_source, batch_target, step):
        source_inputs = self.source_autoencoder.ae_inputs
        target_inputs = self.target_autoencoder.ae_inputs
        print("Iter {}".format(step))
        summary, _ = self.sess.run([self.merged, self.optimize_feedback_source], feed_dict={source_inputs: batch_source, target_inputs: batch_target})
        self.train_writer.add_summary(summary, step)    
