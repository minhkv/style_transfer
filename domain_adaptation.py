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
        
    def feature_classifier(self, inputs):
        with tf.variable_scope("feature_classifier_{}".format(self.name), reuse=tf.AUTO_REUSE):
            net = lays.dense(inputs, 128, activation=tf.nn.relu)
            net = lays.dense(net, 128, activation=tf.nn.relu)
            net = lays.dense(net, 10, activation=tf.nn.relu)
        return net
    def _construct_graph(self):
        self.source_specific_latent = tf.placeholder(tf.float32, shape=(None, 1, 1, 128), name="source_specific_latent_{}".format(self.name))
        self.source_common_latent = tf.placeholder(tf.float32, shape=(None, 1, 1, 128), name="source_common_latent_{}".format(self.name))
        self.target_specific_latent = tf.placeholder(tf.float32, shape=(None, 1, 1, 128), name="target_specific_latent_{}".format(self.name))
        self.target_common_latent = tf.placeholder(tf.float32, shape=(None, 1, 1, 128), name="target_common_latent_{}".format(self.name))
        # Exchange feature
        spe_source_com_target = tf.concat([self.source_specific_latent, self.target_common_latent], 3)
        spe_target_com_source = tf.concat([self.target_specific_latent, self.source_common_latent], 3)
        
        self.img_spe_source_com_target = self.source_autoencoder.decoder(spe_source_com_target)
        self.img_spe_target_com_source = self.source_autoencoder.decoder(spe_target_com_source)
        
        # Feature classifier
        self.source_label = tf.placeholder(tf.float32, (None, 10), name="source_label_{}".format(self.name))  
        self.predict_source_common = self.feature_classifier(self.source_autoencoder.common)
        
        # Construct feedback
        self.feedback_loss_source = self._construct_feedback_loss(
            self.img_spe_source_com_target, 
            self.source_specific_latent, 
            self.target_common_latent,
            self.source_autoencoder)
            
        self.feedback_loss_target = self._construct_feedback_loss(
            self.img_spe_target_com_source, 
            self.target_specific_latent, 
            self.source_common_latent,
            self.target_autoencoder)
            
        # Feature classification loss
        self.loss_feature_classifier = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predict_source_common, labels=self.source_label), name="loss_feature_classifier")
        
        self.optimize_feature_classifier = tf.train.AdamOptimizer(learning_rate=self.lr, name="feature_classifier_optimize").minimize(self.loss_feature_classifier)
        self.optimize_feedback_source = tf.train.AdamOptimizer(learning_rate=self.lr, name="feedback_source_optimize").minimize(self.feedback_loss_source)
        self.optimize_feedback_target = tf.train.AdamOptimizer(learning_rate=self.lr, name="feedback_target_optimize").minimize(self.feedback_loss_target)

    def _construct_feedback_loss(self, gen_img, spe_latent, com_latent, autoencoder):
        feature_feedback = autoencoder.encoder(gen_img)
        spe, com = tf.split(feature_feedback, num_or_size_splits=2, axis=3)
        loss_spe = tf.losses.mean_squared_error(spe_latent, spe)
        loss_com = tf.losses.mean_squared_error(com_latent, com)
        loss = loss_spe + loss_com
        return loss
        
    def _construct_summary(self):
        tf.summary.image("spe_source_com_target", self.img_spe_source_com_target, 5)
        tf.summary.image("spe_target_com_source", self.img_spe_target_com_source, 5)
        tf.summary.scalar("feature_classifier_loss", self.loss_feature_classifier)
        tf.summary.scalar("feedback_loss_source", self.feedback_loss_source)
        tf.summary.scalar("feedback_loss_target", self.feedback_loss_target)
        
    def merge_all(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('/tmp/log', self.sess.graph)
        
    def _feed_dict(self, batch_source, batch_target, source_label=[]):
        spe_source, com_source = self.source_autoencoder.get_split_feature(batch_source, self.sess)
        spe_target, com_target = self.target_autoencoder.get_split_feature(batch_target, self.sess)
        s_label = np.ones((batch_source.shape[0], 10))
        if source_label != []:
            s_label = self.sess.run(tf.one_hot(source_label, depth=10))
        return {
            self.source_specific_latent: spe_source, 
            self.source_common_latent: com_source,
            self.target_specific_latent: spe_target, 
            self.target_common_latent: com_target, 
            self.source_autoencoder.ae_inputs: batch_source, 
            self.target_autoencoder.ae_inputs: batch_target,
            self.source_label: s_label
        }
    
    def run_optimize_feature_classifier(self, batch_source, batch_target, source_label, step):
        summary, loss_fc, _ = self.sess.run(
            [self.merged, self.loss_feature_classifier, self.optimize_feature_classifier],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: loss feature classifier: {:.4f}".format(step, loss_fc))
        self.train_writer.add_summary(summary, step)    
    
    def minimize_autoencoder(self, batch_source, batch_target, step, source_label=None):
        summary, loss_s, loss_t, a, b = self.sess.run(
            [self.merged, self.source_autoencoder.loss, self.target_autoencoder.loss, self.source_autoencoder.optimizer, self.target_autoencoder.optimizer], 
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: loss source: {:.4f}, loss target: {:.4f}".format(step, loss_s, loss_t))
        self.train_writer.add_summary(summary, step)    
        
    def minimize_feedback(self, batch_source, batch_target, step, source_label=None):
        
        summary, f_source, f_target, _, _ = self.sess.run(
            [self.merged, self.feedback_loss_source, self.feedback_loss_target, self.optimize_feedback_source, self.optimize_feedback_target], 
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: feedback source: {:.8f} feedback target: {:.8f}".format(step, f_source, f_target))
        self.train_writer.add_summary(summary, step)    
