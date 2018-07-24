import tensorflow as tf
# import tensorflow.layers as lays
import numpy as np
from feature_discriminator import *
from image_discriminator import *
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
        self.source_label = tf.placeholder(tf.float32, (None, 10), name="source_label_{}".format(self.name)) 
        
        # Autoencoder varlist
        vars_encoder_source = [var for var in tf.trainable_variables() if var.name.startswith('encoder_{}'.format(self.source_autoencoder.name))]
        vars_encoder_target = [var for var in tf.trainable_variables() if var.name.startswith('encoder_{}'.format(self.target_autoencoder.name))]
        vars_decoder_source = [var for var in tf.trainable_variables() if var.name.startswith('decoder_{}'.format(self.source_autoencoder.name))]
        vars_decoder_target = [var for var in tf.trainable_variables() if var.name.startswith('decoder_{}'.format(self.target_autoencoder.name))]
        
        # Exchange feature
        spe_source_com_target = tf.concat([self.source_specific_latent, self.target_common_latent], 3)
        spe_target_com_source = tf.concat([self.target_specific_latent, self.source_common_latent], 3)
        
        
        with tf.variable_scope(self.source_autoencoder.decoder_scope, reuse=True) as scope:
            with tf.name_scope(scope.original_name_scope):
                print("[Info] Decode exchanged feature by decoder_{}".format(self.source_autoencoder.name))
                self.img_spe_source_com_target = self.source_autoencoder.decoder(spe_source_com_target)
                
        with tf.variable_scope(self.target_autoencoder.decoder_scope, reuse=True) as scope:
            with tf.name_scope(scope.original_name_scope):
                print("[Info] Decode exchanged feature by decoder_{}".format(self.target_autoencoder.name))
                self.img_spe_target_com_source = self.target_autoencoder.decoder(spe_target_com_source)
        
        # Feature classifier
        self.predict_source_common = self.feature_classifier(self.source_autoencoder.common)
        
        # Feature discriminator
        self.feature_discriminator = FeatureDiscriminator(
            name="df", 
            endpoints={
                "inputs_source": self.source_autoencoder.common,
                "inputs_target": self.target_autoencoder.common,
                "vars_generator_source": vars_encoder_source,
                "vars_generator_target": vars_encoder_target,
                "class_labels": self.source_label
            }
        )
        # Image discriminator
        self.image_discriminator_source = ImageDiscriminator(
            name="ds",
            endpoints={
                "inputs_real": self.source_autoencoder.ae_outputs,
                "inputs_fake": self.img_spe_source_com_target,
                "vars_generator": vars_decoder_source,
                "class_labels": self.source_label
            }
        )
        
        self.image_discriminator_target = ImageDiscriminator(
            name="dt",
            endpoints={
                "inputs_real": self.target_autoencoder.ae_outputs,
                "inputs_fake": self.img_spe_target_com_source,
                "vars_generator": vars_decoder_target,
                "class_labels": self.source_label
            }
        )
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
        
        # Feedback losses
        self.optimize_feedback_source = tf.train.AdamOptimizer(learning_rate=self.lr, name="feedback_source_optimize").minimize(self.feedback_loss_source)
        self.optimize_feedback_target = tf.train.AdamOptimizer(learning_rate=self.lr, name="feedback_target_optimize").minimize(self.feedback_loss_target)

    def _construct_feedback_loss(self, gen_img, spe_latent, com_latent, autoencoder):
        print("[Info] Construct feedback {}".format(autoencoder.name))
        with tf.variable_scope(autoencoder.encoder_scope, reuse=tf.AUTO_REUSE) as scope:
            with tf.name_scope(scope.original_name_scope):
                feature_feedback = autoencoder.encoder(gen_img)
                spe, com = tf.split(feature_feedback, num_or_size_splits=2, axis=3)
                loss_spe = tf.losses.mean_squared_error(spe_latent, spe)
                loss_com = tf.losses.mean_squared_error(com_latent, com)
                loss = loss_spe + loss_com
        return loss
        
    def _construct_summary(self):
        tf.summary.image("spe_source_com_target", self.img_spe_source_com_target, 3)
        tf.summary.image("spe_target_com_source", self.img_spe_target_com_source, 3)
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
    
    # Image discriminator
    def run_optimize_image_discriminator_source_class(self, batch_source, batch_target, source_label, step):
        summary, loss_df, _ = self.sess.run(
            [self.merged, self.image_discriminator_source.class_loss,self.image_discriminator_source.optimizer_class],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: loss image discriminator class: {:.8f}".format(step, loss_df))
        self.train_writer.add_summary(summary, step)
    
    def run_optimize_image_discriminator_source_type(self, batch_source, batch_target, source_label, step):
        summary, loss_df, _ = self.sess.run(
            [self.merged, self.image_discriminator_source.class_loss,self.image_discriminator_source.optimizer_class],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: loss image discriminator class: {:.8f}".format(step, loss_df))
        self.train_writer.add_summary(summary, step)    
    
    
    # Feature discriminator
    def run_optimize_feature_discriminator_class(self, batch_source, batch_target, source_label, step):
        summary, loss_df, _ = self.sess.run(
            [self.merged, self.feature_discriminator.class_loss,self.feature_discriminator.optimizer_class],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: loss feature discriminator class: {:.8f}".format(step, loss_df))
        self.train_writer.add_summary(summary, step)    
    
    def run_optimize_feature_discriminator_type_g(self, batch_source, batch_target, source_label, step):
        summary, loss_df_type_g, _ = self.sess.run(
            [
                self.merged, 
                self.feature_discriminator.loss_g_feature, 
                self.feature_discriminator.optimizer_g_feature
                ],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: typeloss, gen: {:.8f}".format(step, loss_df_type_g))
        self.train_writer.add_summary(summary, step)    
    
    def run_optimize_feature_discriminator_type_d(self, batch_source, batch_target, source_label, step):
        summary, loss_df_type_d, _ = self.sess.run(
            [
                self.merged, 
                self.feature_discriminator.loss_d_feature, 
                self.feature_discriminator.optimizer_d_feature
            ],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: typeloss, dis: {:.8f}".format(step, loss_df_type_d))
        self.train_writer.add_summary(summary, step)   
    # Feature classifier
    def run_optimize_feature_classifier(self, batch_source, batch_target, source_label, step):
        summary, loss_fc, _ = self.sess.run(
            [self.merged, self.loss_feature_classifier, self.optimize_feature_classifier],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: loss feature classifier: {:.4f}".format(step, loss_fc))
        self.train_writer.add_summary(summary, step)    
    # Autoencoder
    def minimize_autoencoder(self, batch_source, batch_target, step, source_label=None):
        summary, loss_s, loss_t, a, b = self.sess.run(
            [self.merged, self.source_autoencoder.loss, self.target_autoencoder.loss, self.source_autoencoder.optimizer, self.target_autoencoder.optimizer], 
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: loss source: {:.4f}, loss target: {:.4f}".format(step, loss_s, loss_t))
        self.train_writer.add_summary(summary, step)    
    # Feedback
    def minimize_feedback(self, batch_source, batch_target, step, source_label=None):
        summary, f_source, f_target, _, _ = self.sess.run(
            [self.merged, self.feedback_loss_source, self.feedback_loss_target, self.optimize_feedback_source, self.optimize_feedback_target], 
            feed_dict=self._feed_dict(batch_source, batch_target, source_label)
        )
        print("Iter {}: feedback source: {:.8f} feedback target: {:.8f}".format(step, f_source, f_target))
        self.train_writer.add_summary(summary, step)    
