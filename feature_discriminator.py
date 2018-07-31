import tensorflow as tf
import numpy as np
from discriminator import *

lays = tf.layers

class FeatureDiscriminator(Discriminator):
    def model(self, inputs):
        net = lays.dense(inputs, 128, activation=tf.nn.relu, name="F1")
        net = lays.dense(net, 128, activation=tf.nn.relu, name="F2")
        pred_class = lays.dense(net, 10, activation=tf.nn.relu, name="output_class")
        pred_class = tf.nn.softmax(pred_class, name="prob_class")
        pred_type = lays.dense(net, 1, activation=tf.nn.relu, name="output_type")
        return pred_class, pred_type
    
    def _construct_graph(self):
        self.inputs_source = self.endpoints["inputs_source"]
        self.inputs_target = self.endpoints["inputs_target"]
        self.vars_generator_source = self.endpoints["vars_generator_source"]
        self.vars_generator_target = self.endpoints["vars_generator_target"]
        self.class_labels = self.endpoints["class_labels"]
        
        with tf.variable_scope("discriminator_{}".format(self.name)) as scope:
            self.logits_source, self.type_pred_source = self.model(self.inputs_source)
            with tf.name_scope("class_source"):
                self.class_predict_source = tf.reshape(self.logits_source, (-1, 10))
                self.class_predict_source = tf.argmax(self.class_predict_source, axis=1, name="df_class_source")
        
        with tf.variable_scope(scope, reuse=True) as scope2:
            with tf.name_scope(scope2.original_name_scope):
                self.logits_target, self.type_pred_target = self.model(self.inputs_target)
                with tf.name_scope("class_target"):
                    self.class_predict_target = tf.reshape(self.logits_target, (-1, 10))
                    self.class_predict_target = tf.argmax(self.class_predict_target, axis=1, name="df_class_target")

    def _construct_loss(self):
        # Type loss: source or target
        # Adversarial learning
        # source: type 1, target type: 0
        with tf.variable_scope("loss_feature_discriminator_{}".format(self.name)):
            with tf.name_scope("loss_d"):
                self.loss_d_source = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.type_pred_source), self.type_pred_source)
                self.loss_d_target = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.type_pred_target), self.type_pred_target)
                self.loss_d_feature = tf.reduce_mean(0.5 * (self.loss_d_source + self.loss_d_target))
            with tf.name_scope("loss_g"):
                self.loss_g_feature_source = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.type_pred_source), self.type_pred_source)) # gen target common
                self.loss_g_feature_target = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(self.type_pred_target), self.type_pred_target)) # gen source common
                self.loss_g_feature = tf.reduce_mean(0.5 * (self.loss_g_feature_source + self.loss_g_feature_target))
            
            # tf.losses.sigmoid_cross_entropy(multi_class_labels = y_real, logits = y_predict)
            # Class loss: only for source 
            with tf.name_scope("class_loss"):
                self.class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_source, labels=self.class_labels))
            with tf.name_scope("total"):
                self.total_loss_g = self.loss_g_feature + self.class_loss
                self.total_loss_d = self.loss_d_feature + self.class_loss
        self.vars_g = self.vars_generator_source + self.vars_generator_target
        self.vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator_{}'.format(self.name))]
    
    def _construct_summary(self):
        tf.summary.scalar("type_loss_g_source_{}".format(self.name), self.loss_g_feature_source)
        tf.summary.scalar("type_loss_g_target_{}".format(self.name), self.loss_g_feature_target)
        tf.summary.scalar("type_loss_g_{}".format(self.name), self.loss_g_feature)
        tf.summary.scalar("type_loss_d_{}".format(self.name), self.loss_d_feature)
        tf.summary.scalar("type_loss_d_source_{}".format(self.name), self.loss_d_source)
        tf.summary.scalar("type_loss_d_target_{}".format(self.name), self.loss_d_target)
        tf.summary.scalar("class_loss_d_{}".format(self.name), self.class_loss)