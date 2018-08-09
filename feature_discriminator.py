import tensorflow as tf
import numpy as np
from discriminator import *

lays = tf.layers

class FeatureDiscriminator(Discriminator):
    def model(self, inputs):
        with tf.name_scope("F1"):
            net = lays.dense(inputs, 128, name="F1",
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                bias_initializer=tf.constant_initializer(0.1))
            net = tf.contrib.layers.batch_norm(inputs= net, center=True, scale=True, is_training=True)
            net = tf.nn.relu(net)
        tf.summary.histogram("F1_activation_df", net)
        with tf.name_scope("F2"):
            net = lays.dense(net, 128, name="F2",
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                bias_initializer=tf.constant_initializer(0.1))
            net = tf.contrib.layers.batch_norm(inputs= net, center=True, scale=True, is_training=True)
            net = tf.nn.relu(net)
            
        self.endpoints['F2_df'] = net
        
        tf.summary.histogram("F2_activation_df", net)
        pred_class = lays.dense(net, 10, activation=tf.nn.relu, name="output_class")
        pred_class = tf.nn.softmax(pred_class, name="prob_class")
        pred_type = lays.dense(net, 1, activation=tf.nn.sigmoid, name="output_type")
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

            with tf.name_scope("acc_type"):
                pred_type_source = tf.round(self.type_pred_source)
                pred_type_target = tf.round(self.type_pred_target)
                self.acc_type_source = tf.equal(tf.ones_like(self.type_pred_source), pred_type_source)
                self.acc_type_source = tf.reduce_mean(tf.cast(self.acc_type_source, tf.float32))
                self.acc_type_target = tf.equal(tf.zeros_like(self.type_pred_target), pred_type_target)
                self.acc_type_target = tf.reduce_mean(tf.cast(self.acc_type_target, tf.float32))
                self.acc_type = 0.5 * (self.acc_type_source + self.acc_type_target)
            with tf.name_scope("loss_g"):
                self.loss_g_feature_source = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.type_pred_source), self.type_pred_source)) # gen target common
                self.loss_g_feature_target = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(self.type_pred_target), self.type_pred_target)) # gen source common
                self.loss_g_feature = tf.reduce_mean(0.5 * (self.loss_g_feature_source + self.loss_g_feature_target))
            
            # tf.losses.sigmoid_cross_entropy(multi_class_labels = y_real, logits = y_predict)
            # Class loss: only for source 
            with tf.name_scope("class_loss"):
                self.class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_source, labels=self.class_labels))
                
            with tf.name_scope("acc_class_source"):
                correct_prediction = tf.equal(tf.argmax(self.class_labels, 1), self.class_predict_source)
                self.class_acc_source = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            with tf.name_scope("total"):
                self.total_loss_g = self.loss_g_feature #+ self.class_loss 
                self.total_loss_d = self.loss_d_feature + self.class_loss
                self.loss_df_type = self.loss_g_feature + self.loss_d_feature
            self.vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator_{}'.format(self.name))]

    def _construct_summary(self):
        # tf.summary.scalar("type_loss_g_source_{}".format(self.name), self.loss_g_feature_source)
        # tf.summary.scalar("type_loss_g_target_{}".format(self.name), self.loss_g_feature_target)
        with tf.name_scope('feature_discriminator'):
            # tf.summary.scalar("loss_df_g_{}".format(self.name), self.loss_g_feature)
            # tf.summary.scalar("loss_df_d_{}".format(self.name), self.loss_d_feature)
            tf.summary.scalar("loss_df_type", self.loss_df_type)
            # tf.summary.scalar("loss_df_type_source_d_{}".format(self.name), self.loss_d_source)
            # tf.summary.scalar("loss_df_type_target_d_{}".format(self.name), self.loss_d_target)
            tf.summary.scalar("loss_df_class_{}".format(self.name), self.class_loss)
            tf.summary.scalar("acc_df_class_{}".format(self.name), self.class_acc_source)
            # tf.summary.scalar("acc_type_source_d", self.acc_type_source)
            # tf.summary.scalar("acc_type_target_d", self.acc_type_target)
            tf.summary.scalar("acc_type", self.acc_type)
            # tf.summary.histogram("type_pred_source", self.type_pred_source)
            # tf.summary.histogram("type_pred_target", self.type_pred_target)
        
        
        