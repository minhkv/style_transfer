import tensorflow as tf
import numpy as np
from discriminator import *

lays = tf.layers

class ImageDiscriminator(Discriminator):
    def model(self, inputs):
        
        with tf.variable_scope("feature_extract_{}".format(self.name), reuse=tf.AUTO_REUSE):
            with tf.name_scope('C1'):
                net = lays.conv2d(inputs, 64, [5, 5], strides=1, padding='SAME', name="C1",
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    bias_initializer=tf.constant_initializer(0.1))
                net = tf.nn.relu(net)
            net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S1")
            
            with tf.name_scope('C2'):
                net = lays.conv2d(net, 128, [5, 5], strides=1, padding='VALID', name="C2",
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    bias_initializer=tf.constant_initializer(0.1))
                net = tf.nn.relu(net)
            net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S2")
            with tf.name_scope('C3'):
                net = lays.conv2d(net, 256, [5, 5], strides=1, padding='VALID', name="C3",
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    bias_initializer=tf.constant_initializer(0.1))
                net = tf.nn.relu(net)
                # net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, name="S3")
                net = lays.flatten(net, name="C3_flat")

        with tf.variable_scope("fully_connected_{}".format(self.name), reuse=tf.AUTO_REUSE):
            with tf.name_scope('F1'):
                net = lays.dense(net, 256, name="F1",
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    bias_initializer=tf.constant_initializer(0.1))
                net = tf.contrib.layers.batch_norm(inputs= net, center=True, scale=True, is_training=True)
                net = tf.nn.relu(net)
            with tf.name_scope('F2'):
                net = lays.dense(net, 128, name="F2",
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    bias_initializer=tf.constant_initializer(0.1))
                net = tf.contrib.layers.batch_norm(inputs= net, center=True, scale=True, is_training=True)
                net = tf.nn.relu(net)
            pred_common = lays.dense(net, 128, activation=tf.nn.relu, name="common")
            pred_class = lays.dense(net, 10, activation=tf.nn.relu, name="output")
            pred_class = tf.nn.softmax(pred_class, name="prob_class")
            pred_type = lays.dense(net, 1, activation=tf.nn.sigmoid, name="type")
        return pred_common, pred_class, pred_type
        
    def _construct_graph(self):
        
        self.inputs_real = self.endpoints["inputs_real"]
        self.inputs_fake = self.endpoints["inputs_fake"]
        spe, self.inputs_common = tf.split(self.inputs_fake, num_or_size_splits=2, axis=3, name="split_{}".format(self.name))
        self.vars_generator = self.endpoints["vars_generator"]
        self.class_labels = self.endpoints["class_labels"]
        
        with tf.variable_scope("discriminator_{}".format(self.name)) as scope:
            self.logits_real, self.type_pred_real = self.model(self.inputs_real)
            with tf.name_scope("class_real"):
                self.class_predict_real = tf.reshape(self.logits_real, (-1, 10))
                self.class_predict_real = tf.argmax(self.class_predict_real, axis=1, name="df_class_real")
        with tf.variable_scope(scope, reuse=True) as scope2:
            with tf.name_scope(scope2.original_name_scope):
                self.logits_fake, self.type_pred_fake = self.model(self.inputs_fake)
                with tf.name_scope("class_fake"):
                    self.class_predict_fake = tf.reshape(self.logits_real, (-1, 10))
                    self.class_predict_fake = tf.argmax(self.class_predict_real, axis=1, name="df_class_fake")
        
    def _construct_loss(self):
        
        with tf.variable_scope("loss_image_discriminator_{}".format(self.name)):
            with tf.name_scope('loss_d'):
                loss_d_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.type_pred_real), self.type_pred_real)
                loss_d_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.type_pred_fake), self.type_pred_fake)
                self.loss_d_feature = (0.5 * (loss_d_real + loss_d_fake))
            with tf.name_scope('acc_type'):
                pred_type_real = tf.round(self.type_pred_real)
                pred_type_fake = tf.round(self.type_pred_fake)
                self.acc_type_real = tf.equal(tf.ones_like(self.type_pred_real), pred_type_real)
                self.acc_type_real = tf.reduce_mean(tf.cast(self.acc_type_real, tf.float32))
                self.acc_type_fake = tf.equal(tf.zeros_like(self.type_pred_fake), pred_type_fake)
                self.acc_type_fake = tf.reduce_mean(tf.cast(self.acc_type_fake, tf.float32))
                self.acc_type = 0.5 * (self.acc_type_real + self.acc_type_fake)
            with tf.name_scope('loss_g'):
                self.loss_g_feature = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(self.type_pred_fake), self.type_pred_fake)) # style transfer image
                
            with tf.name_scope('loss_class'):
                # Class loss: only for real 
                self.class_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_real, labels=self.class_labels))
                self.class_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_fake, labels=self.class_labels))
            with tf.name_scope('total'):
                # total
                self.loss_type = self.loss_g_feature + self.loss_d_feature
            with tf.name_scope('acc_class_real'):
                correct_prediction_real = tf.equal(tf.argmax(self.class_labels, 1), self.class_predict_real)
                correct_prediction_fake = tf.equal(tf.argmax(self.class_labels, 1), self.class_predict_fake)
                self.class_acc_real = tf.reduce_mean(tf.cast(correct_prediction_real, tf.float32))
                self.class_acc_fake = tf.reduce_mean(tf.cast(correct_prediction_fake, tf.float32))
        self.vars_g = self.vars_generator
        self.vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator_{}'.format(self.name))]

    def _construct_summary(self):
        with tf.name_scope('image_discriminator_{}'.format(self.name)):
            tf.summary.scalar('acc_type_{}'.format(self.name), self.acc_type)
            tf.summary.scalar('acc_class_real_{}'.format(self.name), self.class_acc_real)
            # tf.summary.scalar('loss_type_g_{}'.format(self.name), self.loss_g_feature)
            # tf.summary.scalar('loss_type_d_{}'.format(self.name), self.loss_d_feature)
            tf.summary.scalar('loss_type_{}'.format(self.name), self.loss_type)
            tf.summary.scalar('loss_class_real_{}'.format(self.name), self.class_loss_real)
            tf.summary.scalar('loss_class_fake_{}'.format(self.name), self.class_loss_fake)
        
        
