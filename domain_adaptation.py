import tensorflow as tf
# import tensorflow.layers as lays
import numpy as np
import scipy
import os
from feature_discriminator import *
from image_discriminator import *
from sklearn.metrics import accuracy_score
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.manifold import TSNE
from tsne_utils import *

lays = tf.layers
def entropy_function(pk):
    sm_pk = tf.nn.softmax(pk)
    return tf.losses.softmax_cross_entropy(sm_pk, pk)
def one_hot_encoding(labels, depth=10):
    one_hot_labels = np.zeros((len(labels), depth))
    for o, l in zip(one_hot_labels, labels):
        if l >= depth:
            print("[Error] Label larger than depth")
        o[l] = 1
    return one_hot_labels

class DomainAdaptation:
    def __init__(self, source_autoencoder, target_autoencoder, lr = 0.01, name="domain_adaptation", logdir="/tmp/log", batch_size=100, gpu_fraction=0.7):
        self.source_autoencoder = source_autoencoder
        self.target_autoencoder = target_autoencoder
        
        self.lr = lr
        self.name = name
        self.logdir = logdir
        self.batch_size = batch_size
        self.gpu_fraction = gpu_fraction
        self.feature = np.zeros((0, 1, 1, 128))
        self.meta_data = []
        self.embedding_dir = os.path.join(self.logdir, 'common')
        self._construct_graph()
        self._construct_loss()
        self._construct_optimizer()
        self._construct_summary()
        
    def feature_classifier(self, inputs):
        with tf.name_scope("F1"):
            net = lays.dense(inputs, 128, 
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                bias_initializer=tf.constant_initializer(0.1))
            net = tf.contrib.layers.batch_norm(inputs= net, center=True, scale=True, is_training=True)
            net = tf.nn.relu(net)
        with tf.name_scope("F2"):
            net = lays.dense(net, 128, 
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                bias_initializer=tf.constant_initializer(0.1))
            net = tf.contrib.layers.batch_norm(inputs= net, center=True, scale=True, is_training=True)
            net = tf.nn.relu(net)
        with tf.name_scope("output"):
            net = lays.dense(net, 10, 
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                bias_initializer=tf.constant_initializer(0.1))
            net = tf.nn.relu(net)
        return net
    
    def _construct_graph(self):

        # Exchange feature
        with tf.variable_scope("feature_exchange_to_source"):
            self.source_specific_latent = self.source_autoencoder.specific
            self.target_common_latent = self.target_autoencoder.common
            spe_source_com_target = tf.concat([self.source_specific_latent, self.target_common_latent], 3)
        with tf.variable_scope("feature_exchange_to_target"):
            self.target_specific_latent = self.target_autoencoder.specific
            self.source_common_latent = self.source_autoencoder.common
            spe_target_com_source = tf.concat([self.target_specific_latent, self.source_common_latent], 3)
        with tf.variable_scope("source_label"):
            self.source_label = tf.placeholder(tf.float32, (None, 10), name="source_label_{}".format(self.name)) 
        with tf.variable_scope("target_label"):
            self.target_label = tf.placeholder(tf.float32, (None, 10), name="target_label_{}".format(self.name)) 
        
        
        # Autoencoder varlist
        self.vars_encoder_source = [var for var in tf.trainable_variables() if var.name.startswith('encoder_{}'.format(self.source_autoencoder.name))]
        self.vars_encoder_target = [var for var in tf.trainable_variables() if var.name.startswith('encoder_{}'.format(self.target_autoencoder.name))]
        self.vars_decoder_source = [var for var in tf.trainable_variables() if var.name.startswith('decoder_{}'.format(self.source_autoencoder.name))]
        self.vars_decoder_target = [var for var in tf.trainable_variables() if var.name.startswith('decoder_{}'.format(self.target_autoencoder.name))]
        self.vars_feature_classifier = [var for var in tf.trainable_variables() if var.name.startswith('feature_classifier_{}'.format(self.name))]

        # Feed target input to source ae
        with tf.variable_scope(self.source_autoencoder.encoder_scope, reuse=True) as scope:
            with tf.name_scope(scope.original_name_scope):
                print("[Info] Encode target data by encoder_{}".format(self.source_autoencoder.name))
                self.latent_source_ae_target_data = self.source_autoencoder.encoder(self.target_autoencoder.ae_inputs)
                
        with tf.variable_scope(self.source_autoencoder.decoder_scope, reuse=True) as scope:
            with tf.name_scope(scope.original_name_scope):
                print("[Info] Decode target data by decoder_{}".format(self.source_autoencoder.name))
                self.reconstruct_source_target_data = self.source_autoencoder.decoder(self.latent_source_ae_target_data)

        with tf.variable_scope(self.source_autoencoder.decoder_scope, reuse=True) as scope:
            with tf.name_scope(scope.original_name_scope):
                print("[Info] Decode exchanged feature by decoder_{}".format(self.source_autoencoder.name))
                self.img_spe_source_com_target = self.source_autoencoder.decoder(spe_source_com_target)
                
        with tf.variable_scope(self.target_autoencoder.decoder_scope, reuse=True) as scope:
            with tf.name_scope(scope.original_name_scope):
                print("[Info] Decode exchanged feature by decoder_{}".format(self.target_autoencoder.name))
                self.img_spe_target_com_source = self.target_autoencoder.decoder(spe_target_com_source)
        
        # Feature classifier
        with tf.variable_scope("feature_classifier_{}".format(self.name)):
            self.predict_source_common = self.feature_classifier(self.source_autoencoder.common)
            self.predict_source_common = tf.reshape(self.predict_source_common, (-1, 10))
            
        with tf.variable_scope("feature_classifier_{}".format(self.name), reuse=True):
            self.predict_target_common = self.feature_classifier(self.target_autoencoder.common)
            self.predict_target_common = tf.reshape(self.predict_target_common, (-1, 10))
            
        
        # Feature discriminator
        self.feature_discriminator = FeatureDiscriminator(
            name="df", 
            endpoints={
                "inputs_source": self.source_autoencoder.common,
                "inputs_target": self.target_autoencoder.common,
                "vars_generator_source": self.vars_encoder_source,
                "vars_generator_target": self.vars_encoder_target,
                "class_labels_source": self.source_label,
                "class_labels_target": self.target_label
            }
        )
        # Image discriminator
        self.image_discriminator_source = ImageDiscriminator(
            name="ds",
            endpoints={
                "inputs_real": self.source_autoencoder.ae_outputs,
                "inputs_fake": self.img_spe_source_com_target,
                "common": self.target_autoencoder.common,
                "vars_generator": self.vars_decoder_source,
                "class_labels_real": self.source_label,
                "class_labels_fake": self.target_label
            }
        )
        
        self.image_discriminator_target = ImageDiscriminator(
            name="dt",
            endpoints={
                "inputs_real": self.target_autoencoder.ae_outputs,
                "inputs_fake": self.img_spe_target_com_source,
                "common": self.source_autoencoder.common,
                "vars_generator": self.vars_decoder_target,
                "class_labels_real": self.target_label,
                "class_labels_fake": self.source_label
            }
        )
        
    def _construct_loss(self):
        with tf.name_scope("reconstruct"):
            self.loss_reconstruct = self.source_autoencoder.loss + self.target_autoencoder.loss
        # Construct feedback
        with tf.name_scope("feedback"):
            self.feedback_L2_source = self._construct_feedback_loss(
                self.img_spe_source_com_target, 
                self.source_specific_latent, 
                self.target_common_latent,
                self.source_autoencoder)
                
            self.feedback_L2_target = self._construct_feedback_loss(
                self.img_spe_target_com_source, 
                self.target_specific_latent, 
                self.source_common_latent,
                self.target_autoencoder)
            
            self.total_feedback = self.feedback_L2_source + self.feedback_L2_target
        
        with tf.variable_scope("loss_autoencoder_{}".format(self.source_autoencoder.name)) as scope:
            with tf.name_scope(scope.original_name_scope):
                self.loss_reconstruct_source_img_target = tf.losses.mean_squared_error(self.target_autoencoder.ae_inputs, self.reconstruct_source_target_data)
        
        with tf.variable_scope("loss_feature_classifier"):
            # Feature classification loss
            self.loss_fc_source = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predict_source_common, labels=self.source_label), name="loss_fc_source")
            self.loss_fc_target = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predict_target_common, labels=self.target_label), name="loss_fc_target")
            correct_prediction_source = tf.equal(tf.argmax(self.source_label, 1), tf.argmax(self.predict_source_common, 1))
            self.fc_accuracy_source = tf.reduce_mean(tf.cast(correct_prediction_source, tf.float32))
            
            correct_prediction_target = tf.equal(tf.argmax(self.target_label, 1), tf.argmax(self.predict_target_common, 1))
            self.fc_accuracy_target = tf.reduce_mean(tf.cast(correct_prediction_target, tf.float32))
            
        with tf.variable_scope("loss_entropy"):
            loss_entropy_gs = entropy_function(self.image_discriminator_source.logits_fake)
            loss_entropy_xt = entropy_function(self.image_discriminator_target.logits_real)
            loss_entropy_ct = entropy_function(self.feature_discriminator.logits_target)
            # loss_entropy_ct = tf.reshape(loss_entropy_ct, (-1, 10))
            self.loss_entropy = tf.reduce_mean(loss_entropy_gs + loss_entropy_xt + loss_entropy_ct)
        with tf.variable_scope("loss_semantic"):
            ds_gs = self.image_discriminator_source.logits_fake
            dt_xt = self.image_discriminator_target.logits_real
            self.loss_semantic = tf.losses.mean_squared_error(ds_gs, dt_xt)
        
        with tf.name_scope("Step1"):
            self.loss_step1 = self.loss_fc_source
            var_step1 = self.vars_encoder_source + self.vars_feature_classifier
            self.optimizer_step1 = tf.train.GradientDescentOptimizer(learning_rate=0.01, name="optimize_1").minimize(self.loss_step1, var_list=var_step1)
            
        with tf.name_scope("Step2"):
            self.loss_step2 = self.loss_fc_source + self.source_autoencoder.loss + self.loss_reconstruct_source_img_target
            var_step2 = self.vars_feature_classifier + self.vars_encoder_source + self.vars_decoder_source
            self.optimizer_step2 = tf.train.GradientDescentOptimizer(learning_rate=0.01, name="optimize_2").minimize(self.loss_step2, var_list=var_step2)
            
        with tf.name_scope("Step3"):
            self.loss_step3_g = 10 * self.loss_fc_source + self.source_autoencoder.loss + self.target_autoencoder.loss + self.feature_discriminator.loss_g_feature
            #self.loss_step3_d = 10 * self.loss_fc_source + self.feature_discriminator.total_loss_d
            self.loss_step3_d = self.feature_discriminator.loss_d_feature + self.feature_discriminator.class_loss_source
            varlist_g = self.vars_feature_classifier + self.vars_encoder_source + self.vars_encoder_target + self.vars_decoder_source + self.vars_decoder_target
            varlist_d = self.feature_discriminator.vars_d + self.vars_feature_classifier
            self.optimizer_step3_g = tf.train.GradientDescentOptimizer(learning_rate=0.001, name="optimize_3_g").minimize(self.loss_step3_g, var_list=varlist_g)
            self.optimizer_step3_d = tf.train.GradientDescentOptimizer(learning_rate=0.001, name="optimize_3_d").minimize(self.loss_step3_d, var_list=varlist_d)
        with tf.name_scope("Step4"):
            self.loss_step4_g = 10 * self.loss_fc_source + \
                self.source_autoencoder.loss + \
                self.target_autoencoder.loss + \
                self.feature_discriminator.loss_g_feature + \
                self.image_discriminator_source.loss_g_feature + \
                self.image_discriminator_target.loss_g_feature
            self.loss_step4_d = self.feature_discriminator.loss_d_feature + \
                self.feature_discriminator.class_loss_source + \
                self.image_discriminator_source.loss_d_feature + \
                self.image_discriminator_source.class_loss_real + \
                self.image_discriminator_target.loss_d_feature + \
                self.image_discriminator_target.class_loss_fake
            varlist_g_4 = self.vars_feature_classifier + \
                self.vars_encoder_source + \
                self.vars_encoder_target + \
                self.vars_decoder_source + \
                self.vars_decoder_target 
            varlist_d_4 = self.feature_discriminator.vars_d + \
                self.image_discriminator_source.vars_d + \
                self.image_discriminator_target.vars_d
                 #+ self.vars_feature_classifier
            self.optimizer_step4_g = tf.train.GradientDescentOptimizer(learning_rate=0.001, name="optimize_4_g").minimize(self.loss_step4_g, var_list=varlist_g_4)
            self.optimizer_step4_d = tf.train.GradientDescentOptimizer(learning_rate=0.001, name="optimize_4_d").minimize(self.loss_step4_d, var_list=varlist_d_4)

        with tf.name_scope("Step5"):
            self.loss_step5_g = self.loss_step4_g + self.total_feedback + self.loss_semantic
            self.loss_step5_d = self.loss_step4_d + self.total_feedback + self.loss_semantic
            
            varlist_g_5 = varlist_g_4
            varlist_d_5 = varlist_d_4
            
            self.optimizer_step5_g = tf.train.GradientDescentOptimizer(learning_rate=0.001, name="optimize_5_g").minimize(self.loss_step5_g, var_list=varlist_g_5)
            self.optimizer_step5_d = tf.train.GradientDescentOptimizer(learning_rate=0.001, name="optimize_5_d").minimize(self.loss_step5_d, var_list=varlist_d_5)
    def duplicate_source_ae_to_target_ae(self):
        print("[Info] Duplicate source ae to target ae")
        vars_source = self.vars_encoder_source + self.vars_decoder_source
        vars_target = self.vars_encoder_target + self.vars_decoder_target
        for v_s, v_t in zip(vars_source, vars_target):
            self.sess.run(v_t.assign(v_s))
    def collect_feature(self, batch_source, batch_target, source_label, step):
        spe_source, com_source = self.source_autoencoder.get_split_feature(batch_source, self.sess)
        spe_target, com_target = self.target_autoencoder.get_split_feature(batch_target, self.sess)
        self.feature = np.concatenate((self.feature, spe_source), axis=0)
        self.meta_data = np.concatenate((self.meta_data, np.zeros(len(spe_source))))
        self.feature = np.concatenate((self.feature, com_source), axis=0)
        self.meta_data = np.concatenate((self.meta_data, np.ones(len(com_source))))

        self.feature = np.concatenate((self.feature, spe_target), axis=0)
        self.meta_data = np.concatenate((self.meta_data, 2 * np.ones(len(spe_target))))
        self.feature = np.concatenate((self.feature, com_target), axis=0)
        self.meta_data = np.concatenate((self.meta_data, 3 * np.ones(len(com_target))))
        print(len(self.feature))

    def visualize_feature(self):
        embedding_var = tf.Variable(tf.zeros([len(self.feature),128], dtype=tf.float64), name="tsne")
        meta_file = os.path.join(self.embedding_dir, 'meta.csv')

        if not os.path.exists(self.embedding_dir):
            os.system("mkdir -p {}".format(self.embedding_dir))
        print(self.embedding_dir)
        with open(meta_file, "w") as f:
            for l in self.meta_data:
                f.write("{}\n".format(l))

        # Create summary writer.
        writer = tf.summary.FileWriter(self.embedding_dir, self.sess.graph)
        # Initialize embedding_var
        assign_op = tf.assign(embedding_var, tf.reshape(self.feature, (-1, 128)))
        self.sess.run(assign_op)

        config = projector.ProjectorConfig()
        # Add embedding visualizer
        embedding = config.embeddings.add()
        # Attache the name 'embedding'
        embedding.tensor_name = embedding_var.name
        # Metafile which is described later
        embedding.metadata_path = 'meta.csv'
        # Add writer and config to Projector
        projector.visualize_embeddings(writer, config)
        # Save the model
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(self.sess, os.path.join(self.embedding_dir, 'embedding.ckpt'), 1)
        writer.close()
    def tsne_sklearn(self):
        x = np.reshape(self.feature, (-1, 128))
        X_embedded = TSNE(n_components=2, learning_rate=200, n_iter=5000).fit_transform(x)
        # print(X_embedded.shape)
        # print(np.array(self.meta_data).shape)
        # X_embedded = np.reshape(X_embedded, (-1, 128))
        plot_embedding(X_embedded, self.meta_data)
        plt.savefig("tsne.png")
    def _construct_optimizer(self):
        pass
        
    
    def _construct_feedback_loss(self, gen_img, spe_latent, com_latent, autoencoder):
        print("[Info] Construct feedback {}".format(autoencoder.name))
        with tf.variable_scope(autoencoder.encoder_scope, reuse=tf.AUTO_REUSE) as scope:
            with tf.name_scope(scope.original_name_scope):
                feature_feedback = autoencoder.encoder(gen_img)
                spe, com = tf.split(feature_feedback, num_or_size_splits=2, axis=3)
        with tf.variable_scope("loss_L2_{}".format(autoencoder.name)):
            loss_spe = tf.losses.mean_squared_error(spe_latent, spe)
            loss_com = tf.losses.mean_squared_error(com_latent, com)
            loss_fea = loss_spe + loss_com
                
        # with tf.variable_scope(autoencoder.decoder_scope, reuse=tf.AUTO_REUSE) as scope:
        #     with tf.name_scope(scope.original_name_scope):
        #         rec_feed_img = autoencoder.decoder(feature_feedback)
                
        # with tf.variable_scope("loss_rec_feed_{}".format(self.name)):
        #     loss_rec_feed = tf.losses.mean_squared_error(gen_img, rec_feed_img, scope="loss_{}".format(self.name))
            
        return loss_fea#, loss_rec_feed
        
    def _construct_summary(self):
        tf.summary.image("spe_source_com_target", self.img_spe_source_com_target, 3)
        tf.summary.image("spe_target_com_source", self.img_spe_target_com_source, 3)
        tf.summary.image("reconstruct_target_data", self.reconstruct_source_target_data, 3)
        tf.summary.scalar('loss_reconstruct', self.loss_reconstruct)
        tf.summary.scalar("source_reconstruct_target_data", self.loss_reconstruct_source_img_target)
        tf.summary.scalar("loss_semantic", self.loss_semantic)
        tf.summary.scalar("loss_entropy", self.loss_entropy)
        with tf.name_scope('feedback'):
            tf.summary.scalar("feedback_L2_source", self.feedback_L2_source)
            tf.summary.scalar("feedback_L2_target", self.feedback_L2_target)
            tf.summary.scalar("total_feedback", self.total_feedback)
        
        with tf.name_scope('feature_classifier'):
            tf.summary.scalar("loss_fc_source", self.loss_fc_source)
            tf.summary.scalar("loss_fc_target", self.loss_fc_target)
            tf.summary.scalar("acc_fc_source", self.fc_accuracy_source)
            tf.summary.scalar("acc_fc_target", self.fc_accuracy_target)
        
    def merge_all(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        self.sess.run(init_g)
        self.sess.run(init_l)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
    def set_logdir(self, logdir):
        self.logdir = logdir
        self.embedding_dir = os.path.join(self.logdir, 'common')
        self.train_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        
    def _feed_dict(self, batch_source, batch_target, source_label, target_label):
        return {
            self.source_autoencoder.ae_inputs: batch_source, 
            self.target_autoencoder.ae_inputs: batch_target,
            self.source_label: source_label,
            self.target_label: target_label
        }

    def run_step1(self, batch_source, batch_target, source_label, target_label, step):
        summary, loss, _ = self.sess.run(
            [self.merged, self.loss_step1, self.optimizer_step1],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label, target_label)
        )
        print("Iter {}: loss step1: {:.4f}".format(step, loss))
        self.train_writer.add_summary(summary, step)
        
    def run_step2(self, batch_source, batch_target, source_label, target_label, step):
        summary, loss_1, _ = self.sess.run(
            [self.merged, self.loss_step2, self.optimizer_step2],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label, target_label)
        )
        print("Iter {}: loss step2: {:.4f}".format(step, loss_1))
        self.train_writer.add_summary(summary, step)
    
    def run_step3(self, batch_source, batch_target, source_label, target_label, step):
        summary, loss_g, loss_d, _, _= self.sess.run(
            [self.merged, self.loss_step3_g, self.loss_step3_d, self.optimizer_step3_g, self.optimizer_step3_d],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label, target_label)
        )
        print("Iter {}: loss step3 g: {:.4f}, loss step3 d: {:.4f}".format(step, loss_g, loss_d))
        self.train_writer.add_summary(summary, step)

    def run_step4(self, batch_source, batch_target, source_label, target_label, step):
        summary, loss_g, loss_d, _, _= self.sess.run(
            [self.merged, self.loss_step4_g, self.loss_step4_d, self.optimizer_step4_g, self.optimizer_step4_d],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label, target_label)
        )
        print("Iter {}: loss step4 g: {:.4f}, loss step4 d: {:.4f}".format(step, loss_g, loss_d))
        self.train_writer.add_summary(summary, step)
        
    def run_step5(self, batch_source, batch_target, source_label, target_label, step):
        summary, loss_g, loss_d, _, _= self.sess.run(
            [self.merged, self.loss_step5_g, self.loss_step5_d, self.optimizer_step5_g, self.optimizer_step5_d],
            feed_dict=self._feed_dict(batch_source, batch_target, source_label, target_label)
        )
        print("Iter {}: loss step5 g: {:.4f}, loss step5 d: {:.4f}".format(step, loss_g, loss_d))
        self.train_writer.add_summary(summary, step)
    
    