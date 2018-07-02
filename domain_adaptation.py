import tensorflow as tf
import tensorflow.layers as lays
import numpy as np


class DomainAdaptation:
    def __init__(self, source_autoencoder, target_autoencoder):
        self.source_autoencoder = source_autoencoder
        self.target_autoencoder = target_autoencoder
        self._construct_graph()
        
    def _construct_graph(self):
        source_specific_latent = self.source_autoencoder.specific
        source_common_latent = self.source_autoencoder.common
        target_specific_latent = self.target_autoencoder.specific
        target_common_latent = self.target_autoencoder.common

        latent_transfer_to_source = tf.concat([source_specific_latent, target_common_latent], 3)
        latent_transfer_to_target = tf.concat([target_specific_latent, source_common_latent], 3)
        
        gen_transfer_to_source = self.source_autoencoder.decoder(latent_transfer_to_source)
        

        self.source_pure = self.source_autoencoder.ae_outputs