import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.contrib.distributions import Bernoulli
from utils.progress import WorkSplitter, inhour
from scipy.sparse import vstack, hstack
from utils.regularizers import Regularizer

#test
class E_CDE_VAE(object):

    def __init__(self, observation_dim, keyphrase_dim, latent_dim, batch_size,
                 lamb=0.01,
                 beta=0.2,
                 learning_rate=1e-4,
                 optimizer=tf.train.RMSPropOptimizer,
                 observation_distribution="Multinomial", # or Gaussian or Bernoulli
                 observation_std=0.01):

        self._lamb = lamb
        self._beta = beta
        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._observation_dim = observation_dim
        self._keyphrase_dim = keyphrase_dim
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._observation_distribution = observation_distribution
        self._observation_std = observation_std
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('vae'):
            self.obs_input = tf.placeholder(tf.float32, shape=[None, self._observation_dim])
            self.kp_input = tf.placeholder(tf.float32, shape=[None, self._keyphrase_dim])
            self.corruption = tf.placeholder(tf.float32)
            self.sampling = tf.placeholder(tf.bool)
            # modified_predict dimension change from obs to keyphrase
            self.modified_predict = tf.placeholder(tf.float32, [None, self._keyphrase_dim], name='modified_predict')

            mask1 = tf.nn.dropout(tf.ones_like(self.obs_input), 1 - self.corruption)

            wc = self.obs_input * mask1

            with tf.variable_scope('encoder'):
                encode_weights = tf.Variable(tf.truncated_normal([self._observation_dim, self._latent_dim*2],
                                                                 stddev=1 / 500.0),
                                             name="Weights")
                encode_bias = tf.Variable(tf.constant(0., shape=[self._latent_dim*2]), name="Bias")

                encoded = tf.matmul(wc, encode_weights) + encode_bias

            with tf.variable_scope('latent'):
                self.mean = tf.nn.relu(encoded[:, :self._latent_dim])
                logstd = encoded[:, self._latent_dim:]
                self.stddev = tf.exp(logstd)
                epsilon = tf.random_normal(tf.shape(self.stddev))
                self.z = tf.cond(self.sampling, lambda: self.mean + self.stddev * epsilon, lambda: self.mean)

            latent = tf.stop_gradient(tf.concat([self.mean, logstd], axis=1))

            with tf.variable_scope('decoder'):

                self.decode_weights = tf.Variable(
                    tf.truncated_normal([self._latent_dim, self._observation_dim], stddev=1 / 500.0),
                    name="Weights")
                self.decode_bias = tf.Variable(tf.constant(0., shape=[self._observation_dim]), name="Bias")
                decoded = tf.matmul(self.z, self.decode_weights) + self.decode_bias

                self.obs_mean = decoded

            #keyphrase decoder
            with tf.variable_scope('keyphrase_decoder'):

                self.kp_decode_weights = tf.Variable(
                    tf.truncated_normal([self._latent_dim, self._keyphrase_dim], stddev=1 / 500.0),
                    name="Keyphrase_Weights")
                self.kp_decode_bias = tf.Variable(tf.constant(0., shape=[self._keyphrase_dim]), name="keyphrase_Bias")
                kp_decoded = tf.matmul(self.z, self.kp_decode_weights) + self.kp_decode_bias

                self.kp_mean = kp_decoded

            #looping with keyphrase
            with tf.variable_scope("looping"):
                reconstructed_latent = tf.layers.dense(inputs=self.kp_mean, units=self._latent_dim*2,
                                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._lamb),
                                                       activation=None, name='latent_reconstruction', reuse=False)

                modified_latent = tf.layers.dense(inputs=self.modified_predict, units=self._latent_dim*2,
                                                  activation=None, name='latent_reconstruction', reuse=True)

                self.modified_mean = tf.nn.relu(modified_latent)[:, :self._latent_dim]
                modified_latent = (latent + tf.nn.relu(modified_latent)) / 2.0
                modified_mean = modified_latent[:, :self._latent_dim]

            with tf.variable_scope('decoder'):
                self.modified_decoded = tf.matmul(modified_mean, self.decode_weights) + self.decode_bias

            with tf.variable_scope('loss'):
                with tf.variable_scope('kl-divergence'):
                    kl = self._kl_diagnormal_stdnormal(self.mean, logstd)

                with tf.variable_scope("latent_reconstruction_loss"):
                    latent_loss = tf.losses.mean_squared_error(labels=latent,
                                                               predictions=reconstructed_latent)

                """
                with tf.variable_scope("obs_decoder_reconstruction_loss"):
                    obs_decoder_loss = tf.losses.mean_squared_error(labels=self.obs_input,
                                                                predictions=self.obs_mean)
                """
                with tf.variable_scope("kp_decoder_reconstruction_loss"):
                    kp_decoder_loss = tf.losses.mean_squared_error(labels=self.kp_input,
                                                                predictions=self.kp_mean)



                if self._observation_distribution == 'Gaussian':
                    with tf.variable_scope('gaussian'):
                        obj = self._gaussian_log_likelihood(self.obs_input, self.obs_mean, self._observation_std)
                elif self._observation_distribution == 'Bernoulli':
                    with tf.variable_scope('bernoulli'):
                        obj = self._bernoulli_log_likelihood(self.obs_input, self.obs_mean)
                else:
                    with tf.variable_scope('multinomial'):
                        obj = self._multinomial_log_likelihood(self.obs_input, self.obs_mean)

                with tf.variable_scope('l2'):
                    l2_loss = tf.reduce_mean(tf.nn.l2_loss(encode_weights) + tf.nn.l2_loss(self.decode_weights))
                
                #TODO Loss function Tuning
                self._loss = self._beta * kl + obj + self._lamb * l2_loss + 10 * tf.reduce_mean(latent_loss) + kp_decoder_loss

#                self._loss = self._beta * kl + tf.reduce_mean(decoder_loss) + self._lamb * l2_loss + tf.reduce_mean(latent_loss)

            with tf.variable_scope('optimizer'):
                optimizer = self._optimizer(learning_rate=self._learning_rate)
            with tf.variable_scope('training-step'):
                self._train = optimizer.minimize(self._loss)

            self.sess = tf.Session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    @staticmethod
    def _kl_diagnormal_stdnormal(mu, log_std):

        var_square = tf.exp(2 * log_std)
        kl = 0.5 * tf.reduce_mean(tf.square(mu) + var_square - 1. - 2 * log_std)
        return kl

    @staticmethod
    def _gaussian_log_likelihood(targets, mean, std):
        se = 0.5 * tf.reduce_mean(tf.square(targets - mean)) / (2*tf.square(std)) + tf.log(std)
        return se

    @staticmethod
    def _bernoulli_log_likelihood(targets, outputs, eps=1e-8):

        log_like = -tf.reduce_mean(targets * tf.log(outputs + eps)
                                   + (1. - targets) * tf.log((1. - outputs) + eps))
        return log_like

    @staticmethod
    def _multinomial_log_likelihood(target, outputs, eps=1e-8):
        log_softmax_output = tf.nn.log_softmax(outputs)
        log_like = -tf.reduce_mean(tf.reduce_sum(log_softmax_output * target, axis=1))
        return log_like

    def inference(self, x):
        predict = self.sess.run(self.obs_mean,
                                 feed_dict={self.obs_input: x, self.corruption: 0, self.sampling: False})
        return predict

    def uncertainty(self, x):
        gaussian_parameters = self.sess.run([self.mean, self.stddev],
                                             feed_dict={self.obs_input: x, self.corruption: 0, self.sampling: False})

        return gaussian_parameters

    def critiquing(self, x, modified):
        predict = self.sess.run(self.modified_decoded,
                                 feed_dict={self.obs_input: x, self.corruption: 0,
                                            self.sampling: False, self.modified_predict: modified})

        return predict

    def train_model(self, rating_matrix, keyphrase_matrix, corruption, epoch=100, batches=None, **unused):
        #TODO is pretrained batch needed for training?
        if batches is None:
            batches = self.get_batches(rating_matrix, self._batch_size)
            batches_kp = self.get_batches(keyphrase_matrix, self._batch_size)
        # Training
        pbar = tqdm(range(epoch))
        for i in pbar:
            for step in range(len(batches)):
                feed_dict = {self.obs_input: batches[step].todense(), self.kp_input:batches_kp[step].todense(), self.corruption: corruption, self.sampling: True}
                training = self.sess.run([self._train], feed_dict=feed_dict)

    def get_batches(self, matrix, batch_size):
        remaining_size = matrix.shape[0]
        batch_index = 0
        batches = []
        while remaining_size > 0:
            if remaining_size < batch_size:
                batches.append(matrix[batch_index*batch_size:])
            else:
                batches.append(matrix[batch_index*batch_size:(batch_index+1)*batch_size])
            batch_index += 1
            remaining_size -= batch_size
        return batches

    def get_RQ(self, rating_matrix):
        batches = self.get_batches(rating_matrix, self._batch_size)
        RQ = []
        for step in range(len(batches)):
            feed_dict = {self.obs_input: batches[step].todense(), self.corruption: 0, self.sampling: False}
            embedding = self.sess.run(self.z, feed_dict=feed_dict)
            RQ.append(embedding)

        return np.vstack(RQ)

    def get_Y(self):
        return self.sess.run(self.decode_weights)

    def get_Bias(self):
        return self.sess.run(self.decode_bias)


def e_cde_vae(matrix_train, matrix_train_keyphrase, embeded_matrix=np.empty((0)), epoch=100, lamb=80,
            learning_rate=0.0001, rank=200, corruption=0.5, optimizer="RMSProp", seed=1, **unused):
    progress = WorkSplitter()
    matrix_input = matrix_train
    matrix_input_kp = matrix_train_keyphrase
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape
    _, k = matrix_input_kp.shape

    model = E_CDE_VAE(n, k, rank, 128, lamb=lamb, learning_rate=learning_rate, observation_distribution="Gaussian", optimizer=Regularizer[optimizer])

    model.train_model(matrix_input, matrix_input_kp, corruption, epoch)

    return model

