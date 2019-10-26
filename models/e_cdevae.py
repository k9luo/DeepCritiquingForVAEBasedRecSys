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
                 observation_std=0.01,
                 log_dir='./tensorboard'):

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
        self.sess = tf.Session()
        self.train_writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):

        with tf.variable_scope('vae'):
            self.rating_input = tf.placeholder(tf.float32, shape=[None, self._observation_dim])
            self.keyphrase_input = tf.placeholder(tf.float32, shape=[None, self._keyphrase_dim])
            self.corruption = tf.placeholder(tf.float32)
            self.sampling = tf.placeholder(tf.bool)
            # modified_keyphrase dimension change from rating to keyphrase
            self.modified_keyphrase = tf.placeholder(tf.float32, [None, self._keyphrase_dim], name='modified_keyphrases')

            mask1 = tf.nn.dropout(tf.ones_like(self.rating_input), 1 - self.corruption)

            wc = self.rating_input * mask1

            with tf.variable_scope('encoder'):
                encoded = tf.layers.dense(inputs=wc, units=self._latent_dim*2,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._lamb),
                                          activation=None, name="Encoder_Weights")

            with tf.variable_scope('latent'):
                self.mean = tf.nn.relu(encoded[:, :self._latent_dim])
                # TODO: Might worth trying adding tanh as activation function
                # for variance
                logstd = encoded[:, self._latent_dim:]
                self.stddev = tf.exp(logstd)
                epsilon = tf.random_normal(tf.shape(self.stddev))
                self.z = tf.cond(self.sampling, lambda: self.mean + self.stddev * epsilon, lambda: self.mean)

            latent = tf.stop_gradient(tf.concat([self.mean, logstd], axis=1))

            with tf.variable_scope("prediction", reuse=False):
                rating_prediction = tf.layers.dense(inputs=self.z, units=self._observation_dim,
                                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._lamb),
                                                    activation=None, name='rating_prediction')

                keyphrase_prediction = tf.layers.dense(inputs=self.z, units=self._keyphrase_dim,
                                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._lamb),
                                                       activation=None, name='keyphrase_prediction')

                self.rating_prediction = rating_prediction
                self.keyphrase_prediction = keyphrase_prediction

            # looping with keyphrase
            with tf.variable_scope("looping"):
                reconstructed_latent = tf.layers.dense(inputs=self.keyphrase_prediction, units=self._latent_dim*2,
                                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._lamb),
                                                       activation=None, name='latent_reconstruction', reuse=False)

                modified_latent = tf.layers.dense(inputs=self.modified_keyphrase, units=self._latent_dim*2,
                                                  activation=None, name='latent_reconstruction', reuse=True)

                self.modified_mean = tf.nn.relu(modified_latent)[:, :self._latent_dim]
                modified_latent = (latent + tf.nn.relu(modified_latent)) / 2.0
                modified_mean = modified_latent[:, :self._latent_dim]

            with tf.variable_scope("prediction", reuse=True):
                rating_prediction = tf.layers.dense(inputs=modified_mean, units=self._observation_dim,
                                                    activation=None, name='rating_prediction')
                keyphrase_prediction = tf.layers.dense(inputs=modified_mean, units=self._keyphrase_dim,
                                                       activation=None, name='keyphrase_prediction')

                self.modified_rating_prediction = rating_prediction
                self.modified_keyphrase_prediction = keyphrase_prediction

            with tf.variable_scope('loss'):
                with tf.variable_scope('kl-divergence'):
                    kl = self._kl_diagnormal_stdnormal(self.mean, logstd)

                with tf.variable_scope("latent_reconstruction_loss"):
                    latent_loss = tf.losses.mean_squared_error(labels=latent,
                                                               predictions=reconstructed_latent)

                """
                # For rating loss, we can also try sigmoid cross-entropy loss.
                with tf.variable_scope("rating_decoder_reconstruction_loss"):
                    rating_loss = tf.losses.mean_squared_error(labels=self.rating_input,
                                                                predictions=self.rating_prediction)

                with tf.variable_scope("keyphrase_decoder_reconstruction_loss"):
                    keyphrase_loss = tf.losses.mean_squared_error(labels=self.keyphrase_input,
                                                                  predictions=self.keyphrase_prediction)
                """

                if self._observation_distribution == 'Gaussian':
                    with tf.variable_scope('gaussian'):
                        rating_obj = self._gaussian_log_likelihood(self.rating_input, self.rating_prediction, self._observation_std)
                        keyphrase_obj = self._gaussian_log_likelihood(self.keyphrase_input, self.keyphrase_prediction, self._observation_std)
                elif self._observation_distribution == 'Bernoulli':
                    with tf.variable_scope('bernoulli'):
                        rating_obj = self._bernoulli_log_likelihood(self.rating_input, self.rating_prediction)
                        keyphrase_obj = self._bernoulli_log_likelihood(self.keyphrase_input, self.keyphrase_prediction)
                else:
                    with tf.variable_scope('multinomial'):
                        rating_obj = self._multinomial_log_likelihood(self.rating_input, self.rating_prediction)
                        keyphrase_obj = self._multinomial_log_likelihood(self.keyphrase_input, self.keyphrase_prediction)
                obj = rating_obj + keyphrase_obj

                with tf.variable_scope('l2'):
                    l2_loss = tf.losses.get_regularization_loss()

                # TODO Loss function Tuning
                self._loss = (obj
                              + 5 * tf.reduce_mean(latent_loss)
                              + self._beta * kl
                              + self._lamb * l2_loss
                              )

                """
                self._loss = (5 * tf.reduce_mean(rating_loss)
                              + 5 * tf.reduce_mean(keyphrase_loss)
                              + 5 * tf.reduce_mean(latent_loss)
                              + self._beta * kl
                              + self._lamb * l2_loss
                              )
                """
            #summary list
            with tf.name_scope('loss_set'):
                tf.summary.scalar('kl_loss', kl)
                tf.summary.scalar('l2_loss', l2_loss)
                tf.summary.scalar('latent_loss', latent_loss)
                tf.summary.scalar('observation_loss', rating_obj)
                tf.summary.scalar('keyphrase_loss',keyphrase_obj)
                tf.summary.scalar('total_loss', self._loss)
            self.merged = tf.summary.merge_all()

            with tf.variable_scope('optimizer'):
                optimizer = self._optimizer(learning_rate=self._learning_rate)

            with tf.variable_scope('training-step'):
                self._train = optimizer.minimize(self._loss)

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

    def predict(self, rating_input):
        return self.sess.run([self.rating_prediction,
                              self.keyphrase_prediction],
                             feed_dict={self.rating_input: rating_input,
                                        self.corruption: 0,
                                        self.sampling: False})

    def uncertainty(self, rating_input):
        gaussian_parameters = self.sess.run([self.mean, self.stddev],
                                            feed_dict={self.rating_input: rating_input,
                                                       self.corruption: 0,
                                                       self.sampling: False})

        return gaussian_parameters

    def refined_predict(self, rating_input, critiqued):
        modified_rating, modified_keyphrases = self.sess.run([self.modified_rating_prediction,
                                                              self.modified_keyphrase_prediction],
                                                             feed_dict={self.rating_input: rating_input,
                                                                        self.corruption: 0,
                                                                        self.sampling: False,
                                                                        self.modified_keyphrase: critiqued})

        return modified_rating, modified_keyphrases

    def train_model(self, rating_matrix, keyphrase_matrix, corruption, epoch=100, batches=None, **unused):
        #TODO is pretrained batch needed for training?
        if batches is None:
            batches = self.get_batches(rating_matrix, self._batch_size)
            batches_keyphrase = self.get_batches(keyphrase_matrix, self._batch_size)
        # Training
        pbar = tqdm(range(epoch))
        for i in pbar:
            for step in range(len(batches)):
                feed_dict = {self.rating_input: batches[step].todense(),
                             self.keyphrase_input:batches_keyphrase[step],
                             self.corruption: corruption,
                             self.sampling: True}
                # write summary
                summary, training, loss = self.sess.run([self.merged, self._train, self._loss], feed_dict=feed_dict)
                self.train_writer.add_summary(summary, i)
                pbar.set_description("loss:{}".format(loss))

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


def e_cde_vae(matrix_train, matrix_train_keyphrase, embeded_matrix=np.empty((0)), epoch=100, lamb=80,
            learning_rate=0.0001, rank=200, corruption=0.5, optimizer="RMSProp", seed=1, **unused):
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    matrix_input_keyphrase = matrix_train_keyphrase

    model = E_CDE_VAE(observation_dim=matrix_input.shape[1], keyphrase_dim=matrix_input_keyphrase.shape[1],
                      latent_dim=rank, batch_size=128, lamb=lamb, learning_rate=learning_rate,
                      observation_distribution="Gaussian", optimizer=Regularizer[optimizer])

    model.train_model(matrix_input, matrix_input_keyphrase, corruption, epoch)

    return model

