from consumers.consumer import Consumer
import tensorflow as tf
# import tensorflow.compat.v1 as tf 
# tf.disable_v2_behavior()
import numpy as np
from tensorflow.distributions import kl_divergence, Normal


class ReparamLoss(Consumer):
    def __init__(self, embedding_size, loss_lambda=1.0,
                 clip_sigma=True, use_prior=True, support_sampling=True, uncert_type='logsigma', kl_type='support', embedding_type="both", sentence_type='both', reparam_embedding_type=None):
        self.loss_lambda = loss_lambda
        self.embedding_size = embedding_size
        self.loss_embedding = None
        self.loss_kl = None
        # self.loss_kl_image = None
        # self.loss_kl_q = None
        # self.loss_kl_language = None
        self.embedding_accuracy = None
        self.scale = None
        self.clip_sigma = clip_sigma
        self.use_prior = use_prior
        self.support_sampling = support_sampling
        self.uncert_type = uncert_type
        self.sentence_type=sentence_type
        self.embedding_type=embedding_type
        self.reparam_embedding_type=reparam_embedding_type


    # Input shape: (batch, num_distributions, emb_size)
    def _poe(self, mus, sigmas, eps=1e-8):
        vars = tf.square(sigmas) + eps  # vars = (sigmas)^2 {eps省略} 
        var = tf.reciprocal(tf.reduce_sum(tf.reciprocal(vars), axis=1)) + eps  # var = 1/(sum(1/(sigmas)^2 ) 
        mu = tf.reduce_sum(mus / vars, axis=1) * var  # mu = sum(mus/(sigmas)^2)
        sigma = tf.sqrt(var)  # sigma =  root(1/(sum(1/(sigmas)^2 ))
        return mu, sigma

    # Input shape: (batch, num_distributions, emb_size)
    def _poe_beta(self, mus, betas, eps=1e-8):
        var = tf.reciprocal(tf.reduce_sum(betas, axis=1)) + eps
        mu = tf.reduce_sum(mus * betas, axis=1) * var
        sigma = tf.sqrt(var)
        return mu, sigma

    def _reparam(self, mu, sigma):
        epsilon = tf.random_normal(tf.shape(sigma))
        return mu + epsilon * sigma    # Reparametrization trick

    def _clip_sigma(self, logsigma):
        return 0.1 + 0.9 * tf.sigmoid(logsigma)

    def consume(self, inputs):
        # support => s_normal
        semb, semb_uncert = tf.split(self.get(inputs, 'support_embedding'),
                                     num_or_size_splits=[self.embedding_size, self.embedding_size], axis=2)
        batch_size = tf.shape(semb)[0]
        
        _mus = tf.concat([tf.zeros([batch_size, 1, self.embedding_size]), semb], axis=1)
        _sigmas = tf.concat([tf.ones([batch_size, 1, self.embedding_size]), self._clip_sigma(semb_uncert)], axis=1)
        s_mu, s_sigma = self._poe(_mus, _sigmas)
        p = Normal(s_mu, s_sigma)
        self.scale = s_sigma
        
        # query => q_normal
        qemb, qemb_uncert = tf.split(self.get(inputs, 'query_embedding'),
                                     num_or_size_splits=[self.embedding_size, self.embedding_size], axis=2)  # self.embedding_size//2している  
        q_mu, q_sigma = self._poe(tf.concat([_mus, qemb], axis=1), tf.concat([_sigmas, self._clip_sigma(qemb_uncert)], axis=1))
        q = Normal(q_mu, q_sigma)
    
        if self.embedding_type == "both_embedding":
            # image => image_normal
            imaemb, imaemb_uncert = tf.split(self.get(inputs, 'support_image_embedding'),
                                             num_or_size_splits=[self.embedding_size, self.embedding_size], axis=2)
            ima_mus = tf.concat([tf.zeros([batch_size, 1, self.embedding_size]), imaemb], axis=1)
            ima_sigmas = tf.concat([tf.ones([batch_size, 1, self.embedding_size]), self._clip_sigma(imaemb_uncert)], axis=1)
            image_mu, image_sigma = self._poe(ima_mus, ima_sigmas)
            p_image = Normal(image_mu, image_sigma)
            
            # language => language_normal
            insemb, insemb_uncert = tf.split(self.get(inputs, 'support_language_embedding'),
                                             num_or_size_splits=[self.embedding_size, self.embedding_size], axis=2)
            ins_mus = tf.concat([tf.zeros([batch_size, 1, self.embedding_size]), insemb], axis=1)
            ins_sigmas = tf.concat([tf.ones([batch_size, 1, self.embedding_size]), self._clip_sigma(insemb_uncert)], axis=1)
            language_mu, language_sigma = self._poe(ins_mus, ins_sigmas)
            p_language = Normal(language_mu, language_sigma)

            if self.sentence_type == 'language':
                inputs['sentences'] = tf.cond(inputs['training'], lambda: self._reparam(s_mu, s_sigma), lambda: language_mu)
            elif self.sentence_type == 'image':
                inputs['sentences'] = tf.cond(inputs['training'], lambda: self._reparam(s_mu, s_sigma), lambda: image_mu)
            else:
                inputs['sentences'] = tf.cond(inputs['training'], lambda: self._reparam(s_mu, s_sigma), lambda: s_mu)
        else:
            inputs['sentences'] = tf.cond(inputs['training'], lambda: self._reparam(s_mu, s_sigma), lambda: s_mu)
        inputs['scale'] = s_sigma
        
        # sampling task variable from query embedding during training (from support embedding during test)
        kl = tf.reduce_sum(kl_divergence(q, p), axis=1)
        kl = tf.reduce_mean(kl)
        
        if self.embedding_type == "both_embedding":
            # image部分のkl
            kl_image = tf.reduce_sum(kl_divergence(q, p_image), axis=1)
            kl_image = tf.reduce_mean(kl_image)

            # language部分のkl
            kl_language = tf.reduce_sum(kl_divergence(q, p_language), axis=1)
            kl_language = tf.reduce_mean(kl_language)

            if self.reparam_embedding_type == 'kl':
                self.loss_embedding = self.loss_lambda * (kl_image + kl_language + kl)
            elif self.reparam_embedding_type == 'kl_2image':
                self.loss_embedding = self.loss_lambda * (2 * kl_image + kl_language + kl)
            elif self.reparam_embedding_type == '2image':
                self.loss_embedding = self.loss_lambda * (2 * kl_image + kl_language)
            else:
                self.loss_embedding = self.loss_lambda * (kl_image + kl_language)
        else:
            self.loss_embedding = self.loss_lambda * kl
        
        # Summaries
        inputs['loss_embedding'] = self.loss_embedding
        return inputs
    
    def get_summaries(self, prefix):
        return [tf.summary.scalar(prefix + 'loss_embedding',
                                  self.verify(self.loss_embedding)),
                tf.summary.scalar(prefix + 'mean_scale',
                                  self.verify(tf.reduce_mean(self.scale))),
                tf.summary.scalar(prefix + 'max_scale',
                                  self.verify(tf.reduce_max(self.scale))),
                tf.summary.scalar(prefix + 'min_scale',
                                  self.verify(tf.reduce_min(self.scale)))]
        
    def get_loss(self):
        return self.verify(self.loss_embedding)
