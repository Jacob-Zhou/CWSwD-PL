# -*- coding: utf-8 -*-
import tensorflow as tf
from .SegmentModel import SegmentModel
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import crf
from models import model_utils
import numpy as np


class BaselineModel(SegmentModel):
    '''
    Baseline models
    BiLSTM+CRF and Stacked BiLSTM+CRF
    '''
    def __init__(self, config, features, dropout_keep_prob, init_embeddings=None):
        """Constructor for BertModel.

        Args:
          config: `Config` 实例，用来描述模型.
          features: dict 用来向模型传递输入，和训练时所用的标签
          dropout_keep_prob: float32 Tensor of shape [] 用来传递dropout保留率.
          init_embedding: (optional) float32 np.ndarray of shape [vocab_size, embedding_size] 初始化一元字嵌入.
          bi_embedding: (optional) float32 np.ndarray of shape [bigram_size, embedding_size] 初始化二元字嵌入.
        """

        super(BaselineModel).__init__()
        input_ids = features["input_ids"]
        seq_length = features["seq_length"]
        label_ids = features["label_ids"]

        self.input_ids = input_ids
        self.label_ids = label_ids
        self.seq_length = seq_length

        # 对输入进行嵌入处理
        x, batch_size, feat_size = model_utils.input_embedding(
            input_ids, config, init_embeddings=init_embeddings)
        # 将嵌入后的数据展平处理
        x = tf.reshape(x, [batch_size, -1, feat_size * config.embedding_size])
        x = tf.nn.dropout(x, dropout_keep_prob)

        with tf.variable_scope('rnn'):
            (forward_output, backword_output), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=model_utils.multi_lstm_cell(
                    config.hidden_size, config.num_hidden_layers, dropout_keep_prob),
                cell_bw=model_utils.multi_lstm_cell(
                    config.hidden_size, config.num_hidden_layers, dropout_keep_prob),
                inputs=x,
                sequence_length=self.seq_length,
                dtype=tf.float32
            )
            output = tf.concat([forward_output, backword_output], axis=2)

        with tf.variable_scope('output'):
            scores = layers.fully_connected(
                inputs=output,
                num_outputs=config.num_classes,
                activation_fn=None
            )
            transition_param = tf.get_variable("transitions",
                                               [config.num_classes, config.num_classes])
            self.prediction, _ = crf.crf_decode(scores, transition_param, self.seq_length)

        # with tf.variable_scope('noise_correct'):
        #     pure_noise_matrix = tf.Variable(config.noise_matrix, dtype=tf.float32, name='noise_matrix', trainable=False)
        #     tf.logging.info(f"\n{config.noise_matrix}")
        
        #     eye_matrix = tf.Variable(np.eye(4), dtype=tf.float32, name='noise_matrix', trainable=False)
        
        #     rate = tf.Variable(np.ones([4, 1]), dtype=tf.float32, name='rate')
        
        #     norm_rate = tf.sigmoid(rate)
        #     noise_matrix = tf.broadcast_to(norm_rate, [4, 4]) * pure_noise_matrix + \
        #                    tf.broadcast_to((1 - norm_rate), [4, 4]) * eye_matrix
        #     candidate_label_num = tf.reduce_sum(self.label_ids, axis=2)
        #     part_label_mask = tf.expand_dims(tf.cast(
        #         tf.logical_and(candidate_label_num > 1, candidate_label_num < config.num_classes), dtype=tf.float32),
        #         axis=-1)
        #     scores = part_label_mask * tf.einsum("ji, blj->bli", noise_matrix, scores) + (1 - part_label_mask) * scores

        with tf.variable_scope('loss'):
            # crf
            if config.multitag:
                # 如果是多标签则使用 crf_multitag_log_likelihood
                self.label_ids = tf.cast(self.label_ids, dtype=tf.bool)
                self.log_likelihood, _ = model_utils.crf_multitag_log_likelihood(
                    scores, self.label_ids, self.seq_length, transition_param)
            else:
                self.log_likelihood, _ = crf.crf_log_likelihood(
                    scores, self.label_ids, self.seq_length, transition_param)

            self.loss = tf.reduce_mean(-self.log_likelihood)

