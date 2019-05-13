# -*- coding: utf-8 -*-
import tensorflow as tf

from models.SegmentModel import SegmentModel
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import crf

from models import model_utils


class AttendedDictModel(SegmentModel):
    '''
    Model 1
    Concating outputs of two parallel Bi-LSTMs which take feature vectors and uni_embedding vectors
    as inputs respectively.
    '''

    def __init__(self, config, features, dropout_keep_prob, init_embeddings=None):

        super(AttendedDictModel).__init__()
        input_ids = features["input_ids"]
        input_dicts = features["input_dicts"]
        seq_length = features["seq_length"]
        label_ids = features["label_ids"]

        self.label_ids = label_ids
        self.dict = input_dicts
        self.seq_length = seq_length

        dict_shape = model_utils.get_shape_list(input_dicts, expected_rank=3)
        self.dict_dim = dict_shape[2]

        x, batch_size, feat_size = model_utils.input_embedding(
            input_ids, config, init_embeddings=init_embeddings)
        x = tf.reshape(x, [batch_size, -1, feat_size * config.embedding_size])
        x = tf.nn.dropout(x, dropout_keep_prob)

        with tf.variable_scope('character'):
            (forward_output, backword_output), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=model_utils.multi_lstm_cell(config.hidden_size, config.num_hidden_layers, dropout_keep_prob),
                cell_bw=model_utils.multi_lstm_cell(config.hidden_size, config.num_hidden_layers, dropout_keep_prob),
                inputs=x,
                sequence_length=self.seq_length,
                dtype=tf.float32
            )
            output = tf.concat([forward_output, backword_output], axis=2)

        with tf.variable_scope('dict_attention'):
            dict_attention = layers.fully_connected(
                inputs=output,
                num_outputs=self.dict_dim,
                activation_fn=tf.sigmoid
            )
            # [B, L, D]
            self.dict = tf.cast(self.dict, dtype=tf.float32)
            attend_dict = tf.multiply(self.dict, dict_attention)

        with tf.variable_scope('dict'):
            (forward_output, backword_output), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=model_utils.multi_lstm_cell(config.hidden_size, config.num_hidden_layers, dropout_keep_prob),
                cell_bw=model_utils.multi_lstm_cell(config.hidden_size, config.num_hidden_layers, dropout_keep_prob),
                inputs=attend_dict,
                sequence_length=self.seq_length,
                dtype=tf.float32
            )
            dict_output = tf.concat([forward_output, backword_output], axis=2)

        with tf.variable_scope('output'):
            output = tf.concat([dict_output, output], axis=2)
            scores = layers.fully_connected(
                inputs=output,
                num_outputs=config.num_classes,
                activation_fn=None
            )
            transition_param = tf.get_variable("transitions", [config.num_classes, config.num_classes])
            self.prediction, _ = crf.crf_decode(scores, transition_param, self.seq_length)

        with tf.variable_scope('loss'):
            # crf
            if config.multitag:
                self.label_ids = tf.cast(self.label_ids, dtype=tf.bool)
                self.log_likelihood, _ = model_utils.crf_multitag_log_likelihood(
                    scores, self.label_ids, self.seq_length, transition_param)
            else:
                self.log_likelihood, _ = crf.crf_log_likelihood(
                    scores, self.label_ids, self.seq_length, transition_param)

            self.loss = tf.reduce_mean(-self.log_likelihood)
